import json
import os
import re
import base64
import hashlib
import time
import subprocess
from io import BytesIO
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request
from docx import Document
from pypdf import PdfReader


HOST = os.environ.get("LJ_AI_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT") or os.environ.get("LJ_AI_API_PORT", "8765"))
MAX_REQUEST_BYTES = int(os.environ.get("LJ_MAX_REQUEST_BYTES", str(8 * 1024 * 1024)))
MAX_UPLOAD_BYTES = int(os.environ.get("LJ_MAX_UPLOAD_BYTES", str(6 * 1024 * 1024)))
MAX_UPLOAD_TEXT_CHARS = int(os.environ.get("LJ_MAX_UPLOAD_TEXT_CHARS", "120000"))
REQUEST_WINDOW_SECONDS = int(os.environ.get("LJ_RATE_LIMIT_WINDOW_SECONDS", "60"))
UPLOAD_LIMIT_PER_WINDOW = int(os.environ.get("LJ_UPLOAD_LIMIT_PER_WINDOW", "12"))
AI_LIMIT_PER_WINDOW = int(os.environ.get("LJ_AI_LIMIT_PER_WINDOW", "30"))
ALLOW_ORIGIN = os.environ.get("LJ_ALLOW_ORIGIN", "*").strip() or "*"
UPLOAD_ALLOWED_EXTENSIONS = {".doc", ".docx", ".pdf", ".txt", ".md", ".json"}
UPLOAD_MAGIC_PREFIXES = {
    ".docx": [b"PK\x03\x04"],
    ".pdf": [b"%PDF"],
    ".json": [b"{", b"[", b"\xef\xbb\xbf{"],
    ".txt": [],
    ".md": [],
    ".doc": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],
}
REQUEST_BUCKETS = {}

PROVIDER_DEFAULTS = {
    "openai": {
        "label": "OpenAI",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "endpoint": "/chat/completions",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "endpoint": "/chat/completions",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    },
    "mimo": {
        "label": "MiMo",
        "base_url": os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        "endpoint": os.environ.get("MIMO_ENDPOINT", "/chat/completions"),
        "model": os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"),
        "api_key": os.environ.get("MIMO_API_KEY", ""),
    },
}

DIMENSION_ORDER = [
    "brand",
    "marketing",
    "production",
    "rd",
    "standard",
    "logistics",
    "capital",
    "finance",
]

DIMENSION_LABELS = {
    "brand": "国际化品牌",
    "marketing": "国际化营销",
    "production": "国际化制成",
    "rd": "国际化研发",
    "standard": "国际化标准与认证",
    "logistics": "国际化物流",
    "capital": "国际化资本",
    "finance": "国际化金融",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def client_ip_from_headers(handler):
    forwarded = handler.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if handler.client_address and handler.client_address[0]:
        return handler.client_address[0]
    return "unknown"


def mask_secret(value):
    value = str(value or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def parse_multipart_file(raw_bytes, content_type):
    match = re.search(r'boundary="?([^";]+)"?', str(content_type or ""), re.I)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    for part in raw_bytes.split(delimiter):
        part = part.strip()
        if not part or part == b"--":
            continue
        header_block, separator, body = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers = header_block.decode("utf-8", errors="ignore")
        if 'name="file"' not in headers:
            continue
        filename_match = re.search(r'filename="([^"]*)"', headers)
        file_name = os.path.basename(filename_match.group(1)) if filename_match else ""
        file_bytes = body.rstrip(b"\r\n")
        return file_name, base64.b64encode(file_bytes).decode("ascii")
    raise ValueError("missing file part")


def json_log(event, **fields):
    payload = {"ts": now_iso(), "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def get_bucket_key(handler, scope):
    basis = f"{scope}:{client_ip_from_headers(handler)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def check_rate_limit(handler, scope, limit):
    now = time.time()
    key = get_bucket_key(handler, scope)
    bucket = REQUEST_BUCKETS.get(key)
    if not bucket or now - bucket["window_start"] >= REQUEST_WINDOW_SECONDS:
        bucket = {"window_start": now, "count": 0}
        REQUEST_BUCKETS[key] = bucket
    bucket["count"] += 1
    remaining = max(0, limit - bucket["count"])
    reset_in = max(0, int(REQUEST_WINDOW_SECONDS - (now - bucket["window_start"])))
    allowed = bucket["count"] <= limit
    return {
        "allowed": allowed,
        "remaining": remaining,
        "reset_in": reset_in,
        "count": bucket["count"],
    }


def clamp_score(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 0


def level_from_score(score):
    if score >= 80:
        return "E级-优秀"
    if score >= 60:
        return "D级-良好"
    if score >= 40:
        return "C级-合格"
    if score >= 20:
        return "B级-较差"
    return "A级-差"


def normalize_provider_config(provider):
    base = dict(PROVIDER_DEFAULTS.get(provider, {}))
    if base.get("base_url"):
        base["base_url"] = base["base_url"].rstrip("/")
    if not base.get("endpoint"):
        base["endpoint"] = "/chat/completions"
    if not str(base["endpoint"]).startswith("/"):
        base["endpoint"] = "/" + str(base["endpoint"])
    return base


def validate_upload(file_name, data):
    lower = (file_name or "").lower().strip()
    ext = ""
    if "." in lower:
        ext = lower[lower.rfind(".") :]
    if ext not in UPLOAD_ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported extension: {ext or 'none'}")
    if not data:
        raise ValueError("empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"file too large: {len(data)} bytes")
    prefixes = UPLOAD_MAGIC_PREFIXES.get(ext) or []
    if prefixes and not any(data.startswith(prefix) for prefix in prefixes):
        raise ValueError(f"file signature mismatch for {ext}")
    return ext


def extract_json_object(text):
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("json object not found")
    return json.loads(cleaned[start : end + 1])


def ensure_list(value, fallback=None):
    if isinstance(value, list):
        return value
    return fallback or []


def maybe_decode_text(data):
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def normalize_report(raw, provider, model, company_name):
    scores = raw.get("scores") or {}
    normalized_scores = {}
    for key in DIMENSION_ORDER:
        normalized_scores[key] = clamp_score(scores.get(key, 0))

    total = raw.get("overallScore")
    if total is None:
        total = round(sum(normalized_scores.values()) / len(DIMENSION_ORDER))
    total = clamp_score(total)
    overall_level = raw.get("overallLevel") or level_from_score(total)

    dims = []
    dimension_analysis = raw.get("dimensionAnalysis") or []
    by_key = {item.get("key"): item for item in dimension_analysis if isinstance(item, dict)}
    for key in DIMENSION_ORDER:
        item = by_key.get(key, {})
        score = normalized_scores[key]
        dims.append(
            {
                "key": key,
                "name": DIMENSION_LABELS[key],
                "score": score,
                "level": item.get("level") or level_from_score(score),
                "gap": clamp_score(item.get("gap", 0)),
                "diagnosis": item.get("diagnosis", ""),
                "swotFocus": item.get("swotFocus", ""),
                "actions": ensure_list(item.get("actions")),
                "kpis": ensure_list(item.get("kpis")),
                "risks": ensure_list(item.get("risks")),
            }
        )

    phases = []
    for item in ensure_list(raw.get("phases")):
        if not isinstance(item, dict):
            continue
        phases.append(
            {
                "name": item.get("name", ""),
                "goals": ensure_list(item.get("goals")),
            }
        )

    solutions = []
    for item in ensure_list(raw.get("solutions")):
        if not isinstance(item, dict):
            continue
        steps = []
        for step in ensure_list(item.get("steps")):
            if not isinstance(step, dict):
                continue
            steps.append(
                {
                    "t": step.get("title", ""),
                    "d": step.get("detail", ""),
                    "tm": step.get("timeline", ""),
                }
            )
        solutions.append(
            {
                "title": item.get("title", ""),
                "priority": item.get("priority", "medium"),
                "content": ensure_list(item.get("content")),
                "steps": steps,
            }
        )

    return {
        "provider": provider,
        "model": model,
        "generatedAt": raw.get("generatedAt") or now_iso(),
        "companyName": company_name,
        "overallScore": total,
        "overallLevel": overall_level,
        "executiveSummary": raw.get("executiveSummary", ""),
        "methodology": raw.get("methodology", ""),
        "researchBasis": raw.get("researchBasis", ""),
        "coreFindings": ensure_list(raw.get("coreFindings")),
        "dimensionAnalysis": dims,
        "phases": phases,
        "solutions": solutions,
    }


def build_prompt(payload):
    provider_label = PROVIDER_DEFAULTS.get(payload["provider"], {}).get("label", payload["provider"])
    compact_input = payload["analysis_input"]
    return f"""
你是{provider_label}产业集群诊断引擎。请按“五维融合、八要素、SWOT、头雁企业带动、阶段跃迁”输出企业升级诊断。

要求：
1. 只输出合法 JSON，不要 Markdown。
2. 评分范围 0-100，八要素都要给出分数与解释。
3. 结论必须具体，给出评分依据、差距、行动、KPI、风险、阶段方案。
4. 结论要能支撑网页中的卡片点击详情，因此各字段必须完整。
5. `researchBasis` 用一句话概括论文/研究方法，不要展开成长段。

JSON keys:
generatedAt, overallScore, overallLevel, executiveSummary, methodology, researchBasis,
scores, coreFindings, dimensionAnalysis, phases, solutions

其中：
- scores 必须包含 brand, marketing, production, rd, standard, logistics, capital, finance
- dimensionAnalysis 每项必须包含 key, level, gap, diagnosis, swotFocus, actions, kpis, risks
- phases 每项包含 name, goals
- solutions 每项包含 title, priority, content, steps
- steps 每项包含 title, detail, timeline

输入数据：
{json.dumps(compact_input, ensure_ascii=False)}
""".strip()


def build_messages(payload):
    analysis_input = payload.get("analysis_input") or {}
    custom_prompt = analysis_input.get("prompt")
    if custom_prompt:
        return [
            {
                "role": "system",
                "content": "你是严谨的产业集群软件 AI 模块引擎。必须只输出合法 JSON。",
            },
            {"role": "user", "content": str(custom_prompt)},
        ]
    prompt = build_prompt(payload)
    return [
        {
            "role": "system",
            "content": "你是严谨的区域产业集群研究分析系统，擅长把论文方法转化为企业升级诊断。必须只输出合法 JSON。",
        },
        {"role": "user", "content": prompt},
    ]


def call_gateway_via_browser(config, body):
    script_path = os.path.join(os.path.dirname(__file__), "gateway_browser_proxy.mjs")
    node_path = os.environ.get(
        "LJ_NODE_PATH",
        r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.16.0-win-x64\node.exe",
    )
    payload = {
        "base_url": config["base_url"],
        "api_key": config["api_key"],
        "body": body,
        "force_json": True,
    }
    last_error = None
    for _ in range(2):
        proc = subprocess.run(
            [node_path, script_path, json.dumps(payload, ensure_ascii=False)],
            capture_output=True,
            text=False,
            timeout=180,
            check=False,
        )
        stdout_text = (proc.stdout or b"").decode("utf-8", errors="ignore")
        stderr_text = (proc.stderr or b"").decode("utf-8", errors="ignore")
        if proc.returncode != 0:
            last_error = stderr_text.strip() or stdout_text.strip()
            if "ERR_CONNECTION_CLOSED" in last_error:
                time.sleep(1)
                continue
            raise RuntimeError(f"browser gateway proxy failed: {last_error}")
        try:
            result = json.loads(stdout_text.strip())
            break
        except Exception as exc:
            last_error = stdout_text[:1000]
            raise RuntimeError(f"browser gateway proxy returned invalid json: {last_error}") from exc
    else:
        raise RuntimeError(f"browser gateway proxy failed: {last_error or 'unknown error'}")
    if not result.get("ok"):
        raise RuntimeError(f"browser gateway proxy HTTP {result.get('status')}: {result.get('text', '')[:1000]}")
    try:
        return json.loads(result.get("text") or "{}")
    except Exception as exc:
        raise RuntimeError(f"browser gateway proxy returned non-json body: {str(result.get('text') or '')[:1000]}") from exc


FIELD_ALIASES = {
    "companyName": [r"企业名称", r"公司名称", r"单位名称"],
    "industry": [r"所属行业", r"行业", r"产业方向"],
    "legalPerson": [r"法定代表人", r"法人代表", r"法人"],
    "registeredCapital": [r"注册资本"],
    "establishDate": [r"成立日期", r"成立时间", r"创立时间"],
    "mainProducts": [r"主要产品", r"主营产品", r"核心产品"],
    "phone": [r"联系电话", r"联系方式", r"电话"],
    "email": [r"电子邮箱", r"邮箱", r"E-?mail"],
    "address": [r"企业地址", r"地址", r"办公地址"],
    "employees": [r"员工人数", r"职工人数", r"员工数量"],
    "revenue": [r"年营业额", r"营业收入", r"年营收", r"营收"],
    "exportRatio": [r"出口占比", r"出口比例"],
    "businessScope": [r"经营范围", r"业务范围"],
    "rdRatio": [r"研发投入占比", r"研发占比", r"研发经费占比"],
    "rdStaff": [r"研发人员数量", r"研发人员", r"研发团队人数"],
    "patentCount": [r"专利数量", r"专利数"],
    "rdInstitution": [r"研发机构", r"研发平台", r"技术中心"],
    "rdCapability": [r"自主研发能力", r"研发能力"],
    "isoCert": [r"ISO认证", r"ISO9001"],
    "ceCert": [r"CE认证"],
    "otherCerts": [r"其他认证", r"国际认证"],
    "standardCoverage": [r"标准覆盖率", r"标准覆盖"],
    "brandCount": [r"自主品牌数量", r"品牌数量"],
    "brandRatio": [r"品牌投入占比", r"品牌投入比例"],
    "oemRatio": [r"OEM/ODM占比", r"OEM占比", r"ODM占比"],
    "ecommerceRatio": [r"跨境电商占比", r"电商占比"],
    "exportCountries": [r"出口国家数量", r"出口国家数", r"出口国家"],
    "brandAwareness": [r"品牌知名度"],
    "logisticsCost": [r"物流成本占比", r"物流成本"],
    "deliveryDays": [r"平均交货周期", r"交货周期", r"交付周期"],
    "warehouseCount": [r"海外仓数量", r"海外仓数"],
    "supplyChainLevel": [r"供应链管理能力", r"供应链能力"],
    "mainIssues": [r"企业主要问题", r"主要问题", r"当前问题"],
    "upgradeGoals": [r"企业升级目标", r"升级目标", r"发展目标"],
}


def decode_upload_text(file_name, content_base64):
    data = base64.b64decode(content_base64)
    ext = validate_upload(file_name, data)
    lower = (file_name or "").lower()
    if lower.endswith(".docx"):
        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text[:MAX_UPLOAD_TEXT_CHARS]
    if lower.endswith(".doc"):
        text = maybe_decode_text(data)
        return text[:MAX_UPLOAD_TEXT_CHARS]
    if lower.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text[:MAX_UPLOAD_TEXT_CHARS]
    if lower.endswith(".json") or lower.endswith(".txt") or lower.endswith(".md"):
        return maybe_decode_text(data)[:MAX_UPLOAD_TEXT_CHARS]
    return maybe_decode_text(data)[:MAX_UPLOAD_TEXT_CHARS]


def cleanup_value(value):
    value = re.sub(r"\s+", " ", value or "").strip("：:;；，, ")
    return value


def normalize_date_text(value):
    value = cleanup_value(value)
    match = re.search(r"(\d{4})[年\-/\.](\d{1,2})[月\-/\.](\d{1,2})", value)
    if not match:
        return value
    y, m, d = match.groups()
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def regex_extract(text, aliases):
    for alias in aliases:
        patterns = [
            rf"{alias}\s*[：:]\s*(.+)",
            rf"{alias}\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                line = match.group(1).splitlines()[0]
                return cleanup_value(line)
    return ""


def infer_industry(text):
    if "小家电" in text:
        return "小家电制造"
    if "大家电" in text:
        return "大家电制造"
    if "电子" in text:
        return "电子产品"
    if "五金" in text:
        return "五金制品"
    return ""


def parse_document_fields(text):
    result = {}
    for field, aliases in FIELD_ALIASES.items():
        value = regex_extract(text, aliases)
        if value:
            result[field] = normalize_date_text(value) if field == "establishDate" else value

    if not result.get("industry"):
        inferred = infer_industry(text)
        if inferred:
            result["industry"] = inferred

    if not result.get("companyName"):
        lines = [cleanup_value(x) for x in text.splitlines() if cleanup_value(x)]
        if lines:
            result["companyName"] = lines[0][:80]

    return result


def build_parse_summary(fields):
    labels = {
        "companyName": "企业名称",
        "industry": "所属行业",
        "legalPerson": "法定代表人",
        "registeredCapital": "注册资本",
        "establishDate": "成立日期",
        "mainProducts": "主要产品",
        "employees": "员工人数",
        "revenue": "年营业额",
        "rdRatio": "研发投入占比",
        "brandCount": "自主品牌数量",
        "logisticsCost": "物流成本占比",
        "upgradeGoals": "升级目标",
    }
    summary = []
    for key, label in labels.items():
        if fields.get(key):
            summary.append({"field": key, "label": label, "value": fields[key]})
    return summary


def call_openai_compatible(provider, config, payload):
    analysis_input = payload.get("analysis_input") or {}
    is_independent_module = bool(analysis_input.get("module"))
    body = {
        "model": config["model"],
        "messages": build_messages(payload),
        "temperature": 0.3,
    }
    if provider != "deepseek":
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
        "Connection": "close",
    }
    if provider == "mimo":
        headers["api-key"] = config["api_key"]
    if provider == "openai" and "gpt.fengxiaole.top" in str(config.get("base_url", "")):
        headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/149.0.0.0 Safari/537.36",
                "Referer": "https://gpt.fengxiaole.top/login",
                "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            }
        )
    raw = None
    url = config["base_url"] + config["endpoint"]
    for attempt in range(2):
        try:
            if attempt == 1:
                body["temperature"] = 0.2
            req = request.Request(
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
            break
        except error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="ignore")
            if (
                provider == "openai"
                and "gpt.fengxiaole.top" in str(config.get("base_url", ""))
                and exc.code >= 500
                and "Upstream access forbidden" in response_text
            ):
                raw = call_gateway_via_browser(config, body)
                break
            raise RuntimeError(f"{provider} HTTP {exc.code}: {response_text}") from exc
        except Exception as exc:
            if attempt == 0:
                time.sleep(1)
                continue
            if provider == "openai" and "gpt.fengxiaole.top" in str(config.get("base_url", "")):
                raw = call_gateway_via_browser(config, body)
                break
            raise RuntimeError(f"{provider} network error: {exc}") from exc
    if raw is None:
        raise RuntimeError(f"{provider} returned empty response")

    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider} returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(text_parts)
    parsed = extract_json_object(content or "")
    if is_independent_module:
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{provider} returned invalid module json")
        if not parsed.get("model"):
            parsed["model"] = config["model"]
        return parsed
    company_name = payload["analysis_input"]["company"].get("name", "")
    return normalize_report(parsed, provider, config["model"], company_name)


class Handler(BaseHTTPRequestHandler):
    server_version = "LJAI/1.0"

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        json_log(
            "http_access",
            ip=client_ip_from_headers(self),
            method=getattr(self, "command", ""),
            path=getattr(self, "path", ""),
            detail=(format % args) if args else format,
        )

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.startswith("/api/status"):
            providers = {}
            for key, info in PROVIDER_DEFAULTS.items():
                providers[key] = {
                    "label": info["label"],
                    "configured": bool(info.get("api_key")),
                    "server_managed": True,
                }
            self._send(
                200,
                {
                    "ok": True,
                    "serverTime": now_iso(),
                    "providers": providers,
                },
            )
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        request_id = hashlib.sha256(f"{time.time()}:{self.path}:{client_ip_from_headers(self)}".encode("utf-8")).hexdigest()[:16]
        length_header = self.headers.get("Content-Length", "0")
        if self.path.startswith("/api/intake/parse"):
            limit_state = check_rate_limit(self, "upload", UPLOAD_LIMIT_PER_WINDOW)
            if not limit_state["allowed"]:
                self._send(
                    429,
                    {
                        "ok": False,
                        "error": "upload rate limit exceeded",
                        "retryAfterSeconds": limit_state["reset_in"],
                        "requestId": request_id,
                    },
                )
                return
            try:
                length = int(length_header)
                if length <= 0:
                    self._send(400, {"ok": False, "error": "empty request", "requestId": request_id})
                    return
                if length > MAX_REQUEST_BYTES:
                    self._send(413, {"ok": False, "error": "request too large", "requestId": request_id})
                    return
            except Exception:
                self._send(400, {"ok": False, "error": "invalid content length", "requestId": request_id})
                return

            file_name = ""
            content_base64 = ""
            content_type = (self.headers.get("Content-Type", "") or "").lower()
            if "multipart/form-data" in content_type:
                try:
                    raw_bytes = self.rfile.read(length)
                    file_name, content_base64 = parse_multipart_file(raw_bytes, self.headers.get("Content-Type", ""))
                except Exception as exc:
                    self._send(400, {"ok": False, "error": f"invalid multipart upload: {exc}", "requestId": request_id})
                    return
            else:
                try:
                    raw_body = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw_body or "{}")
                except Exception:
                    self._send(400, {"ok": False, "error": "invalid json", "requestId": request_id})
                    return
                file_name = payload.get("fileName", "")
                content_base64 = payload.get("contentBase64", "")

            if not file_name or not content_base64:
                self._send(400, {"ok": False, "error": "missing file", "requestId": request_id})
                return
            try:
                text = decode_upload_text(file_name, content_base64)
                fields = parse_document_fields(text)
            except Exception as exc:
                json_log(
                    "upload_parse_failed",
                    requestId=request_id,
                    ip=client_ip_from_headers(self),
                    fileName=file_name,
                    error=str(exc),
                )
                self._send(400, {"ok": False, "error": f"parse failed: {exc}", "requestId": request_id})
                return
            json_log(
                "upload_parsed",
                requestId=request_id,
                ip=client_ip_from_headers(self),
                fileName=file_name,
                extractedFields=list(fields.keys()),
                previewChars=min(len(text), 3000),
            )
            self._send(
                200,
                {
                    "ok": True,
                    "fileName": file_name,
                    "fields": fields,
                    "summary": build_parse_summary(fields),
                    "preview": text[:3000],
                    "requestId": request_id,
                },
            )
            return

        try:
            length = int(length_header)
            if length <= 0:
                self._send(400, {"ok": False, "error": "empty request", "requestId": request_id})
                return
            if length > MAX_REQUEST_BYTES:
                self._send(413, {"ok": False, "error": "request too large", "requestId": request_id})
                return
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
        except Exception:
            self._send(400, {"ok": False, "error": "invalid json", "requestId": request_id})
            return

        if not self.path.startswith("/api/ai/analyze"):
            self._send(404, {"ok": False, "error": "not found", "requestId": request_id})
            return

        limit_state = check_rate_limit(self, "ai", AI_LIMIT_PER_WINDOW)
        if not limit_state["allowed"]:
            self._send(
                429,
                {
                    "ok": False,
                    "error": "ai rate limit exceeded",
                    "retryAfterSeconds": limit_state["reset_in"],
                    "requestId": request_id,
                },
            )
            return

        provider = payload.get("provider", "mimo")
        if provider not in PROVIDER_DEFAULTS:
            self._send(400, {"ok": False, "error": "unsupported provider", "requestId": request_id})
            return

        config = normalize_provider_config(provider)
        if not config.get("base_url") or not config.get("model") or not config.get("api_key"):
            self._send(
                400,
                {
                    "ok": False,
                    "error": f"{provider} not configured",
                    "missing": {
                        "base_url": not bool(config.get("base_url")),
                        "model": not bool(config.get("model")),
                        "api_key": not bool(config.get("api_key")),
                    },
                    "requestId": request_id,
                },
            )
            return

        analysis_input = payload.get("analysis_input") or {}
        company_name = ((analysis_input.get("company") or {}).get("name") if isinstance(analysis_input.get("company"), dict) else "") or ""
        module_key = analysis_input.get("module") or "analysis"
        started_at = time.time()
        try:
            report = call_openai_compatible(provider, config, payload)
        except Exception as exc:
            json_log(
                "ai_request_failed",
                requestId=request_id,
                ip=client_ip_from_headers(self),
                provider=provider,
                model=config.get("model"),
                baseUrl=config.get("base_url"),
                apiKeyMasked=mask_secret(config.get("api_key")),
                module=module_key,
                companyName=company_name,
                error=str(exc),
                elapsedMs=int((time.time() - started_at) * 1000),
            )
            self._send(502, {"ok": False, "error": str(exc), "requestId": request_id})
            return

        json_log(
            "ai_request_succeeded",
            requestId=request_id,
            ip=client_ip_from_headers(self),
            provider=provider,
            model=config.get("model"),
            baseUrl=config.get("base_url"),
            apiKeyMasked=mask_secret(config.get("api_key")),
            module=module_key,
            companyName=company_name,
            elapsedMs=int((time.time() - started_at) * 1000),
        )
        self._send(200, {"ok": True, "report": report, "requestId": request_id})


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"LJ AI API server running at http://{HOST}:{PORT}")
    httpd.serve_forever()
