import json
import os
import re
import base64
import hashlib
import mimetypes
import time
import subprocess
import threading
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request
from urllib.parse import unquote, urlsplit
from docx import Document
from pypdf import PdfReader
from backend_secret_store import DEFAULT_SECRET_FILE, PLAINTEXT_SECRET_FILE, load_secret_store


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
AI_JOBS = {}
AI_JOBS_LOCK = threading.Lock()
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
SECRET_FILE = Path(
    os.environ.get(
        "LJ_SECRET_FILE",
        str(DEFAULT_SECRET_FILE if os.name == "nt" else PLAINTEXT_SECRET_FILE),
    )
).resolve()
AI_JOB_TTL_SECONDS = int(os.environ.get("LJ_AI_JOB_TTL_SECONDS", str(30 * 60)))


def load_backend_secrets():
    secrets = {}
    try:
        secrets = load_secret_store(SECRET_FILE)
    except Exception as exc:
        json_log("secret_store_load_failed", file=str(SECRET_FILE), error=str(exc))
    for env_name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MIMO_API_KEY"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            secrets[env_name] = env_value
    return secrets


def cleanup_expired_jobs():
    now = time.time()
    expired = []
    with AI_JOBS_LOCK:
        for job_id, job in AI_JOBS.items():
            updated_at_ts = float(job.get("updatedAtTs") or 0)
            if updated_at_ts and now - updated_at_ts > AI_JOB_TTL_SECONDS:
                expired.append(job_id)
        for job_id in expired:
            AI_JOBS.pop(job_id, None)


def build_job_snapshot(job):
    return {
        "jobId": job["jobId"],
        "provider": job["provider"],
        "status": job["status"],
        "progress": dict(job.get("progress") or {}),
        "createdAt": job["createdAt"],
        "updatedAt": job["updatedAt"],
        "requestId": job["requestId"],
        "report": job.get("report"),
        "error": job.get("error", ""),
    }


def set_job_progress(job_id, *, status=None, percent=None, label=None, meta=None, report=None, error=None):
    with AI_JOBS_LOCK:
        job = AI_JOBS.get(job_id)
        if not job:
            return None
        if status is not None:
            job["status"] = status
        progress = job.setdefault("progress", {})
        if percent is not None:
            progress["percent"] = max(0, min(100, int(percent)))
        if label is not None:
            progress["label"] = label
        if meta is not None:
            progress["meta"] = meta
        if report is not None:
            job["report"] = report
        if error is not None:
            job["error"] = str(error)
        job["updatedAtTs"] = time.time()
        job["updatedAt"] = now_iso()
        return build_job_snapshot(job)


def create_ai_job(provider, request_id):
    cleanup_expired_jobs()
    created_at = now_iso()
    job_id = uuid.uuid4().hex
    job = {
        "jobId": job_id,
        "provider": provider,
        "status": "queued",
        "progress": {
            "percent": 4,
            "label": "任务已创建",
            "meta": "后端已接收任务，准备调用模型",
        },
        "createdAt": created_at,
        "updatedAt": created_at,
        "updatedAtTs": time.time(),
        "requestId": request_id,
        "report": None,
        "error": "",
    }
    with AI_JOBS_LOCK:
        AI_JOBS[job_id] = job
    return build_job_snapshot(job)


def get_ai_job(job_id):
    cleanup_expired_jobs()
    with AI_JOBS_LOCK:
        job = AI_JOBS.get(job_id)
        return build_job_snapshot(job) if job else None

PROVIDER_DEFAULTS = {
    "openai": {
        "label": "OpenAI",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "endpoint": "/chat/completions",
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        "api_key": BACKEND_SECRETS.get("OPENAI_API_KEY", ""),
        "request_timeout_seconds": int(os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "240")),
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "endpoint": "/chat/completions",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": BACKEND_SECRETS.get("DEEPSEEK_API_KEY", ""),
        "request_timeout_seconds": int(os.environ.get("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "180")),
    },
    "mimo": {
        "label": "MiMo",
        "base_url": os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        "endpoint": os.environ.get("MIMO_ENDPOINT", "/chat/completions"),
        "model": os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"),
        "api_key": BACKEND_SECRETS.get("MIMO_API_KEY", ""),
        "request_timeout_seconds": int(os.environ.get("MIMO_REQUEST_TIMEOUT_SECONDS", "300")),
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


BACKEND_SECRETS = load_backend_secrets()


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
    try:
        base["request_timeout_seconds"] = max(60, int(base.get("request_timeout_seconds") or 120))
    except Exception:
        base["request_timeout_seconds"] = 120
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
    candidate = cleaned[start : end + 1]
    attempts = [candidate]
    sanitized = (
        candidate.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\ufeff", "")
    )
    sanitized = re.sub(r",(\s*[}\]])", r"\1", sanitized)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
    if sanitized != candidate:
        attempts.append(sanitized)
    last_error = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"json parse failed: {last_error}")


def ensure_list(value, fallback=None):
    if isinstance(value, list):
        return value
    return fallback or []


def normalize_text_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [normalize_text_value(item) for item in value]
        return "；".join(part for part in parts if part)
    if isinstance(value, dict):
        ordered_pairs = [
            ("title", value.get("title")),
            ("name", value.get("name")),
            ("label", value.get("label")),
            ("metric", value.get("metric")),
            ("detail", value.get("detail")),
            ("description", value.get("description")),
            ("content", value.get("content")),
            ("diagnosis", value.get("diagnosis")),
            ("summary", value.get("summary")),
            ("approach", value.get("approach")),
            ("logic", value.get("logic")),
            ("conflictResolution", value.get("conflictResolution")),
            ("judgementPrinciple", value.get("judgementPrinciple")),
            ("integrationRules", value.get("integrationRules")),
            ("integrationLogic", value.get("integrationLogic")),
            ("evidenceBase", value.get("evidenceBase")),
            ("timeline", value.get("timeline")),
            ("expectedResult", value.get("expectedResult")),
            ("whyNow", value.get("whyNow")),
            ("value", value.get("value")),
        ]
        primary = []
        for key, raw in ordered_pairs:
            text = normalize_text_value(raw)
            if text:
                primary.append(f"{key}: {text}" if key in {"timeline", "expectedResult", "whyNow"} else text)
        if primary:
            return "；".join(primary[:4])
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def normalize_text_list(value, fallback=None, limit=None):
    items = value if isinstance(value, list) else (fallback or [])
    normalized = []
    seen = set()
    for item in items:
        text = normalize_text_value(item)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if limit and len(normalized) >= limit:
            break
    return normalized


def normalize_phase_list(value):
    phases = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        goals = normalize_text_list(item.get("goals"))
        milestones = normalize_text_list(item.get("milestones"))
        phases.append(
            {
                "name": normalize_text_value(item.get("name")) or "",
                "goals": goals,
                "milestones": milestones,
                "focusDimensions": normalize_text_list(item.get("focusDimensions")),
                "explanation": normalize_text_value(item.get("explanation")),
            }
        )
    return phases


def normalize_solution_list(value):
    solutions = []
    for item in ensure_list(value):
        if not isinstance(item, dict):
            continue
        steps = []
        for step in ensure_list(item.get("steps")):
            if isinstance(step, dict):
                steps.append(
                    {
                        "t": normalize_text_value(step.get("title") or step.get("name")),
                        "d": normalize_text_value(step.get("detail") or step.get("description")),
                        "tm": normalize_text_value(step.get("timeline")),
                    }
                )
            else:
                text = normalize_text_value(step)
                if text:
                    steps.append({"t": text, "d": "", "tm": ""})
        solutions.append(
            {
                "title": normalize_text_value(item.get("title")) or "",
                "priority": normalize_text_value(item.get("priority")) or "medium",
                "targetDimensions": normalize_text_list(item.get("targetDimensions")),
                "content": normalize_text_list(item.get("content")),
                "steps": steps,
                "expectedResult": normalize_text_value(item.get("expectedResult")),
                "whyNow": normalize_text_value(item.get("whyNow")),
            }
        )
    return solutions


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
                "goals": item.get("goals"),
                "milestones": item.get("milestones"),
                "focusDimensions": item.get("focusDimensions"),
                "explanation": item.get("explanation"),
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
                "targetDimensions": item.get("targetDimensions"),
                "content": item.get("content"),
                "steps": steps,
                "expectedResult": item.get("expectedResult"),
                "whyNow": item.get("whyNow"),
            }
        )

    return {
        "provider": provider,
        "model": model,
        "generatedAt": raw.get("generatedAt") or now_iso(),
        "companyName": company_name,
        "overallScore": total,
        "overallLevel": overall_level,
        "executiveSummary": normalize_text_value(raw.get("executiveSummary")),
        "methodology": normalize_text_value(raw.get("methodology")),
        "researchBasis": normalize_text_value(raw.get("researchBasis")),
        "coreFindings": normalize_text_list(raw.get("coreFindings")),
        "dimensionAnalysis": dims,
        "phases": normalize_phase_list(phases),
        "solutions": normalize_solution_list(solutions),
    }


def build_peer_consensus(peer_reports):
    reports = [item for item in ensure_list(peer_reports) if isinstance(item, dict)]
    dim_buckets = {key: {"scores": [], "gaps": [], "diagnosis": [], "actions": [], "kpis": [], "risks": []} for key in DIMENSION_ORDER}
    overall_scores = []
    levels = []
    findings = []
    phase_names = []
    solution_titles = []

    for item in reports:
        report = item.get("report") if isinstance(item.get("report"), dict) else {}
        score = report.get("overallScore")
        if score is not None:
            overall_scores.append(clamp_score(score))
        level = normalize_text_value(report.get("overallLevel"))
        if level:
            levels.append(level)
        findings.extend(normalize_text_list(report.get("coreFindings"), limit=5))
        for phase in ensure_list(report.get("phases"))[:4]:
            if isinstance(phase, dict):
                name = normalize_text_value(phase.get("name"))
                if name:
                    phase_names.append(name)
        for solution in ensure_list(report.get("solutions"))[:6]:
            if isinstance(solution, dict):
                title = normalize_text_value(solution.get("title"))
                if title:
                    solution_titles.append(title)
        for dim in ensure_list(report.get("dimensionAnalysis")):
            if not isinstance(dim, dict):
                continue
            key = dim.get("key")
            if key not in dim_buckets:
                continue
            bucket = dim_buckets[key]
            if dim.get("score") is not None:
                bucket["scores"].append(clamp_score(dim.get("score")))
            if dim.get("gap") is not None:
                bucket["gaps"].append(clamp_score(dim.get("gap")))
            bucket["diagnosis"].append(normalize_text_value(dim.get("diagnosis"))[:120])
            bucket["actions"].extend(normalize_text_list(dim.get("actions"), limit=3))
            bucket["kpis"].extend(normalize_text_list(dim.get("kpis"), limit=3))
            bucket["risks"].extend(normalize_text_list(dim.get("risks"), limit=2))

    dim_consensus = []
    for key in DIMENSION_ORDER:
        bucket = dim_buckets[key]
        avg_score = round(sum(bucket["scores"]) / len(bucket["scores"])) if bucket["scores"] else 0
        avg_gap = round(sum(bucket["gaps"]) / len(bucket["gaps"])) if bucket["gaps"] else 0
        dim_consensus.append(
            {
                "key": key,
                "name": DIMENSION_LABELS[key],
                "avgScore": avg_score,
                "avgGap": avg_gap,
                "level": level_from_score(avg_score),
                "diagnosisHints": normalize_text_list(bucket["diagnosis"], limit=3),
                "priorityActions": normalize_text_list(bucket["actions"], limit=4),
                "priorityKpis": normalize_text_list(bucket["kpis"], limit=4),
                "priorityRisks": normalize_text_list(bucket["risks"], limit=3),
            }
        )

    return {
        "providers": [
            {
                "provider": item.get("provider") or "",
                "label": item.get("label") or item.get("provider") or "",
                "overallScore": (item.get("report") or {}).get("overallScore"),
                "overallLevel": (item.get("report") or {}).get("overallLevel"),
                "executiveSummary": normalize_text_value((item.get("report") or {}).get("executiveSummary"))[:180],
            }
            for item in reports
        ],
        "avgOverallScore": round(sum(overall_scores) / len(overall_scores)) if overall_scores else 0,
        "levelHints": normalize_text_list(levels, limit=3),
        "coreFindings": normalize_text_list(findings, limit=8),
        "phaseHints": normalize_text_list(phase_names, limit=6),
        "solutionHints": normalize_text_list(solution_titles, limit=6),
        "dimensionConsensus": dim_consensus,
    }


def repair_json_via_model(provider, config, broken_text):
    snippet = str(broken_text or "").strip()
    if not snippet:
        raise ValueError("empty broken response")
    repair_provider = provider
    repair_config = config
    openai_config = normalize_provider_config("openai")
    if openai_config.get("api_key") and openai_config.get("base_url") and openai_config.get("model"):
        repair_provider = "openai"
        repair_config = openai_config
    body = {
        "model": repair_config["model"],
        "messages": [
            {
                "role": "system",
                "content": "你是严格 JSON 修复器。保留原字段语义，只输出一个合法 JSON 对象，不要解释，不要省略字段。",
            },
            {
                "role": "user",
                "content": "把下面内容修复为严格合法 JSON。要求：1. 保持原字段含义不变；2. 不要删掉关键字段；3. 只返回 JSON 本身：\n" + snippet[:30000],
            },
        ],
        "temperature": 0,
    }
    if repair_provider != "deepseek":
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {repair_config['api_key']}",
        "Connection": "close",
    }
    if repair_provider == "mimo":
        headers["api-key"] = repair_config["api_key"]
    if repair_provider == "openai" and "gpt.fengxiaole.top" in str(repair_config.get("base_url", "")):
        headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/149.0.0.0 Safari/537.36",
                "Referer": "https://gpt.fengxiaole.top/login",
                "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
            }
        )
    url = repair_config["base_url"] + repair_config["endpoint"]
    req = request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    repair_timeout = repair_config.get("request_timeout_seconds", config.get("request_timeout_seconds", 120))
    with request.urlopen(req, timeout=repair_timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"{repair_provider} repair returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return extract_json_object(content or "")


def build_prompt(payload):
    provider_label = PROVIDER_DEFAULTS.get(payload["provider"], {}).get("label", payload["provider"])
    compact_input = payload["analysis_input"]
    if compact_input.get("batchSource"):
        return build_batch_source_prompt(payload)
    return f"""
你是{provider_label}产业集群诊断引擎。请按“五维融合、八要素、SWOT、头雁企业带动、阶段跃迁”输出企业升级诊断。

要求：
1. 只输出合法 JSON，不要 Markdown。
2. 评分范围 0-100，八要素都要给出分数与解释。
3. 结论必须具体，给出评分依据、差距、行动、KPI、风险、阶段方案，不能只写“加强、提升、优化、完善”这类空话。
4. 结论要能支撑网页中的卡片点击详情，因此各字段必须完整，不能留空，不能用“略”“同上”“待补充”。
5. `researchBasis` 用一句话概括论文/研究方法，不要展开成长段。
6. `executiveSummary` 必须写成可给企业管理层直接看的结论，至少包含：当前阶段判断、最关键的3个问题、未来12个月主线，总长度控制在 180-260 字。
7. `coreFindings` 至少 6 条，每条都要带判断逻辑或证据，不要重复。
8. `dimensionAnalysis` 的每个维度都必须写出：
   - diagnosis：60-120字，说明“现状-成因-影响”
   - swotFocus：40-90字，说明该维度最关键的强弱机会威胁
   - actions：至少4条，每条都要写清动作对象、动作内容、责任主体/资源或应用场景
   - kpis：至少4条，必须可量化，带时间或阈值
   - risks：至少3条，写出风险和控制要点
9. `phases` 必须严格输出 4 个阶段，每个阶段都要有：
   - goals：至少3条
   - milestones：至少3条
   - focusDimensions：2-4个重点维度
   - explanation：50-100字，说明为什么此阶段这样安排
10. `solutions` 输出 6-8 个完整方案，每个方案都要有：
   - content：至少4条，必须是具体动作包
   - steps：至少4条，每条都包含 title、detail、timeline
   - expectedResult：35-80字
   - whyNow：35-80字
11. 优先输出企业可以直接执行的建议，必须尽量给出负责人角色、周期、阶段目标、量化里程碑。
12. 如果输入信息不足，可以做合理推断，但要基于制造业企业升级场景，不得编造明显不合理事实。
13. 在保证完整性的前提下，避免冗长，每条 action、kpi、risk、goal、milestone、content、step 尽量控制在 18-50 字。

JSON keys:
generatedAt, overallScore, overallLevel, executiveSummary, methodology, researchBasis,
scores, coreFindings, dimensionAnalysis, phases, solutions

其中：
- scores 必须包含 brand, marketing, production, rd, standard, logistics, capital, finance
- dimensionAnalysis 每项必须包含 key, level, gap, diagnosis, swotFocus, actions, kpis, risks
- phases 每项包含 name, goals, milestones, focusDimensions, explanation
- solutions 每项包含 title, priority, targetDimensions, content, steps, expectedResult, whyNow
- steps 每项包含 title, detail, timeline

输入数据：
{json.dumps(compact_input, ensure_ascii=False)}
""".strip()


def build_batch_source_prompt(payload):
    provider_label = PROVIDER_DEFAULTS.get(payload["provider"], {}).get("label", payload["provider"])
    compact_input = payload["analysis_input"]
    return f"""
你是{provider_label}企业升级分析引擎。当前任务是为“三模型联合会诊”提供一份高质量、但相对精简的结构化底稿，供后续 ChatGPT 生成最终完整版总报告。

要求：
1. 只输出合法 JSON，不要 Markdown。
2. 结论必须具体，不允许泛泛而谈。
3. 本次输出重点是“判断逻辑 + 关键动作 + 量化指标”，不要写冗长铺陈。
4. 评分范围 0-100，八要素都要给出明确分数。
5. `executiveSummary` 控制在 120-180 字，必须包含：发展阶段、核心瓶颈、未来12个月主线。
6. `coreFindings` 输出 5-6 条，每条都要有判断依据。
7. `dimensionAnalysis` 的每个维度都必须写出：
   - diagnosis：45-90字
   - swotFocus：25-60字
   - actions：至少3条
   - kpis：至少3条
   - risks：至少2条
8. `phases` 只需输出 3 个关键阶段，每阶段 goals 至少 2 条。
9. `solutions` 只需输出 4 个高优先级方案，每个方案至少包含 3 条 content、3 个 steps。
10. 内容务必能直接服务最终综合报告，不要留空，不要写“略”“同上”。

JSON keys:
generatedAt, overallScore, overallLevel, executiveSummary, methodology, researchBasis,
scores, coreFindings, dimensionAnalysis, phases, solutions

其中：
- scores 必须包含 brand, marketing, production, rd, standard, logistics, capital, finance
- dimensionAnalysis 每项必须包含 key, level, gap, diagnosis, swotFocus, actions, kpis, risks
- phases 每项包含 name, goals, milestones, focusDimensions, explanation
- solutions 每项包含 title, priority, targetDimensions, content, steps, expectedResult, whyNow
- steps 每项包含 title, detail, timeline

输入数据：
{json.dumps(compact_input, ensure_ascii=False)}
""".strip()


def build_synthesis_prompt(payload):
    analysis_input = payload.get("analysis_input") or {}
    provider_label = PROVIDER_DEFAULTS.get(payload["provider"], {}).get("label", payload["provider"])
    company = analysis_input.get("company") if isinstance(analysis_input.get("company"), dict) else {}
    peer_reports = analysis_input.get("peerReports") or []
    consensus = build_peer_consensus(peer_reports)
    synthesis_input = {
        "company": company,
        "peerConsensus": consensus,
        "module": analysis_input.get("module") or "analysis",
    }
    return f"""
你是{provider_label}总报告整合引擎。你的任务不是重复三份报告，而是把 DeepSeek、MiMo、ChatGPT 三份企业诊断结果交叉验证后，整合成一份企业可以直接使用的最终分析报告。

整合原则：
1. 只输出合法 JSON，不要 Markdown。
2. 先比对三份报告的一致结论，再处理冲突结论；不要简单平均，更不要把三份答案拼接。
3. 最终报告要体现“结论-原因-动作-里程碑-KPI-风险控制”的完整逻辑链。
4. 对空泛内容要主动压实成可执行动作；对冲突内容要选择更稳健、更适合制造业企业落地的方案。
5. 报告应直接服务企业经营班子、招商主管理层或项目负责人，不能写成泛泛论文摘要。
6. 评分范围 0-100，所有维度都要给出明确判断依据和差距。
7. `executiveSummary` 控制在 180-260 字，必须说清当前发展阶段、主要瓶颈、未来 12 个月主线、优先顺序。
8. `coreFindings` 输出 6-8 条，每条都要有逻辑依据，不要重复。
9. `dimensionAnalysis` 每个维度必须包含：
   - diagnosis：60-110字
   - swotFocus：35-80字
   - actions：至少4条，动作必须足够细
   - kpis：至少4条，必须量化并带时间界限
   - risks：至少3条，写出应对方式
10. `phases` 必须严格输出 4 个阶段，每阶段 goals 和 milestones 都至少 3 条，且 explanation 必须说明阶段衔接逻辑。
11. `solutions` 输出 6 个方案，每个方案都要覆盖 targetDimensions、content、steps、expectedResult、whyNow，并可被企业直接拆成项目。
12. 若三份报告出现分歧，优先保留：更具体、可量化、可执行、与企业当前基础更匹配的方案。
13. 在保证密度的前提下避免过长，单条 action、kpi、risk、step 尽量控制在 16-45 字。
14. 输入中的 `peerConsensus` 已经是压缩后的三模型共识摘要，请直接基于这些摘要完成最终报告，不要要求更多原文。

JSON keys:
generatedAt, overallScore, overallLevel, executiveSummary, methodology, researchBasis,
scores, coreFindings, dimensionAnalysis, phases, solutions

输入数据：
{json.dumps(synthesis_input, ensure_ascii=False)}
""".strip()


def build_messages(payload):
    analysis_input = payload.get("analysis_input") or {}
    if analysis_input.get("peerReports"):
        prompt = build_synthesis_prompt(payload)
        return [
            {
                "role": "system",
                "content": "你是严谨的产业集群综合诊断总控引擎。必须只输出合法 JSON，并把多模型意见整合成一份最终可执行报告。",
            },
            {"role": "user", "content": prompt},
        ]
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
    module_name = normalize_text_value(analysis_input.get("module")).lower()
    is_independent_module = bool(module_name and module_name != "analysis")
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
            with request.urlopen(req, timeout=config.get("request_timeout_seconds", 120)) as resp:
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
    try:
        parsed = extract_json_object(content or "")
    except Exception:
        parsed = repair_json_via_model(provider, config, content or "")
    if is_independent_module:
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{provider} returned invalid module json")
        if not parsed.get("model"):
            parsed["model"] = config["model"]
        return parsed
    company_name = payload["analysis_input"]["company"].get("name", "")
    return normalize_report(parsed, provider, config["model"], company_name)


def run_ai_job(job_id, provider, payload, config, request_id, client_ip):
    analysis_input = payload.get("analysis_input") or {}
    company_name = ((analysis_input.get("company") or {}).get("name") if isinstance(analysis_input.get("company"), dict) else "") or ""
    module_key = analysis_input.get("module") or "analysis"
    started_at = time.time()
    set_job_progress(job_id, status="running", percent=12, label="准备请求", meta="正在整理企业资料与模型提示词")
    try:
        set_job_progress(job_id, status="running", percent=26, label="发送请求", meta=f"已向 {config.get('label') or provider} 发起分析请求")
        set_job_progress(job_id, status="running", percent=64, label="模型分析中", meta="模型正在生成结构化诊断、阶段路径和解决方案")
        report = call_openai_compatible(provider, config, payload)
        set_job_progress(job_id, status="running", percent=88, label="结果校验", meta="正在校验 JSON 结构并整理卡片数据")
        set_job_progress(job_id, status="completed", percent=100, label="分析完成", meta=f"{config.get('label') or provider} 已生成完整分析报告", report=report, error="")
        json_log(
            "ai_request_succeeded",
            requestId=request_id,
            ip=client_ip,
            provider=provider,
            model=config.get("model"),
            baseUrl=config.get("base_url"),
            apiKeyMasked=mask_secret(config.get("api_key")),
            module=module_key,
            companyName=company_name,
            elapsedMs=int((time.time() - started_at) * 1000),
            asyncJobId=job_id,
        )
    except Exception as exc:
        set_job_progress(job_id, status="failed", percent=100, label="分析失败", meta=str(exc), error=str(exc))
        json_log(
            "ai_request_failed",
            requestId=request_id,
            ip=client_ip,
            provider=provider,
            model=config.get("model"),
            baseUrl=config.get("base_url"),
            apiKeyMasked=mask_secret(config.get("api_key")),
            module=module_key,
            companyName=company_name,
            error=str(exc),
            elapsedMs=int((time.time() - started_at) * 1000),
            asyncJobId=job_id,
        )


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

    def _send_bytes(self, code, body, content_type, cache_control="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path, cache_control="no-store"):
        try:
            resolved = Path(file_path).resolve()
        except Exception:
            self._send(404, {"ok": False, "error": "not found"})
            return
        try:
            resolved.relative_to(BASE_DIR)
        except Exception:
            self._send(403, {"ok": False, "error": "forbidden"})
            return
        if not resolved.is_file():
            self._send(404, {"ok": False, "error": "not found"})
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self._send_bytes(200, body, content_type, cache_control=cache_control)

    def _serve_app_asset(self):
        parsed = urlsplit(self.path)
        request_path = unquote(parsed.path or "/")
        if request_path in ("/", "/index.html"):
            self._serve_file(INDEX_FILE)
            return True
        candidate = (BASE_DIR / request_path.lstrip("/")).resolve()
        if candidate.is_file():
            cache_control = "public, max-age=3600" if candidate.suffix.lower() not in {".html"} else "no-store"
            self._serve_file(candidate, cache_control=cache_control)
            return True
        return False

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
        parsed = urlsplit(self.path)
        route_path = parsed.path or "/"
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
                    "secretStore": {
                        "file": str(SECRET_FILE),
                        "configured": bool(BACKEND_SECRETS),
                    },
                    "providers": providers,
                },
            )
            return
        if route_path.startswith("/api/ai/jobs/"):
            job_id = route_path.rsplit("/", 1)[-1].strip()
            job = get_ai_job(job_id)
            if not job:
                self._send(404, {"ok": False, "error": "job not found", "jobId": job_id})
                return
            self._send(200, {"ok": True, **job})
            return
        if self._serve_app_asset():
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
        wants_async = bool(payload.get("async"))
        client_ip = client_ip_from_headers(self)
        started_at = time.time()
        if wants_async:
            job = create_ai_job(provider, request_id)
            worker = threading.Thread(
                target=run_ai_job,
                args=(job["jobId"], provider, payload, config, request_id, client_ip),
                daemon=True,
            )
            worker.start()
            self._send(
                202,
                {
                    "ok": True,
                    "accepted": True,
                    "requestId": request_id,
                    "jobId": job["jobId"],
                    "status": job["status"],
                    "progress": job["progress"],
                },
            )
            return
        try:
            report = call_openai_compatible(provider, config, payload)
        except Exception as exc:
            json_log(
                "ai_request_failed",
                requestId=request_id,
                ip=client_ip,
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
            ip=client_ip,
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
