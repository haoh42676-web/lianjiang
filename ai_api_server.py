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
from collections import deque
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
SERVER_EVENTS = deque(maxlen=int(os.environ.get("LJ_ADMIN_EVENT_LIMIT", "800")))
SERVER_EVENTS_LOCK = threading.Lock()
SERVER_STARTED_AT = time.time()
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
SECRET_FILE = Path(
    os.environ.get(
        "LJ_SECRET_FILE",
        str(DEFAULT_SECRET_FILE if os.name == "nt" else PLAINTEXT_SECRET_FILE),
    )
).resolve()
AI_JOB_TTL_SECONDS = int(os.environ.get("LJ_AI_JOB_TTL_SECONDS", str(30 * 60)))
SECRET_FILE_SIGNATURE = None


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


def get_secret_file_signature():
    try:
        stat = SECRET_FILE.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        return None
    except Exception as exc:
        json_log("secret_store_stat_failed", file=str(SECRET_FILE), error=str(exc))
        return None


def refresh_backend_secrets_if_needed(force=False):
    global BACKEND_SECRETS, SECRET_FILE_SIGNATURE
    signature = get_secret_file_signature()
    if not force and signature == SECRET_FILE_SIGNATURE:
        return
    BACKEND_SECRETS = load_backend_secrets()
    SECRET_FILE_SIGNATURE = signature
    if "PROVIDER_DEFAULTS" in globals():
        PROVIDER_DEFAULTS["openai"]["api_key"] = BACKEND_SECRETS.get("OPENAI_API_KEY", "")
        PROVIDER_DEFAULTS["deepseek"]["api_key"] = BACKEND_SECRETS.get("DEEPSEEK_API_KEY", "")
        PROVIDER_DEFAULTS["mimo"]["api_key"] = BACKEND_SECRETS.get("MIMO_API_KEY", "")


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


def list_ai_jobs(limit=50):
    cleanup_expired_jobs()
    with AI_JOBS_LOCK:
        jobs = [build_job_snapshot(job) for job in AI_JOBS.values()]
    jobs.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    return jobs[: max(1, int(limit or 50))]


def record_server_event(payload):
    with SERVER_EVENTS_LOCK:
        SERVER_EVENTS.append(payload)


def list_server_events(limit=100):
    with SERVER_EVENTS_LOCK:
        events = list(SERVER_EVENTS)
    events.sort(key=lambda item: item.get("ts", ""), reverse=True)
    return events[: max(1, int(limit or 100))]


def build_admin_overview():
    jobs = list_ai_jobs(120)
    events = list_server_events(200)
    provider_stats = {}
    for provider_key, info in PROVIDER_DEFAULTS.items():
        provider_stats[provider_key] = {
            "label": info["label"],
            "configured": bool(info.get("api_key")),
            "total": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "queued": 0,
            "lastUpdatedAt": "",
        }
    for job in jobs:
        provider_key = job.get("provider") or ""
        stats = provider_stats.get(provider_key)
        if not stats:
            continue
        stats["total"] += 1
        status = (job.get("status") or "").lower()
        if status in ("completed", "failed", "running", "queued"):
            stats[status] += 1
        updated_at = job.get("updatedAt") or ""
        if updated_at and updated_at > stats["lastUpdatedAt"]:
            stats["lastUpdatedAt"] = updated_at

    ai_events = [item for item in events if item.get("event") in ("ai_request_succeeded", "ai_request_failed")]
    request_rows = []
    for item in ai_events[:40]:
        request_rows.append(
            {
                "time": item.get("ts", ""),
                "provider": item.get("provider", ""),
                "module": item.get("module", ""),
                "companyName": item.get("companyName", ""),
                "status": "success" if item.get("event") == "ai_request_succeeded" else "failed",
                "elapsedMs": item.get("elapsedMs", 0),
                "error": item.get("error", ""),
                "requestId": item.get("requestId", ""),
            }
        )

    http_events = [item for item in events if item.get("event") == "http_access"]
    recent_access = [
        {
            "time": item.get("ts", ""),
            "ip": item.get("ip", ""),
            "method": item.get("method", ""),
            "path": item.get("path", ""),
            "detail": item.get("detail", ""),
        }
        for item in http_events[:30]
    ]

    return {
        "serverTime": now_iso(),
        "uptimeSeconds": int(max(0, time.time() - SERVER_STARTED_AT)),
        "secretStore": {
            "file": str(SECRET_FILE),
            "configured": bool(BACKEND_SECRETS),
        },
        "providers": provider_stats,
        "jobs": jobs[:40],
        "recentRequests": request_rows,
        "recentAccess": recent_access,
        "summary": {
            "totalJobs": len(jobs),
            "completedJobs": sum(1 for item in jobs if (item.get("status") or "").lower() == "completed"),
            "failedJobs": sum(1 for item in jobs if (item.get("status") or "").lower() == "failed"),
            "runningJobs": sum(1 for item in jobs if (item.get("status") or "").lower() == "running"),
            "queuedJobs": sum(1 for item in jobs if (item.get("status") or "").lower() == "queued"),
            "recentRequestCount": len(request_rows),
        },
    }

BACKEND_SECRETS = load_backend_secrets()

PROVIDER_DEFAULTS = {
    "openai": {
        "label": "OpenAI",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://gpt.fengxiaole.top/v1"),
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
        "request_timeout_seconds": int(os.environ.get("MIMO_REQUEST_TIMEOUT_SECONDS", "360")),
    },
}

refresh_backend_secrets_if_needed(force=True)

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
    record_server_event(payload)
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
    refresh_backend_secrets_if_needed()
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
    if provider == "mimo":
        base["request_timeout_seconds"] = max(180, base["request_timeout_seconds"])
    return base


def provider_attempt_count(provider):
    if provider == "mimo":
        return 3
    return 2


def is_timeout_error(exc):
    text = str(exc or "").lower()
    return "timed out" in text or "timeout" in text


def extract_message_content(raw, provider, stage="response"):
    if isinstance(raw, list):
        if len(raw) == 1:
            raw = raw[0]
        else:
            raise RuntimeError(f"{provider} {stage} returned list payload")
    if not isinstance(raw, dict):
        raise RuntimeError(f"{provider} {stage} returned invalid payload type: {type(raw).__name__}")

    choices = raw.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{provider} {stage} returned no choices")

    first = choices[0]
    if isinstance(first, str):
        return first
    if not isinstance(first, dict):
        raise RuntimeError(f"{provider} {stage} returned invalid choice type: {type(first).__name__}")

    message = first.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "\n".join(normalize_text_value(item) for item in message if normalize_text_value(item))
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = first.get("text", "")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(part for part in text_parts if part)

    return normalize_text_value(content)


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


def normalize_report(raw, provider, model, company_name, input_company=None):
    company_profile = merge_company_profile(
        input_company if isinstance(input_company, dict) else {},
        raw.get("companyProfile") if isinstance(raw.get("companyProfile"), dict) else {},
    )
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
                "diagnosis": normalize_text_value(item.get("diagnosis")),
                "scoreExplanation": normalize_text_value(item.get("scoreExplanation")),
                "benchmarkComparison": normalize_text_value(item.get("benchmarkComparison")),
                "rootCauses": normalize_text_list(item.get("rootCauses")),
                "evidenceFromInput": normalize_text_list(item.get("evidenceFromInput")),
                "swotFocus": normalize_text_value(item.get("swotFocus")),
                "actions": normalize_text_list(item.get("actions")),
                "kpis": normalize_text_list(item.get("kpis")),
                "risks": normalize_text_list(item.get("risks")),
                "owner": normalize_text_value(item.get("owner")),
                "resources": normalize_text_list(item.get("resources")),
                "milestones": normalize_text_list(item.get("milestones")),
                "expectedImpact": normalize_text_value(item.get("expectedImpact")),
                "paperMethodMapping": normalize_text_value(item.get("paperMethodMapping")),
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
        "companyProfile": company_profile,
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


def enrich_dimension_details(report):
    dimensions = ensure_list(report.get("dimensionAnalysis"))
    enriched = []
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        diagnosis = normalize_text_value(item.get("diagnosis"))
        score_explanation = normalize_text_value(item.get("scoreExplanation"))
        benchmark = normalize_text_value(item.get("benchmarkComparison"))
        root_causes = normalize_text_list(item.get("rootCauses"))
        evidence = normalize_text_list(item.get("evidenceFromInput"))
        actions = normalize_text_list(item.get("actions"))
        kpis = normalize_text_list(item.get("kpis"))
        risks = normalize_text_list(item.get("risks"))
        owner = normalize_text_value(item.get("owner"))
        resources = normalize_text_list(item.get("resources"))
        milestones = normalize_text_list(item.get("milestones"))
        expected_impact = normalize_text_value(item.get("expectedImpact"))
        paper_method_mapping = normalize_text_value(item.get("paperMethodMapping"))
        swot_focus = normalize_text_value(item.get("swotFocus"))

        score = clamp_score(item.get("score"))
        gap = clamp_score(item.get("gap"))
        dim_name = normalize_text_value(item.get("name")) or DIMENSION_LABELS.get(item.get("key"), "关键维度")

        if not diagnosis:
            diagnosis = f"{dim_name}当前得分{score}分，与标杆差距{gap}分，说明该维度仍存在明显短板，需围绕关键环节补强执行能力。"
        if not score_explanation:
            score_explanation = f"该维度按当前企业数据与行业标杆对比后形成{score}分判断，分数反映的是现状成熟度而不是单项能力上限。"
        if not benchmark:
            benchmark = f"当前得分{score}分；标杆差距{gap}分，说明该维度仍需通过连续动作缩小与头部企业的经营差距。"
        if not root_causes:
            root_causes = [f"{dim_name}缺少稳定的执行机制", f"{dim_name}相关资源投入不足", f"{dim_name}与市场/订单反馈联动不够"]
        if not evidence:
            evidence = [diagnosis, score_explanation, benchmark]
        if not actions:
            actions = [f"围绕{dim_name}建立专项台账与责任分工", f"按月复盘{dim_name}关键指标并纠偏", f"针对{dim_name}短板启动专项改进项目"]
        if not kpis:
            kpis = [f"3个月内完成{dim_name}专项台账", f"6个月内{dim_name}关键指标形成月度达标机制", f"12个月内{dim_name}得分提升5-10分"]
        if not risks:
            risks = [f"{dim_name}改进可能出现资源不足，需要预算锁定", f"{dim_name}跨部门协同不足会拖慢项目节奏"]
        if not owner:
            owner = "建议由企业负责人牵头"
        if not resources:
            resources = [f"{dim_name}专项预算", "跨部门协同资源"]
        if not milestones:
            milestones = [f"30天内明确{dim_name}责任人与计划", f"90天内完成阶段性动作闭环", f"180天内完成阶段复盘并升级方案"]
        if not expected_impact:
            expected_impact = f"若完成上述动作，可优先缩小{dim_name}维度差距，并带动综合评分持续提升。"
        if not paper_method_mapping:
            paper_method_mapping = "按八要素评分、产业集群对标、SWOT与阶段跃迁逻辑综合判断。"
        if not swot_focus:
            swot_focus = f"{dim_name}既有基础能力，也存在组织与市场衔接短板，需要把优势转成可复制经营动作。"

        enriched.append(
            {
                **item,
                "diagnosis": diagnosis,
                "scoreExplanation": score_explanation,
                "benchmarkComparison": benchmark,
                "rootCauses": root_causes[:4],
                "evidenceFromInput": evidence[:6],
                "swotFocus": swot_focus,
                "actions": actions[:5],
                "kpis": kpis[:4],
                "risks": risks[:3],
                "owner": owner,
                "resources": resources[:3],
                "milestones": milestones[:4],
                "expectedImpact": expected_impact,
                "paperMethodMapping": paper_method_mapping,
            }
        )
    report["dimensionAnalysis"] = enriched
    return report


def build_company_profile_snapshot(report):
    company = report.get("companyProfile") if isinstance(report.get("companyProfile"), dict) else {}
    return {
        "name": normalize_text_value(company.get("name") or report.get("companyName")),
        "industry": normalize_text_value(company.get("industry")),
        "revenue": normalize_text_value(company.get("revenue")),
        "employees": normalize_text_value(company.get("employees")),
        "exportRatio": normalize_text_value(company.get("exportRatio")),
        "brandRatio": normalize_text_value(company.get("brandRatio")),
        "oemRatio": normalize_text_value(company.get("oemRatio")),
        "ecommerceRatio": normalize_text_value(company.get("ecommerceRatio")),
        "rdRatio": normalize_text_value(company.get("rdRatio")),
        "deliveryDays": normalize_text_value(company.get("deliveryDays")),
        "logisticsCost": normalize_text_value(company.get("logisticsCost")),
        "mainIssues": normalize_text_value(company.get("mainIssues")),
        "upgradeGoals": normalize_text_value(company.get("upgradeGoals")),
    }


def profile_value(profile, key, fallback):
    value = normalize_text_value((profile or {}).get(key))
    return value or fallback


def clean_owner_label(owner):
    owner_text = normalize_text_value(owner)
    for token in ("建议由", "建议", "牵头", "负责"):
        owner_text = owner_text.replace(token, "")
    owner_text = owner_text.strip("：: ，,")
    return owner_text or "企业负责人"


def build_dimension_defaults_v2(dim_key, dim_name, profile, score, gap):
    company_name = profile_value(profile, "name", "企业")
    industry = profile_value(profile, "industry", "当前行业")
    revenue = profile_value(profile, "revenue", "当前营收规模")
    export_ratio = profile_value(profile, "exportRatio", "当前出口占比")
    brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
    oem_ratio = profile_value(profile, "oemRatio", "当前OEM占比")
    ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
    rd_ratio = profile_value(profile, "rdRatio", "当前研发投入占比")
    delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
    logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
    employees = profile_value(profile, "employees", "当前人员规模")
    main_issues = profile_value(profile, "mainIssues", "当前升级难点")
    upgrade_goals = profile_value(profile, "upgradeGoals", "当前升级目标")

    defaults = {
        "diagnosis": f"{company_name}在{dim_name}上当前约{score}分，关键矛盾集中在{main_issues}，需要把升级目标“{upgrade_goals}”拆成可执行动作。",
        "scoreExplanation": f"{dim_name}评分不是看单一资源，而是看资源是否转成经营结果、是否有稳定组织机制、是否能支撑阶段升级。",
        "benchmarkComparison": f"{dim_name}当前得分{score}分。{'已具备一定基础，但还没形成复制优势。' if score >= 60 else '仍是当前转型中的明显短板，需要优先补课。'}",
        "rootCauses": [
            f"{dim_name}缺少量化经营目标，导致执行动作容易泛化",
            f"{dim_name}责任链与复盘节奏不稳定，跨部门协同效果差",
            f"{dim_name}没有与当前升级目标“{upgrade_goals}”形成清晰的转化路径",
        ],
        "evidenceFromInput": [f"企业营收：{revenue}", f"当前问题：{main_issues}", f"升级目标：{upgrade_goals}"],
        "swotFocus": f"{dim_name}既要利用{company_name}现有能力，也要补齐从资源到结果的经营转化机制。",
        "actions": [
            f"先把{dim_name}拆成90天项目清单、责任人和预算，避免继续停留在口号层面",
            f"围绕{dim_name}选1个样板场景，先做出首轮可验证结果",
            f"把{dim_name}指标纳入月度经营会，持续复盘完成率、投入产出和偏差原因",
        ],
        "kpis": [
            f"{dim_name}90天内形成项目台账与月度复盘机制",
            f"{dim_name}180天内形成至少1个可复制样板",
            f"{dim_name}12个月内带动综合评分提升5分以上",
        ],
        "risks": [
            f"{dim_name}如果只加预算不改责任链，容易继续产出低密度动作",
            f"{dim_name}涉及多部门协同，若没有一把手节奏管理，执行会走样",
        ],
        "owner": "企业负责人",
        "resources": ["专项预算", "跨部门周例会", "经营数据看板"],
        "milestones": [
            f"30天内完成{dim_name}责任人、预算和项目清单确认",
            f"60天内拿出首轮量化结果并识别执行偏差",
            f"90天内完成样板复盘并决定扩围或纠偏",
        ],
        "expectedImpact": f"{dim_name}补强后，将直接提升{company_name}围绕“{upgrade_goals}”推进的落地速度。",
        "paperMethodMapping": f"{dim_name}按八要素评分、阶段跃迁逻辑、经营约束与企业输入数据综合判断。",
    }

    if dim_key == "brand":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前品牌收入占比为{brand_ratio}，但OEM占比仍有{oem_ratio}，说明企业利润和议价能力仍主要受代工结构限制。",
                "scoreExplanation": f"品牌维度重点看品牌收入占比、品牌独立性以及品牌是否能反向带动研发和渠道；当前{brand_ratio}仍偏低。",
                "rootCauses": [
                    f"品牌主张没有从{industry}制造优势中提炼成消费者可感知价值",
                    "品牌预算、产品定义、渠道运营之间没有形成统一负责人机制",
                    "品牌建设没有绑定自主产品销售目标，动作容易碎片化",
                ],
                "evidenceFromInput": [f"品牌收入占比：{brand_ratio}", f"OEM占比：{oem_ratio}", f"升级目标：{upgrade_goals}"],
                "actions": [
                    "30天内明确1个主品牌、2个主推SKU和1套统一价值主张",
                    "重做品牌物料体系，统一包装、详情页、短视频脚本和渠道话术",
                    "要求新品立项必须说明品牌贡献、价格带策略和目标渠道",
                ],
                "kpis": ["自主品牌收入占比", "主推SKU毛利率", "品牌搜索指数或私域沉淀人数"],
                "owner": "品牌负责人",
                "resources": ["品牌预算", "主推SKU资源位", "内容设计与渠道投放支持"],
            }
        )
    elif dim_key == "marketing":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前电商占比为{ecommerce_ratio}，出口占比为{export_ratio}，说明终端渠道和用户运营能力仍弱，渠道结构偏订单导向。",
                "scoreExplanation": f"营销维度重点看渠道结构、流量获取、转化效率和复购能力；当前{ecommerce_ratio}说明距离稳定终端运营仍有明显距离。",
                "rootCauses": [
                    "渠道布局仍围绕存量客户和传统订单，缺少面向终端用户的增长机制",
                    "内容、投放、转化、复购没有形成统一经营闭环",
                    "营销动作没有与产品和库存节奏联动，爆品逻辑不强",
                ],
                "actions": [
                    "先选1个平台和1个重点价格带做样板，避免多平台同时投入",
                    "建立内容-投放-转化日报，重点跟踪点击率、转化率、退货率和投产比",
                    "把营销动作与新品上市节奏绑定，活动前先准备卖点、库存和客服话术",
                ],
                "kpis": ["样板渠道月销售额", "投放ROI", "加购转化率/复购率"],
                "owner": "渠道营销负责人",
                "resources": ["渠道投放预算", "内容团队", "数据归因看板"],
            }
        )
    elif dim_key == "production":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前交付周期约为{delivery_days}，以{employees}人员规模支撑业务，说明生产基础在，但柔性排产和交付效率仍有提升空间。",
                "scoreExplanation": f"生产维度不只看产能，还看交付速度、排产柔性、换线效率和异常响应速度；{delivery_days}说明交付能力还没完全跟上升级要求。",
                "rootCauses": [
                    "生产仍偏向大单逻辑，面对品牌化小批量、多批次需求时响应慢",
                    "排产、采购、库存和销售预测联动不足，产销协同效率不高",
                    "异常问题没有沉淀成标准化纠偏机制",
                ],
                "actions": [
                    "把生产目标从“完成订单”升级为“交付周期和毛利共同达标”",
                    "先挑1条主力产品线做周排产试点，拆解缺料、换线、返工损失时间",
                    "建立销售预测与生产计划周对齐机制，减少新品和促销插单",
                ],
                "kpis": ["平均交付周期", "换线时长", "返工率/计划达成率"],
                "owner": "生产运营负责人",
                "resources": ["排产数据台账", "工序瓶颈改善预算", "跨部门周计划会"],
            }
        )
    elif dim_key == "rd":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前研发投入占比约为{rd_ratio}，但研发是否真正转成可卖产品和可讲卖点，仍是核心问题。",
                "scoreExplanation": f"研发维度重点看投入是否转成产品差异化、上市节奏和销售贡献；单有{rd_ratio}投入并不代表研发已有效。",
                "rootCauses": [
                    "研发立项与用户需求、渠道卖点之间没有形成闭环",
                    "研发成果评估偏重完成项目，不够重视上市表现和毛利贡献",
                    "新品定义、试产验证、渠道反馈之间节奏脱节",
                ],
                "actions": [
                    "把研发项目按引流款、利润款、形象款重排，先保证立项与销售目标对齐",
                    "建立新品上市复盘表，跟踪首月销量、毛利、退货原因和用户差评",
                    "要求研发、营销、生产三方共同参加新品立项和试产评审",
                ],
                "kpis": ["新品上市成功率", "新品首月销售额", "研发项目按期转产率"],
                "owner": "研发负责人",
                "resources": ["样机验证预算", "用户反馈样本", "研发-营销联合评审机制"],
            }
        )
    elif dim_key == "standard":
        defaults.update(
            {
                "diagnosis": f"{company_name}所在行业为{industry}，当前出口占比为{export_ratio}，说明标准、认证和合规能力会直接影响客户信任和市场进入速度。",
                "scoreExplanation": "标准维度重点看认证覆盖、内控标准化和对关键市场规则的响应速度，而不是只看是否“有证”。",
                "rootCauses": [
                    "标准与认证更多停留在满足当前订单，缺少支持升级目标的前置规划",
                    "不同市场、不同渠道的合规要求没有沉淀成可复用清单",
                    "产品开发阶段缺少认证前置介入，导致后期反复补件和返工",
                ],
                "actions": [
                    "先梳理目标市场和渠道对应的认证清单，按优先级排序",
                    "把认证准备前置到新品开发阶段，避免样机完成后再补合规",
                    "建立法规变化月报，让销售、研发、品控同步知道红线变化",
                ],
                "kpis": ["重点市场认证完成率", "认证周期", "因合规问题导致的延误次数"],
                "owner": "质量与认证负责人",
                "resources": ["认证费用预算", "法规顾问/检测机构", "认证进度台账"],
            }
        )
    elif dim_key == "logistics":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前物流成本占比约为{logistics_cost}，交付周期约为{delivery_days}，说明供应链和履约效率已经开始影响利润和渠道体验。",
                "scoreExplanation": f"物流维度重点看成本、时效、库存周转和异常履约；{logistics_cost}与{delivery_days}都说明改进空间明显。",
                "rootCauses": [
                    "物流目标更多看发出去，没有把时效、成本和渠道服务体验一起管理",
                    "库存布局、补货策略和销售节奏没有形成联动",
                    "异常订单、退货和售后问题没有反向进入供应链优化",
                ],
                "actions": [
                    "先按渠道拆开统计物流成本和时效，找到利润被侵蚀最严重的场景",
                    "把库存周转、履约时效和售后退换货一起纳入供应链周报",
                    "围绕样板渠道优化仓配策略，优先缩短高频SKU履约链路",
                ],
                "kpis": ["物流成本占比", "72小时内发货率", "库存周转天数/退货时效"],
                "owner": "供应链负责人",
                "resources": ["仓配数据看板", "重点SKU库存策略", "物流承运商谈判资源"],
            }
        )
    elif dim_key == "capital":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前营收规模为{revenue}，升级目标是“{upgrade_goals}”，说明资本安排必须服务于转型节奏，而不是只追求拿钱。",
                "scoreExplanation": "资本维度重点看企业是否有与阶段目标匹配的资金安排、投资优先级和资本沟通能力。",
                "rootCauses": [
                    "转型项目很多，但资金优先级没有明确排序",
                    "资本动作和经营动作脱节，融资或补贴申请没有对准关键短板",
                    "缺少能向投资人或政府清楚说明项目回报逻辑的材料体系",
                ],
                "actions": [
                    "按品牌、研发、产能、渠道四类项目重排资本优先级，只保留真正影响主线的投入",
                    "梳理政府补贴、银行授信、股权融资三类资金路径，明确每条路径服务什么项目",
                    "建立项目回报测算表，要求每一笔投入都能说明回收周期和经营结果",
                ],
                "kpis": ["重点项目资金覆盖率", "融资/补贴到位金额", "单项目投入回收周期"],
                "owner": "战略与资本负责人",
                "resources": ["项目回报测算表", "融资材料", "政府项目申报资源"],
            }
        )
    elif dim_key == "finance":
        defaults.update(
            {
                "diagnosis": f"{company_name}当前营收规模为{revenue}，同时OEM占比{oem_ratio}、物流成本占比{logistics_cost}，说明财务管理要从记账视角转向经营决策视角。",
                "scoreExplanation": "财务维度重点看是否能把收入结构、毛利结构、费用投放和现金流变化及时翻译成经营动作。",
                "rootCauses": [
                    "财务口径与业务口径没有完全打通，经营层拿不到可直接决策的数据",
                    "利润分析停留在总账层面，无法快速看清SKU、渠道和客户的真实贡献",
                    "预算、复盘、纠偏没有闭环，财务对经营动作的牵引偏弱",
                ],
                "actions": [
                    "先按SKU、渠道、客户三层建立毛利分析表，找出真正赚钱和拖利润的部分",
                    "把预算执行、投放费用、库存占压和回款节奏放到一张经营报表里统一复盘",
                    "要求重大促销、新品、渠道动作都同步做财务测算和复盘",
                ],
                "kpis": ["SKU级毛利可视化覆盖率", "预算偏差率", "经营现金流/回款周期"],
                "owner": "财务负责人",
                "resources": ["经营分析报表", "预算滚动机制", "业务-财务对账口径"],
            }
        )
    return defaults


def looks_generic_text_list(items):
    generic_flags = (
        "执行机制",
        "资源投入不足",
        "市场/订单反馈",
        "专项台账",
        "按月复盘",
        "专项预算",
        "责任人与计划",
        "阶段性动作闭环",
        "牵头确认",
        "最相关的产品线",
    )
    normalized = normalize_text_list(items)
    if not normalized:
        return True
    joined = " | ".join(normalized)
    return any(flag in joined for flag in generic_flags)


def looks_generic_step_list(steps):
    if not steps:
        return True
    generic_flags = (
        "明确负责人",
        "明确责任链",
        "启动样板",
        "先做一个样板",
        "月度复盘",
        "结果并入经营会",
    )
    normalized = []
    for step in ensure_list(steps):
        if isinstance(step, dict):
            title = normalize_text_value(step.get("t") or step.get("title") or step.get("name"))
            detail = normalize_text_value(step.get("d") or step.get("detail") or step.get("description"))
            timeline = normalize_text_value(step.get("tm") or step.get("timeline"))
            combined = " ".join(part for part in (title, detail, timeline) if part)
            if combined:
                normalized.append(combined)
        else:
            text = normalize_text_value(step)
            if text:
                normalized.append(text)
    if not normalized:
        return True
    joined = " | ".join(normalized)
    return any(flag in joined for flag in generic_flags)


def build_phase_defaults_v2(index, focus, profile):
    focus_text = "、".join([item for item in ensure_list(focus) if normalize_text_value(item)][:2]) or "关键短板"
    brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
    ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
    rd_ratio = profile_value(profile, "rdRatio", "当前研发投入占比")
    delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
    logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
    revenue = profile_value(profile, "revenue", "当前营收规模")
    export_ratio = profile_value(profile, "exportRatio", "当前出口占比")
    if index == 0:
        return {
            "goals": [
                f"围绕{focus_text}建立一张专项台账，明确负责人、预算、目标值和周复盘节奏",
                f"把品牌占比{brand_ratio}、电商占比{ecommerce_ratio}、研发投入{rd_ratio}统一成月度经营指标",
                "完成重点客户、主推产品、核心渠道和供应链异常问题清单，并按影响利润优先级排序",
                "用一页经营看板同步老板、业务、生产、财务对当前短板的同一判断口径",
            ],
            "milestones": [
                "第2周前确认专项负责人、预算上限、目标值和例会机制",
                "第4周前完成客户/产品/渠道/供应链问题清单和优先级排序",
                "第6周前形成第一版经营看板并进入管理层周会",
                "第8周前冻结第一轮样板项目名单和验证口径",
            ],
            "explanation": "这一阶段先统一判断口径和抓手，避免后面看起来很忙，但每个部门理解的优先级都不一样。",
        }
    if index == 1:
        return {
            "goals": [
                f"围绕{focus_text}各做1个样板项目，要求6周内拿出可量化结果",
                f"把交付周期{delivery_days}、物流成本{logistics_cost}纳入专项周报，不再只看最终出货",
                "建立跨部门周例会、异常升级和负责人闭环机制，问题超过48小时必须有人接单",
                "每个样板项目都同步记录投入、产出、异常、修正动作，形成可复盘证据",
            ],
            "milestones": [
                "第10周前完成样板项目启动会并锁定验证指标",
                "第12周前拿到首轮样板结果，区分有效动作和无效动作",
                "第14周前完成一次跨部门纠偏，解决最关键的执行堵点",
                "第16周前给出是否扩围、暂停或重构的管理层决策",
            ],
            "explanation": "这一阶段不追求面面俱到，而是先把最影响经营结果的短板做出样板，确保动作不是空转。",
        }
    if index == 2:
        return {
            "goals": [
                "把样板项目里验证有效的动作沉淀为流程、模板、例会规则和考核口径",
                f"围绕营收{revenue}和出口占比{export_ratio}设定季度提升目标，并拆分到责任部门",
                "把样板经验复制到更多产品线、渠道或重点客户群，验证是否具备规模化可复制性",
                "同步梳理预算、人员、系统和数据需求，避免复制时因资源断档失败",
            ],
            "milestones": [
                "第5个月前完成模板、SOP、例会规则和考核口径发布",
                "第6个月前至少完成2条产品线或2个渠道的复制落地",
                "第7个月前输出复制效果复盘，识别放大后的新瓶颈",
                "第8个月前形成下一轮优化清单和资源补位方案",
            ],
            "explanation": "这一阶段重点是从单点改善走向组织能力，防止前面的结果只停留在几个能打的个人身上。",
        }
    return {
        "goals": [
            "把阶段成果转化为全年经营计划、预算安排和季度经营目标，不再作为临时专项存在",
            "形成对标头部企业的年度跃迁目标，明确品牌、研发、交付、资金四条主线的联动节奏",
            "把关键提升指标纳入董事会或经营班子固定复盘，确保升级动作不因短期订单波动中断",
            "完成下一轮增长方案储备，为新的渠道、产品和区域扩张预留资源和方法论",
        ],
        "milestones": [
            "第9个月前把专项成果写入年度经营预算和部门KPI",
            "第10个月前完成一次对标头部企业的差距复盘和来年规划",
            "第11个月前明确下一轮品牌、研发、交付和资金协同项目储备",
            "第12个月前提交年度总结和下一年度升级路线图",
        ],
        "explanation": "这一阶段的任务是把前面有效的动作升级为全年经营能力，让改善可以持续滚动，而不是做一轮就结束。",
    }


def build_solution_content_v2(dim_key, dim_name, owner, owner_label, profile):
    company_name = profile_value(profile, "name", "企业")
    brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
    ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
    rd_ratio = profile_value(profile, "rdRatio", "当前研发投入占比")
    delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
    logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
    upgrade_goals = profile_value(profile, "upgradeGoals", "当前升级目标")
    default_items = [
        f"由{owner}牵头，把{dim_name}拆成90天专项，写清目标值、预算上限、负责人和周复盘节奏",
        f"围绕{dim_name}只选1个样板场景先做深，避免同时铺太多动作导致看不到结果",
        f"把{dim_name}相关动作拆到产品、渠道、供应链、财务等责任人，每周检查完成率和异常原因",
        f"要求每个动作同步记录投入、产出、风险和下周修正项，月底进入经营会复盘",
    ]
    mapping = {
        "brand": [
            f"由{owner}牵头，明确1个主品牌、3个主推SKU和统一价值主张，目标是拉动品牌收入占比脱离当前{brand_ratio}",
            "重做包装、详情页、短视频脚本、招商话术和渠道陈列素材，确保品牌表达在各渠道一致",
            "新产品立项必须写清品牌贡献、目标价格带、目标客群和渠道打法，不能再只按生产逻辑立项",
            "把品牌预算优先投向主推SKU验证，而不是平均分散到所有产品线上",
        ],
        "marketing": [
            f"围绕当前电商占比{ecommerce_ratio}，只选1个平台和1个价格带做增长样板，先把转化跑通再扩平台",
            "建立内容、投放、转化、复购一体化日报，至少盯住点击率、转化率、退货率和ROI",
            "把营销节奏与新品上市、库存深度和客服话术绑定，避免活动做起来但供货与承接掉链子",
            "沉淀高转化素材、页面结构和人群策略，形成可复制的渠道打法模板",
        ],
        "production": [
            f"把交付周期从当前{delivery_days}作为一号改进指标，拆解缺料、换线、返工和插单造成的时间损失",
            "先在1条主力产品线做周排产样板，建立生产、采购、仓储、销售联动节奏",
            "把异常停线、返工、插单原因写入日报，超过阈值的异常必须在48小时内给出处置人",
            "把生产目标从只看完工量改成同时看交付周期、毛利影响和计划达成率",
        ],
        "rd": [
            f"围绕升级目标“{upgrade_goals}”重排研发项目，先做能带来销量、毛利或品牌增量的产品",
            f"要求研发投入{rd_ratio}对应到具体新品和差异化卖点，避免投入与结果脱钩",
            "建立新品上市复盘表，跟踪首月销量、毛利、差评原因、退货原因和渠道反馈",
            "研发、营销、生产三方共同参与新品立项和试产评审，减少产品定义偏差",
        ],
        "standard": [
            f"围绕{company_name}目标市场建立认证清单、时间表和预算，不再等订单来了再被动补证",
            "把认证准备前移到新品开发阶段，样机设计时就校验法规、检测和标签要求",
            "建立法规变化月报，销售、研发、品控同步更新，减少后期返工和延误",
            "沉淀一份按市场和渠道分类的合规资料包，缩短客户审核和入驻准备周期",
        ],
        "logistics": [
            f"围绕物流成本占比{logistics_cost}和交付周期{delivery_days}建立渠道级成本与时效看板",
            "按渠道拆分库存布局、补货节奏、售后退换货和承运商表现，找出利润被侵蚀最严重的环节",
            "先优化高频SKU的仓配链路，缩短高销量产品的履约路径和异常处理时间",
            "把售后、退货、延误和缺货原因反向写回供应链周报，推动采购与仓配共同纠偏",
        ],
        "capital": [
            f"围绕升级目标“{upgrade_goals}”重排资金优先级，只保留真正影响增长主线的投入项目",
            f"把品牌、研发、产能、渠道四类投入分别写清预算、回收周期和负责人，避免资源分散",
            "同步梳理银行授信、政府补贴、产业基金等资金路径，明确各自服务哪个专项",
            "建立项目回报测算表，每笔大额投入都要在经营会上解释回本逻辑和阶段结果",
        ],
        "finance": [
            "先按SKU、渠道、客户三层建立毛利分析表，识别哪些订单在放大营收但吞噬利润",
            f"把OEM占比、物流成本占比和回款周期一起放进经营报表，形成真实经营质量视角",
            "所有重大促销、新品、渠道动作都要做事前测算和事后复盘，财务不再只做记账",
            "建立预算偏差、库存占压、回款异常三类预警，超过阈值自动进入月度经营会",
        ],
    }
    return mapping.get(dim_key, default_items)


def build_solution_steps_v2(dim_key, dim_name, owner_label, profile):
    company_name = profile_value(profile, "name", "企业")
    brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
    ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
    delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
    logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
    common_steps = [
        {
            "t": "确认专项口径与目标值",
            "d": f"由{owner_label}在第1周内确认{dim_name}专项负责人、预算上限、季度目标值、例会频次和升级机制，避免执行口径继续分散。",
            "tm": "第1周",
        },
        {
            "t": "锁定单一样板场景",
            "d": f"围绕{dim_name}选择1个最能出结果的样板产品、样板渠道或样板流程，只做一组可验证动作，不同时铺开多个方向。",
            "tm": "第2周",
        },
        {
            "t": "连续跟踪过程指标",
            "d": f"从第3周开始按周跟踪{dim_name}的投入、完成率、异常原因和修正动作，要求每周都有新增证据，不接受只报进度口号。",
            "tm": "第3-8周",
        },
        {
            "t": "月度经营会决策扩围",
            "d": f"第2个月末把{dim_name}样板结果带入经营会，按结果决定继续加码、复制到更多场景，还是停止无效动作重做方案。",
            "tm": "第2-3个月",
        },
    ]
    mapping = {
        "brand": [
            {
                "t": "确定主品牌与主推SKU",
                "d": f"第1周内确认1个主品牌、3个主推SKU和统一价值主张，目标是让品牌收入占比逐步脱离当前{brand_ratio}水平。",
                "tm": "第1周",
            },
            {
                "t": "重做品牌表达物料",
                "d": "第2-4周统一完成包装、详情页、视频脚本、销售话术和招商素材，确保品牌表达在不同渠道不再各说各话。",
                "tm": "第2-4周",
            },
            {
                "t": "把新品立项绑定品牌贡献",
                "d": "从第2个月开始，所有新品立项必须同时写清目标客群、价格带、渠道打法和品牌贡献，否则不进入开发排期。",
                "tm": "第2个月",
            },
            {
                "t": "复盘品牌带来的经营结果",
                "d": "第3个月复盘品牌搜索、私域沉淀、主推SKU毛利和品牌收入占比变化，决定下一轮要加码的产品与渠道。",
                "tm": "第3个月",
            },
        ],
        "marketing": [
            {
                "t": "确定单平台增长样板",
                "d": f"第1周锁定1个平台和1个价格带，围绕当前电商占比{ecommerce_ratio}先跑通单平台转化模型，不同时追多个平台。",
                "tm": "第1周",
            },
            {
                "t": "搭建内容投放转化日报",
                "d": "第2周起按日跟踪内容产出、投放花费、点击率、转化率、退货率和ROI，并标记每条素材的真实表现。",
                "tm": "第2-8周",
            },
            {
                "t": "把活动节奏和库存承接打通",
                "d": "所有促销或上新动作上线前，必须同步检查库存深度、客服话术、发货承诺和售后预案，避免流量来了接不住。",
                "tm": "第3-8周",
            },
            {
                "t": "沉淀可复制打法模板",
                "d": "第3个月汇总高转化素材、页面结构、人群策略和客服话术，形成可复制的渠道作战模板。",
                "tm": "第3个月",
            },
        ],
        "production": [
            {
                "t": "拆解交付周期损失",
                "d": f"第1周开始把当前{delivery_days}交付周期拆成缺料、换线、返工、插单四类损失，找出最拖后腿的工序与班组。",
                "tm": "第1-2周",
            },
            {
                "t": "上线主力产品周排产样板",
                "d": "第3周选1条主力产品线做周排产试点，强制采购、仓储、生产、销售按同一节奏滚动更新。",
                "tm": "第3-6周",
            },
            {
                "t": "建立异常48小时闭环",
                "d": "所有停线、返工、缺料、插单异常在48小时内必须给出责任人、临时措施和永久改善动作，避免周周重复发生。",
                "tm": "第3-8周",
            },
            {
                "t": "按交付与毛利双指标复盘",
                "d": "第3个月用交付周期、计划达成率、返工率和毛利影响四个指标复盘是否值得扩到更多产线。",
                "tm": "第3个月",
            },
        ],
        "rd": [
            {
                "t": "重排研发项目优先级",
                "d": "第1周按销量潜力、毛利贡献、品牌贡献重排研发项目，砍掉不能支撑经营目标的低价值立项。",
                "tm": "第1周",
            },
            {
                "t": "建立新品立项共评机制",
                "d": f"第2周起研发、营销、生产共同评审新品，确保{company_name}的研发动作不是只做技术完成，而是直接服务销售结果。",
                "tm": "第2-4周",
            },
            {
                "t": "跟踪新品上市首月表现",
                "d": "从第2个月开始跟踪首月销量、毛利、差评、退货和渠道反馈，判断研发投入是否真正转成市场结果。",
                "tm": "第2-3个月",
            },
            {
                "t": "淘汰无效研发路径",
                "d": "第3个月根据上市结果决定继续加码、调整卖点还是终止项目，把研发资源集中到有效方向。",
                "tm": "第3个月",
            },
        ],
        "logistics": [
            {
                "t": "建立渠道级成本时效看板",
                "d": f"第1周按渠道拆开物流成本占比{logistics_cost}和交付周期{delivery_days}，找出利润被侵蚀最明显的渠道与SKU。",
                "tm": "第1-2周",
            },
            {
                "t": "优化高频SKU仓配链路",
                "d": "第3-5周优先优化高销量SKU的仓库发货路径、补货策略和承运商组合，先把大头问题解决。",
                "tm": "第3-5周",
            },
            {
                "t": "把售后异常反向写回供应链",
                "d": "第2个月开始把延误、破损、退货和售后异常写回供应链周报，让采购、仓储、客服对同一问题负责。",
                "tm": "第2个月",
            },
            {
                "t": "评估是否扩展到更多渠道",
                "d": "第3个月用成本、时效、退货和客户投诉四类数据评估优化动作是否应复制到更多仓配场景。",
                "tm": "第3个月",
            },
        ],
    }
    return mapping.get(dim_key, common_steps)


def build_solution_expected_result_v2(dim_key, dim_name, profile):
    brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
    ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
    delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
    logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
    mapping = {
        "brand": f"90天内形成主品牌与主推SKU统一打法，品牌收入占比相对当前{brand_ratio}开始抬升，销售与渠道对品牌表达口径一致。",
        "marketing": f"90天内跑通单平台增长样板，相对当前电商占比{ecommerce_ratio}形成更清晰的流量、转化和复购改进路径。",
        "production": f"90天内识别并压缩交付周期中的关键损失环节，交付与计划达成的改善开始可量化体现，不再只靠加班兜底。",
        "rd": "90天内完成研发项目优先级重排，并能用新品销量、毛利和用户反馈判断研发投入是否有效。",
        "logistics": f"90天内形成渠道级成本时效看板，物流成本占比{logistics_cost}与交付周期{delivery_days}至少有一项出现实质改善。",
    }
    return mapping.get(dim_key, f"{dim_name}在90天内形成首轮量化改善，管理层能看清哪些动作值得继续投、哪些动作应该停止。")


def build_solution_why_now_v2(dim_key, dim_name, profile):
    upgrade_goals = profile_value(profile, "upgradeGoals", "当前升级目标")
    main_issues = profile_value(profile, "mainIssues", "当前经营短板")
    mapping = {
        "brand": f"如果品牌能力不先补齐，企业就会继续被代工结构锁住，升级目标“{upgrade_goals}”很难转成更高毛利和更强议价权。",
        "marketing": f"当前核心问题“{main_issues}”已经说明订单结构不稳，营销如果不能直接沉淀终端用户和高质量渠道，增长会继续波动。",
        "production": f"{dim_name}直接影响客户体验、回款节奏和利润兑现，若交付和计划执行仍旧脆弱，前端品牌与营销投入会被后端拖垮。",
        "rd": f"升级目标“{upgrade_goals}”最终要靠产品兑现，如果研发继续只追项目完成，不追市场结果，企业很难形成真正差异化。",
        "logistics": f"{dim_name}已经不只是成本问题，而是订单履约与客户体验问题，拖得越久，渠道投诉和利润侵蚀会越难纠偏。",
    }
    return mapping.get(dim_key, f"{dim_name}当前已经成为企业升级中的真实约束，如果不先解决，其他改善动作也很难稳定兑现经营结果。")


def build_priority_dim_key(dim):
    key = normalize_text_value((dim or {}).get("key")).lower()
    if key:
        return key
    name = normalize_text_value((dim or {}).get("name"))
    mapping = {
        "品牌": "brand",
        "营销": "marketing",
        "生产": "production",
        "研发": "rd",
        "标准": "standard",
        "物流": "logistics",
        "资本": "capital",
        "财务": "finance",
    }
    for token, token_key in mapping.items():
        if token in name:
            return token_key
    return ""


def normalize_dimension_key_token(value):
    text = normalize_text_value(value).strip().lower()
    if not text:
        return ""
    alias_map = {
        "brand": "brand",
        "品牌": "brand",
        "国际化品牌": "brand",
        "marketing": "marketing",
        "营销": "marketing",
        "国际化营销": "marketing",
        "production": "production",
        "制成": "production",
        "生产": "production",
        "国际化制成": "production",
        "rd": "rd",
        "研发": "rd",
        "国际化研发": "rd",
        "standard": "standard",
        "标准": "standard",
        "认证": "standard",
        "国际化标准": "standard",
        "国际化标准与认证": "standard",
        "logistics": "logistics",
        "物流": "logistics",
        "供应链": "logistics",
        "国际化物流": "logistics",
        "capital": "capital",
        "资本": "capital",
        "国际化资本": "capital",
        "finance": "finance",
        "财务": "finance",
        "金融": "finance",
        "国际化金融": "finance",
    }
    if text in alias_map:
        return alias_map[text]
    for token, dim_key in alias_map.items():
        if token and token in text:
            return dim_key
    return ""


def infer_solution_dim_key(item, fallback_dim):
    fallback_key = build_priority_dim_key(fallback_dim)
    target_dimensions = normalize_text_list((item or {}).get("targetDimensions"))
    for target in target_dimensions:
        dim_key = normalize_dimension_key_token(target)
        if dim_key:
            return dim_key
    text_candidates = [
        normalize_text_value((item or {}).get("title")),
        normalize_text_value((item or {}).get("expectedResult")),
        normalize_text_value((item or {}).get("whyNow")),
    ]
    text_candidates.extend(normalize_text_list((item or {}).get("content"), limit=4))
    text_candidates.extend(
        normalize_text_value(step.get("title") or step.get("name") or step.get("detail") or step.get("description"))
        for step in ensure_list((item or {}).get("steps"))
        if isinstance(step, dict)
    )
    merged_text = " ".join(part for part in text_candidates if part)
    dim_key = normalize_dimension_key_token(merged_text)
    return dim_key or fallback_key


def merge_company_profile(input_company, report_company_profile):
    merged = {}
    for source in (input_company, report_company_profile):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            text = normalize_text_value(value)
            if text:
                merged[key] = text
    return merged


def normalize_solution_step_entries(steps):
    normalized = []
    for step in ensure_list(steps):
        if not isinstance(step, dict):
            continue
        title = normalize_text_value(step.get("t") or step.get("title") or step.get("name"))
        detail = normalize_text_value(step.get("d") or step.get("detail") or step.get("description"))
        timeline = normalize_text_value(step.get("tm") or step.get("timeline"))
        if title or detail or timeline:
            normalized.append({"t": title, "d": detail, "tm": timeline})
    return normalized
def enrich_dimension_details_v2(report):
    profile = build_company_profile_snapshot(report)
    dimensions = ensure_list(report.get("dimensionAnalysis"))
    enriched = []
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        score = clamp_score(item.get("score"))
        gap = clamp_score(item.get("gap"))
        dim_key = normalize_text_value(item.get("key"))
        dim_name = normalize_text_value(item.get("name")) or DIMENSION_LABELS.get(dim_key, "关键维度")
        defaults = build_dimension_defaults_v2(dim_key, dim_name, profile, score, gap)

        diagnosis = normalize_text_value(item.get("diagnosis")) or defaults["diagnosis"]
        score_explanation = normalize_text_value(item.get("scoreExplanation")) or defaults["scoreExplanation"]
        benchmark = normalize_text_value(item.get("benchmarkComparison")) or defaults["benchmarkComparison"]
        root_causes = normalize_text_list(item.get("rootCauses"))
        if not root_causes or looks_generic_text_list(root_causes):
            root_causes = defaults["rootCauses"]
        evidence = normalize_text_list(item.get("evidenceFromInput")) or defaults["evidenceFromInput"]
        actions = normalize_text_list(item.get("actions"))
        if not actions or looks_generic_text_list(actions):
            actions = defaults["actions"]
        kpis = normalize_text_list(item.get("kpis"))
        if not kpis or looks_generic_text_list(kpis):
            kpis = defaults["kpis"]
        risks = normalize_text_list(item.get("risks"))
        if not risks or looks_generic_text_list(risks):
            risks = defaults["risks"]
        owner = normalize_text_value(item.get("owner")) or defaults["owner"]
        resources = normalize_text_list(item.get("resources"))
        if not resources or looks_generic_text_list(resources):
            resources = defaults["resources"]
        milestones = normalize_text_list(item.get("milestones"))
        if not milestones or looks_generic_text_list(milestones):
            milestones = defaults["milestones"]
        expected_impact = normalize_text_value(item.get("expectedImpact")) or defaults["expectedImpact"]
        paper_method_mapping = normalize_text_value(item.get("paperMethodMapping")) or defaults["paperMethodMapping"]
        swot_focus = normalize_text_value(item.get("swotFocus")) or defaults["swotFocus"]

        enriched.append(
            {
                **item,
                "name": dim_name,
                "diagnosis": diagnosis,
                "scoreExplanation": score_explanation,
                "benchmarkComparison": benchmark,
                "rootCauses": root_causes[:4],
                "evidenceFromInput": evidence[:6],
                "swotFocus": swot_focus,
                "actions": actions[:5],
                "kpis": kpis[:4],
                "risks": risks[:3],
                "owner": owner,
                "resources": resources[:3],
                "milestones": milestones[:4],
                "expectedImpact": expected_impact,
                "paperMethodMapping": paper_method_mapping,
            }
        )
    report["dimensionAnalysis"] = enriched
    return report
def enrich_phase_and_solution_details(report):
    dimensions = [item for item in ensure_list(report.get("dimensionAnalysis")) if isinstance(item, dict)]
    profile = build_company_profile_snapshot(report)
    if not dimensions:
        return report

    priority_dims = sorted(
        dimensions,
        key=lambda item: (-(clamp_score(item.get("gap", 0))), clamp_score(item.get("score", 0))),
    )
    top_names = [normalize_text_value(item.get("name")) for item in priority_dims[:4] if normalize_text_value(item.get("name"))]

    phases = []
    raw_phases = [item for item in ensure_list(report.get("phases")) if isinstance(item, dict)]
    default_phase_names = ["诊断校准期", "专项突破期", "复制固化期", "规模跃迁期"]
    for index in range(4):
        item = raw_phases[index] if index < len(raw_phases) else {}
        name = normalize_text_value(item.get("name")) or default_phase_names[index]
        goals = normalize_text_list(item.get("goals"))
        milestones = normalize_text_list(item.get("milestones"))
        focus = normalize_text_list(item.get("focusDimensions"))
        explanation = normalize_text_value(item.get("explanation"))
        if not focus:
            focus = top_names[index:index + 2] or top_names[:2]
        phase_defaults = build_phase_defaults_v2(index, focus, profile)
        if not goals or looks_generic_text_list(goals):
            goals = phase_defaults["goals"]
        if not milestones or looks_generic_text_list(milestones):
            milestones = phase_defaults["milestones"]
        if not explanation:
            explanation = phase_defaults["explanation"]
        phases.append(
            {
                "name": name,
                "goals": goals[:4],
                "milestones": milestones[:4],
                "focusDimensions": focus[:4],
                "explanation": explanation,
            }
        )
    report["phases"] = phases

    raw_solutions = [item for item in ensure_list(report.get("solutions")) if isinstance(item, dict)]
    solutions = []
    for index, dim in enumerate(priority_dims[:6]):
        item = raw_solutions[index] if index < len(raw_solutions) else {}
        dim_name = normalize_text_value(dim.get("name")) or f"关键维度{index + 1}"
        dim_key = infer_solution_dim_key(item, dim)
        target_dimensions = normalize_text_list(item.get("targetDimensions")) or [normalize_text_value(dim.get("key") or dim_name)]
        owner = normalize_text_value(dim.get("owner")) or "经营负责人"
        owner_label = clean_owner_label(owner)
        content = normalize_text_list(item.get("content"))
        if not content or looks_generic_text_list(content):
            content = build_solution_content_v2(dim_key, dim_name, owner, owner_label, profile)
        steps = normalize_solution_step_entries(item.get("steps"))
        if looks_generic_step_list(steps):
            steps = build_solution_steps_v2(dim_key, dim_name, owner_label, profile)
        expected_result = normalize_text_value(item.get("expectedResult"))
        if not expected_result or len(expected_result) < 25:
            expected_result = build_solution_expected_result_v2(dim_key, dim_name, profile)
        why_now = normalize_text_value(item.get("whyNow"))
        if not why_now or len(why_now) < 25:
            why_now = build_solution_why_now_v2(dim_key, dim_name, profile)
        solutions.append(
            {
                "title": normalize_text_value(item.get("title")) or f"{dim_name}专项提升方案",
                "priority": normalize_text_value(item.get("priority")) or ("high" if index < 2 else "medium"),
                "targetDimensions": target_dimensions[:4],
                "content": content[:5],
                "steps": steps[:5],
                "expectedResult": expected_result,
                "whyNow": why_now,
            }
        )
    report["solutions"] = solutions[:6]

    core_findings = normalize_text_list(report.get("coreFindings"))
    if len(core_findings) < 6:
        generated_findings = []
        for dim in priority_dims[:6]:
            dim_name = normalize_text_value(dim.get("name")) or "关键维度"
            score = clamp_score(dim.get("score", 0))
            gap = clamp_score(dim.get("gap", 0))
            evidence = normalize_text_list(dim.get("evidenceFromInput"), limit=2)
            finding = f"{dim_name}当前{score}分、差距{gap}分，说明该维度仍是企业升级主短板。"
            if evidence:
                finding += " 依据：" + "；".join(evidence[:2])
            generated_findings.append(finding)
        report["coreFindings"] = (core_findings + generated_findings)[:8]
    else:
        report["coreFindings"] = core_findings[:8]

    summary = normalize_text_value(report.get("executiveSummary"))
    if len(summary) < 120:
        top_desc = "、".join(top_names[:3]) or "品牌、营销与供应链"
        brand_ratio = profile_value(profile, "brandRatio", "当前品牌收入占比")
        ecommerce_ratio = profile_value(profile, "ecommerceRatio", "当前电商占比")
        rd_ratio = profile_value(profile, "rdRatio", "当前研发投入占比")
        delivery_days = profile_value(profile, "deliveryDays", "当前交付周期")
        logistics_cost = profile_value(profile, "logisticsCost", "当前物流成本占比")
        report["executiveSummary"] = (
            f"{profile.get('name') or report.get('companyName') or '该企业'}当前处于从制造能力积累向品牌与经营能力补课的升级阶段，"
            f"最关键的瓶颈集中在{top_desc}。现状表现为品牌占比{brand_ratio}、电商占比{ecommerce_ratio}、研发投入{rd_ratio}、"
            f"交付周期{delivery_days}与物流成本{logistics_cost}未形成协同改善闭环。未来12个月主线应先做口径统一和样板突破，"
            f"再把有效动作固化为制度与预算安排，最终把专项结果并入年度经营计划，带动综合评分持续提升。"
        )
    return report


def build_peer_consensus(peer_reports):
    reports = [item for item in ensure_list(peer_reports) if isinstance(item, dict)]
    dim_buckets = {
        key: {
            "scores": [],
            "gaps": [],
            "diagnosis": [],
            "score_explanations": [],
            "benchmarks": [],
            "root_causes": [],
            "evidence": [],
            "actions": [],
            "kpis": [],
            "risks": [],
            "owners": [],
            "resources": [],
            "milestones": [],
            "expected_impacts": [],
        }
        for key in DIMENSION_ORDER
    }
    overall_scores = []
    levels = []
    findings = []
    phase_names = []
    solution_titles = []
    solution_reasons = []

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
                solution_reasons.extend(normalize_text_list(solution.get("content"), limit=3))
                solution_reasons.extend(normalize_text_list(solution.get("steps"), limit=2))
                reason = normalize_text_value(solution.get("whyNow"))
                if reason:
                    solution_reasons.append(reason)
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
            bucket["score_explanations"].append(normalize_text_value(dim.get("scoreExplanation"))[:120])
            bucket["benchmarks"].append(normalize_text_value(dim.get("benchmarkComparison"))[:120])
            bucket["root_causes"].extend(normalize_text_list(dim.get("rootCauses"), limit=3))
            bucket["evidence"].extend(normalize_text_list(dim.get("evidenceFromInput"), limit=4))
            bucket["actions"].extend(normalize_text_list(dim.get("actions"), limit=3))
            bucket["kpis"].extend(normalize_text_list(dim.get("kpis"), limit=3))
            bucket["risks"].extend(normalize_text_list(dim.get("risks"), limit=2))
            bucket["owners"].append(normalize_text_value(dim.get("owner")))
            bucket["resources"].extend(normalize_text_list(dim.get("resources"), limit=2))
            bucket["milestones"].extend(normalize_text_list(dim.get("milestones"), limit=3))
            bucket["expected_impacts"].append(normalize_text_value(dim.get("expectedImpact"))[:120])

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
                "scoreExplanationHints": normalize_text_list(bucket["score_explanations"], limit=3),
                "benchmarkHints": normalize_text_list(bucket["benchmarks"], limit=3),
                "rootCauseHints": normalize_text_list(bucket["root_causes"], limit=4),
                "evidenceHints": normalize_text_list(bucket["evidence"], limit=5),
                "priorityActions": normalize_text_list(bucket["actions"], limit=4),
                "priorityKpis": normalize_text_list(bucket["kpis"], limit=4),
                "priorityRisks": normalize_text_list(bucket["risks"], limit=3),
                "ownerHints": normalize_text_list(bucket["owners"], limit=3),
                "resourceHints": normalize_text_list(bucket["resources"], limit=3),
                "milestoneHints": normalize_text_list(bucket["milestones"], limit=4),
                "expectedImpactHints": normalize_text_list(bucket["expected_impacts"], limit=3),
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
        "solutionReasonHints": normalize_text_list(solution_reasons, limit=12),
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
    content = extract_message_content(raw, repair_provider, "repair response")
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
    attempt_count = provider_attempt_count(provider)
    for attempt in range(attempt_count):
        try:
            if attempt >= 1:
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
            is_last_attempt = attempt >= attempt_count - 1
            if not is_last_attempt and (provider != "mimo" or is_timeout_error(exc)):
                time.sleep(1 + attempt)
                continue
            if provider == "openai" and "gpt.fengxiaole.top" in str(config.get("base_url", "")):
                raw = call_gateway_via_browser(config, body)
                break
            raise RuntimeError(f"{provider} network error: {exc}") from exc
    if raw is None:
        raise RuntimeError(f"{provider} returned empty response")

    content = extract_message_content(raw, provider)
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
    report = normalize_report(parsed, provider, config["model"], company_name, payload["analysis_input"].get("company"))
    report = enrich_dimension_details(report)
    report = enrich_dimension_details_v2(report)
    return enrich_phase_and_solution_details(report)


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
        if route_path == "/api/admin/overview":
            self._send(
                200,
                {
                    "ok": True,
                    "overview": build_admin_overview(),
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
