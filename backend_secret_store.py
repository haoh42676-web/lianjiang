import argparse
import base64
import json
import os
import sys
from pathlib import Path


DEFAULT_SECRET_FILE = Path(__file__).resolve().parent / ".backend-secrets.dpapi.json"
SECRET_NAMES = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MIMO_API_KEY")
PLAINTEXT_SECRET_FILE = Path(__file__).resolve().parent / ".backend-secrets.json"


def _require_windows():
    if os.name != "nt":
        raise RuntimeError("DPAPI secret store is only supported on Windows")


def _crypt32():
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DATA_BLOB),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DATA_BLOB),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    return ctypes, DATA_BLOB, crypt32, kernel32


def protect_text(value):
    _require_windows()
    ctypes, data_blob_cls, crypt32, kernel32 = _crypt32()
    raw = (value or "").encode("utf-8")
    if not raw:
        return ""
    in_blob = data_blob_cls(len(raw), (ctypes.c_byte * len(raw)).from_buffer_copy(raw))
    out_blob = data_blob_cls()
    if not crypt32.CryptProtectData(ctypes.byref(in_blob), "LJ backend secret", None, None, None, 0, ctypes.byref(out_blob)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def unprotect_text(encoded):
    _require_windows()
    if not encoded:
        return ""
    ctypes, data_blob_cls, crypt32, kernel32 = _crypt32()
    raw = base64.b64decode(encoded.encode("ascii"))
    in_blob = data_blob_cls(len(raw), (ctypes.c_byte * len(raw)).from_buffer_copy(raw))
    out_blob = data_blob_cls()
    description = ctypes.c_wchar_p()
    if not crypt32.CryptUnprotectData(ctypes.byref(in_blob), ctypes.byref(description), None, None, None, 0, ctypes.byref(out_blob)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData).decode("utf-8")
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)


def load_secret_store(secret_file=DEFAULT_SECRET_FILE):
    path = Path(secret_file)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    file_format = str(data.get("format") or "").strip().lower()
    if file_format == "plain-text-v1":
        secrets = data.get("secrets") or {}
        return {key: str(value or "") for key, value in secrets.items()}
    secrets = {}
    for key, encoded in (data.get("secrets") or {}).items():
        try:
            secrets[key] = unprotect_text(encoded)
        except Exception:
            secrets[key] = ""
    return secrets


def save_secret_store(secret_map, secret_file=DEFAULT_SECRET_FILE):
    path = Path(secret_file)
    if os.name != "nt":
        payload = {
            "format": "plain-text-v1",
            "secrets": {key: str(value or "") for key, value in secret_map.items() if value},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    payload = {
        "format": "dpapi-v1",
        "secrets": {key: protect_text(value) for key, value in secret_map.items() if value},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def configured_secret_names(secret_file=DEFAULT_SECRET_FILE):
    secrets = load_secret_store(secret_file)
    return [name for name in SECRET_NAMES if secrets.get(name)]


def parse_args():
    parser = argparse.ArgumentParser(description="Manage local DPAPI-protected backend API keys")
    parser.add_argument("--file", default=str(DEFAULT_SECRET_FILE), help="Secret store path")
    sub = parser.add_subparsers(dest="command", required=True)

    set_parser = sub.add_parser("set", help="Write/update protected secrets")
    set_parser.add_argument("--openai", default="", help="OpenAI API key")
    set_parser.add_argument("--deepseek", default="", help="DeepSeek API key")
    set_parser.add_argument("--mimo", default="", help="MiMo API key")

    sub.add_parser("status", help="Show which secrets are present")
    return parser.parse_args()


def main():
    args = parse_args()
    secret_file = Path(args.file)
    if args.command == "set":
        existing = load_secret_store(secret_file)
        merged = {
            "OPENAI_API_KEY": args.openai or existing.get("OPENAI_API_KEY", ""),
            "DEEPSEEK_API_KEY": args.deepseek or existing.get("DEEPSEEK_API_KEY", ""),
            "MIMO_API_KEY": args.mimo or existing.get("MIMO_API_KEY", ""),
        }
        save_secret_store(merged, secret_file)
        present = [name for name, value in merged.items() if value]
        print(json.dumps({"ok": True, "file": str(secret_file), "configured": present}, ensure_ascii=False))
        return 0
    if args.command == "status":
        configured = configured_secret_names(secret_file)
        print(json.dumps({"ok": True, "file": str(secret_file), "configured": configured}, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
