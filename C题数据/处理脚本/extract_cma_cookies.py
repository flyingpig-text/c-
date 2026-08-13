"""Extract China Meteorological Data Service cookies from a Chrome profile copy.

Run after copying "Local State" and "Default/Network/Cookies" to %TEMP%\\cma_profile.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_ulong), ("pbData", ctypes.POINTER(ctypes.c_char))]


def dpapi_unprotect(data: bytes) -> bytes:
    blob_in = DATA_BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    )
    if not ok:
        raise RuntimeError("DPAPI unprotect failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def decrypt_cookie(value: str, key: bytes) -> str | None:
    if not value or value.startswith(("v10", "v11")):
        try:
            raw = base64.b64decode(value[3:])
        except Exception:
            return None
        if len(raw) < 28:
            return None
        nonce, ciphertext = raw[:12], raw[12:-16]
        tag = raw[-16:]
        try:
            return AESGCM(key).decrypt(nonce, ciphertext + tag, None).decode("utf-8")
        except Exception:
            return None
    return value


def main() -> None:
    profile = Path(tempfile.gettempdir()) / "cma_profile"
    local_state = json.loads((profile / "Local State").read_text(encoding="utf-8"))
    enc_key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
    assert enc_key.startswith(b"DPAPI")
    key = dpapi_unprotect(enc_key[5:])

    conn = sqlite3.connect(profile / "Cookies")
    rows = conn.execute(
        """
        SELECT host_key, name, path, is_secure, expires_utc, value
        FROM cookies
        WHERE host_key LIKE '%data.cma.cn%' OR host_key LIKE '%cma.cn%' OR host_key LIKE '%weather.com.cn%'
        """
    ).fetchall()
    cookies = []
    for host, name, path, secure, expires, value in rows:
        plain = decrypt_cookie(value, key)
        if plain is None:
            continue
        cookies.append(
            {
                "name": name,
                "value": plain,
                "domain": host,
                "path": path,
                "secure": bool(secure),
                "expires": expires,
            }
        )
    out = Path(tempfile.gettempdir()) / "cma_cookies.json"
    out.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"decrypted {len(cookies)} cookies -> {out}")
    for c in cookies[:20]:
        print(f"  {c['domain']}  {c['name']}  {c['value'][:40]}")


if __name__ == "__main__":
    main()
