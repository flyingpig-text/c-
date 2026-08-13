"""Check Chrome/Edge profile copies for any China Meteorological login state."""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path


def check_cookies(label: str, db: Path) -> None:
    if not db.exists():
        print(f"{label}: no cookies db")
        return
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT host_key, name FROM cookies WHERE host_key LIKE '%cma%' OR name LIKE '%cma%' OR host_key LIKE '%weather%'"
    ).fetchall()
    print(f"{label}: {len(rows)} cma/weather cookies")
    for host, name in rows[:20]:
        print("  ", host, name)


def check_leveldb(label: str, folder: Path, needle: bytes = b"data.cma.cn") -> None:
    if not folder.exists():
        print(f"{label}: no leveldb dir")
        return
    found = False
    for f in folder.iterdir():
        if not f.is_file() or f.name in ("LOCK", "CURRENT", "MANIFEST-000001", "LOG", "LOG.old"):
            continue
        data = f.read_bytes()
        if needle in data or b"cma" in data.lower():
            found = True
            print(f"{label}: hit in {f.name}")
    if not found:
        print(f"{label}: no data.cma.cn/cma strings")


def main() -> None:
    tmp = Path(tempfile.gettempdir())
    check_cookies("edge-default", tmp / "cma_profiles_edge" / "Default" / "Cookies")
    check_cookies("chrome-default", tmp / "cma_profiles_chrome" / "Default" / "Cookies")
    check_leveldb("chrome-session-storage", tmp / "cma_ss_leveldb")
    check_leveldb("chrome-local-storage", tmp / "cma_ls_leveldb")


if __name__ == "__main__":
    main()
