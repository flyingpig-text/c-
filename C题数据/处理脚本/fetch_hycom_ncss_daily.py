# -*- coding: utf-8 -*-
"""通过 HYCOM THREDDS NCSS 下载 2021 年南海站点逐日 0Z 海流时序。

数据：GOFS 3.1 GLBy0.08 uv3z，3 小时输出中的 00Z 快照。
NCSS 返回的是 Int16 打包值（scale_factor=0.001，即 mm/s），此处乘 0.001 转为 m/s。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
RAW_DIR = BASE / "海洋环境数据" / "HYCOM_GOFS3.1_2021_南海站点海流"
OUT = BASE / "清洗后数据" / "HYCOM_2021_南海站点海流_clean.csv"
URL = "https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0/uv3z"
SITES = {"珠海": (113.75, 22.25), "陵水": (110.00, 18.50)}
SCALE = 0.001
WRITE_LOCK = Lock()


def fetch_day(date: str, lo: float, la: float) -> list[list[str]]:
    params = [
        ("var", "water_u"),
        ("var", "water_v"),
        ("latitude", la),
        ("longitude", lo),
        ("time", f"{date}T00:00:00Z"),
        ("accept", "text/csv"),
    ]
    url = URL + "?" + urlencode(params)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "math-modeling-c-task/1.0"})
            with urlopen(req, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            if "Validation errors" in text or "html" in text[:200].lower():
                raise RuntimeError(text[:200])
            rows: list[list[str]] = []
            for line in text.splitlines()[1:]:
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                _, _, _, depth, u_raw, v_raw = parts[:6]
                u = float(u_raw) * SCALE if float(u_raw) > -30000 else ""
                v = float(v_raw) * SCALE if float(v_raw) > -30000 else ""
                speed = (
                    round(math.hypot(float(u), float(v)), 6)
                    if u != "" and v != ""
                    else ""
                )
                rows.append([depth, u, v, speed])
            return rows
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{date}: {last_err}")


def checkpoint_path(name: str) -> Path:
    return RAW_DIR / f"HYCOM_2021_{name}_uv3z_daily_raw.csv"


def load_done() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for name in SITES:
        path = checkpoint_path(name)
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] not in ("date", ""):
                    done.add((row[0], row[1]))
    return done


def save_checkpoint(name: str, date: str, rows: list[list[str]]) -> None:
    path = checkpoint_path(name)
    is_new = not path.exists() or path.stat().st_size == 0
    with WRITE_LOCK:
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["date", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
            for row in rows:
                w.writerow([date, name, *row])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 天（测试用）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true", help="跳过已断点落盘日期")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    days = [dt.date(2021, 1, 1) + dt.timedelta(days=i) for i in range(365)]
    if args.limit:
        days = days[: args.limit]

    done = load_done() if args.resume else set()
    tasks = []
    for d in days:
        for name, (lo, la) in SITES.items():
            key = (d.isoformat(), name)
            if key in done:
                continue
            tasks.append((*key, lo, la))

    print(f"tasks to fetch: {len(tasks)}", flush=True)
    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_day, date, lo, la): (date, name)
            for date, name, lo, la in tasks
        }
        for fut in as_completed(futures):
            date, name = futures[fut]
            rows = fut.result()
            save_checkpoint(name, date, rows)
            finished += 1
            if finished % 20 == 0:
                print(f"progress {finished}/{len(tasks)}", flush=True)

    all_rows: list[list[str]] = []
    for name in SITES:
        path = checkpoint_path(name)
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) == 6 and row[0] != "date":
                    all_rows.append(row)

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
        w.writerows(all_rows)
    print(f"clean -> {OUT.name} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
