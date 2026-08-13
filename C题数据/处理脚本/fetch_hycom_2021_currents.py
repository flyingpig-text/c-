# -*- coding: utf-8 -*-
"""从 APDRC 下载 HYCOM GOFS 3.1 2021 年南海站点海流并生成清洗 CSV。

数据：HYCOM global daily snapshot 0Z 1/12 degree GOFS 3.1 (GLBy0.08) U/V
来源：http://apdrc.soest.hawaii.edu/dods/public_data/Model_output/HYCOM/gofs3.1/
说明：变量单位为 m/s（netCDF4 读取时已应用 scale_factor=0.001）。
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
URL = (
    "http://apdrc.soest.hawaii.edu:80/dods/public_data/Model_output/"
    "HYCOM/gofs3.1/hycom_GLBy0.08_a0_uv3z"
)
RAW_DIR = BASE / "海洋环境数据" / "HYCOM_GOFS3.1_2021_南海站点海流"
OUT = BASE / "清洗后数据" / "HYCOM_2021_南海站点海流_clean.csv"
SITES = {"珠海": (113.75, 22.25), "陵水": (110.00, 18.50)}
YEAR = 2021


def nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def window_mean(grid: np.ndarray, half: int = 2) -> float:
    """3x3/5x5 窗口内有效网格均值；-30000 为填缺值。"""
    vals = grid[~np.isnan(grid)]
    vals = vals[np.abs(vals) < 30]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with Dataset(URL) as ds:
        times = np.asarray(ds.variables["time"][:])
        lev = np.asarray(ds.variables["lev"][:])
        lat = np.asarray(ds.variables["lat"][:])
        lon = np.asarray(ds.variables["lon"][:])
        dates = num2date(
            times,
            units=ds.variables["time"].units,
            only_use_python_datetimes=True,
        )
        date0 = dt.datetime(YEAR, 1, 1)
        date1 = dt.datetime(YEAR, 12, 31)
        t0 = next(i for i, d in enumerate(dates) if d >= date0)
        t1 = next(
            len(dates) - 1 - i for i, d in enumerate(reversed(dates)) if d <= date1
        )

        all_rows: list[list[str | float]] = []
        for name, (lo, la) in SITES.items():
            i0 = max(nearest(lon, lo) - 2, 0)
            i1 = min(nearest(lon, lo) + 2, len(lon) - 1)
            j0 = max(nearest(lat, la) - 2, 0)
            j1 = min(nearest(lat, la) + 2, len(lat) - 1)
            u = np.asarray(
                ds.variables["water_u"][t0 : t1 + 1, :, j0 : j1 + 1, i0 : i1 + 1]
            )
            v = np.asarray(
                ds.variables["water_v"][t0 : t1 + 1, :, j0 : j1 + 1, i0 : i1 + 1]
            )
            u[u <= -30000] = np.nan
            v[v <= -30000] = np.nan
            raw_rows = []
            for ti, d in enumerate(dates[t0 : t1 + 1]):
                for zi, depth in enumerate(lev):
                    um = window_mean(u[ti, zi])
                    vm = window_mean(v[ti, zi])
                    speed = np.hypot(um, vm) if not np.isnan(um) and not np.isnan(vm) else np.nan
                    raw_rows.append([
                        d.strftime("%Y-%m-%d"), name, float(depth),
                        um, vm, speed,
                    ])
            all_rows.extend(raw_rows)
            raw_csv = RAW_DIR / f"HYCOM_2021_{name}_uv3z_raw.csv"
            with raw_csv.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["date", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
                w.writerows(raw_rows)
            print(f"{name}: {len(raw_rows)} records -> {raw_csv.name}")

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
        w.writerows(all_rows)
    print(f"clean -> {OUT.name} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
