# -*- coding: utf-8 -*-
"""从 NOAA PSL GODAS 下载 2021 年月均海流并生成清洗 CSV。

数据：NCEP GODAS，月均，1° 网格，u/v（m/s），覆盖 2021 全年 12 个月。
来源：https://psl.noaa.gov/thredds/dodsC/Datasets/godas/ucur.2021.nc
说明：分辨率 1°×1°、月均，适合补全年季节/年际尺度；逐时细节需另用 HYCOM/CMEMS。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
RAW_DIR = BASE / "海洋环境数据" / "GODAS_2021_南海站点海流"
OUT = BASE / "清洗后数据" / "GODAS_2021_南海站点海流_clean.csv"
UCUR = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/ucur.2021.nc"
VCUR = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/vcur.2021.nc"
SITES = {"珠海": (113.75, 22.25), "陵水": (110.00, 18.50)}


def nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with Dataset(UCUR) as du, Dataset(VCUR) as dv:
        lon = np.asarray(du.variables["lon"][:])
        lat = np.asarray(du.variables["lat"][:])
        lev = np.asarray(du.variables["level"][:])
        times = np.asarray(du.variables["time"][:])
        months = num2date(
            times,
            units=du.variables["time"].units,
            only_use_python_datetimes=True,
        )
        rows: list[list[str | float]] = []
        for name, (lo, la) in SITES.items():
            i = nearest(lon, lo)
            j = nearest(lat, la)
            u = np.asarray(du.variables["ucur"][:, :, j, i])
            v = np.asarray(dv.variables["vcur"][:, :, j, i])
            u[u < -1e30] = np.nan
            v[v < -1e30] = np.nan
            raw_rows = []
            for ti, m in enumerate(months):
                for zi, depth in enumerate(lev):
                    um = float(u[ti, zi])
                    vm = float(v[ti, zi])
                    speed = (
                        round(math.hypot(um, vm), 6)
                        if not math.isnan(um) and not math.isnan(vm)
                        else ""
                    )
                    raw_rows.append([
                        m.strftime("%Y-%m"), name, float(depth), um, vm, speed,
                    ])
            rows.extend(raw_rows)
            raw = RAW_DIR / f"GODAS_2021_{name}_monthly_raw.csv"
            with raw.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["month", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
                w.writerows(raw_rows)
            print(f"{name}: {len(raw_rows)} records -> {raw.name}")

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["month", "站点", "深度_m", "u_m_s", "v_m_s", "流速_m_s"])
        w.writerows(rows)
    print(f"clean -> {OUT.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
