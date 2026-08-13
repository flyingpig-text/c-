# -*- coding: utf-8 -*-
"""拉取 WOA18 南海站点盐度剖面并计算 20℃ 等温线深度。

来源：APDRC OPeNDAP，WOA18 0.25°，1981-2010 年气候态，annual。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
URL = (
    "http://apdrc.soest.hawaii.edu:80/dods/public_data/WOA/WOA18/"
    "0.25_deg/annual/Avg_Decades_1981-2010/salt"
)
RAW = BASE / "海洋环境数据" / "WOA18_南海盐度剖面.csv"
CLEAN = BASE / "清洗后数据" / "WOA18_南海盐度剖面_clean.csv"
ISOTHERM = BASE / "清洗后数据" / "WOA18_20C等温线深度.csv"
TEMP_CLEAN = BASE / "清洗后数据" / "WOA18_南海温度剖面_clean.csv"
SITES = {"珠海": (113.75, 22.25), "陵水": (110.00, 18.50)}


def nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def window_mean(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray, lo: float, la: float, half: float = 0.75) -> float:
    lon_ok = np.abs(lon - lo) <= half
    lat_ok = np.abs(lat - la) <= half
    sub = grid[np.ix_(lat_ok, lon_ok)]
    vals = np.asarray(sub, dtype=float)
    vals = vals[~np.isnan(vals) & (vals > 0) & (vals < 45)]
    return float(np.mean(vals)) if vals.size else np.nan


def main() -> None:
    with Dataset(URL) as ds:
        lon = np.asarray(ds.variables["lon"][:])
        lat = np.asarray(ds.variables["lat"][:])
        lev = np.asarray(ds.variables["lev"][:])

    raw_rows = []
    for name, (lo, la) in SITES.items():
        i0 = max(nearest(lon, lo) - 3, 0)
        i1 = min(nearest(lon, lo) + 3, len(lon) - 1)
        j0 = max(nearest(lat, la) - 3, 0)
        j1 = min(nearest(lat, la) + 3, len(lat) - 1)
        with Dataset(URL) as ds:
            sub = np.asarray(
                ds.variables["san"][0, :, j0 : j1 + 1, i0 : i1 + 1]
            )
        lon_w = lon[i0 : i1 + 1]
        lat_w = lat[j0 : j1 + 1]
        for k, d in enumerate(lev):
            raw_rows.append([name, float(d), window_mean(sub[k], lon_w, lat_w, lo, la)])

    with RAW.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "深度_m", "盐度_PSU"])
        w.writerows(raw_rows)

    # 清洗：删除站点最大有效深度以下掩膜值
    max_depth = {"珠海": 50.0, "陵水": 100.0}
    clean_rows = [r for r in raw_rows if r[1] <= max_depth.get(r[0], 50.0)]
    with CLEAN.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "深度_m", "盐度_PSU"])
        w.writerows(clean_rows)

    # 20℃ 等温线深度（线性插值；超出剖面范围则用最底两段线性外推并标注）
    temp_rows = {}
    with TEMP_CLEAN.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            temp_rows.setdefault(r["站点"], []).append((float(r["深度_m"]), float(r["温度_degC"])))
    salt_rows = {}
    for r in clean_rows:
        salt_rows.setdefault(r[0], []).append((r[1], r[2]))

    iso_rows = []
    for name in SITES:
        prof = sorted(temp_rows[name])
        depths = [d for d, t in prof]
        temps = [t for d, t in prof]
        if min(temps) <= 20.0 <= max(temps):
            method = "插值"
            for i in range(len(prof) - 1):
                d0, t0 = prof[i]
                d1, t1 = prof[i + 1]
                if t0 >= 20.0 >= t1:
                    depth = d0 + (20.0 - t0) * (d1 - d0) / (t1 - t0)
                    break
        else:
            method = "外推"
            d0, t0 = prof[-2]
            d1, t1 = prof[-1]
            depth = d1 + (20.0 - t1) * (d1 - d0) / (t1 - t0)
        s = np.interp(depth, depths, [s for d, s in sorted(salt_rows[name])])
        iso_rows.append([name, round(depth, 1), round(s, 2), method])

    with ISOTHERM.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "20C等温线深度_m", "该深度盐度_PSU", "方法"])
        w.writerows(iso_rows)
    print(RAW.name, len(raw_rows), "rows")
    print(CLEAN.name, len(clean_rows), "rows")
    print(ISOTHERM.name, iso_rows)


if __name__ == "__main__":
    main()
