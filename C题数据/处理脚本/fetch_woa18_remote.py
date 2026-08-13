"""Fetch WOA18 0.25-degree temperature climatology via APDRC OPeNDAP."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


BASE = Path(r"D:\46884\Documents\数学建模\C题数据\海洋环境数据")
ANNUAL_URL = (
    "http://apdrc.soest.hawaii.edu:80/dods/public_data/WOA/WOA18/"
    "0.25_deg/annual/Avg_Decades_1981-2010/temp"
)
MONTHLY_URL = (
    "http://apdrc.soest.hawaii.edu:80/dods/public_data/WOA/WOA18/"
    "0.25_deg/monthly/Avg_Decades_1981-2010/temp"
)
SITES = {"珠海": (113.75, 22.25), "陵水": (110.00, 18.50)}


def nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def window_mean(
    grid: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    lo: float,
    la: float,
    half_deg: float = 0.75,
) -> float:
    """Mean of valid ocean cells within +/-half_deg of a coastal site."""
    lon_ok = np.abs(lon - lo) <= half_deg
    lat_ok = np.abs(lat - la) <= half_deg
    if np.count_nonzero(lon_ok) == 0 or np.count_nonzero(lat_ok) == 0:
        return np.nan
    sub = grid[np.ix_(lat_ok, lon_ok)]
    if sub.size == 0:
        return np.nan
    vals = np.asarray(sub, dtype=float)
    vals = vals[~np.isnan(vals) & (np.abs(vals) < 100)]
    return float(np.mean(vals)) if vals.size else np.nan


def annual_profiles() -> None:
    with Dataset(ANNUAL_URL) as ds:
        lon = np.asarray(ds.variables["lon"][:])
        lat = np.asarray(ds.variables["lat"][:])
        lev = np.asarray(ds.variables["lev"][:])
        sub = {}
        for name, (lo, la) in SITES.items():
            i0 = max(nearest(lon, lo) - 3, 0)
            i1 = min(nearest(lon, lo) + 3, len(lon) - 1)
            j0 = max(nearest(lat, la) - 3, 0)
            j1 = min(nearest(lat, la) + 3, len(lat) - 1)
            sub[name] = (
                lon[i0 : i1 + 1],
                lat[j0 : j1 + 1],
                ds.variables["tan"][0, :, j0 : j1 + 1, i0 : i1 + 1],
            )
    out = BASE / "WOA18_南海温度剖面.csv"
    rows = []
    for name, (lo, la) in SITES.items():
        lon_w, lat_w, tan_w = sub[name]
        for k, d in enumerate(lev):
            rows.append([name, float(d), window_mean(tan_w[k], lon_w, lat_w, lo, la, 0.75)])
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "深度_m", "温度_degC"])
        w.writerows(rows)
    print(f"WOA18 annual profiles -> {out.name} ({len(rows)} rows)")


def monthly_surface() -> None:
    with Dataset(MONTHLY_URL) as ds:
        lon = np.asarray(ds.variables["lon"][:])
        lat = np.asarray(ds.variables["lat"][:])
        lev = np.asarray(ds.variables["lev"][:])
        surface_idx = nearest(lev, 0.0)
        out = BASE / "WOA18_1981-2010_月均表层温度_南海站点.csv"
        rows = []
        for name, (lo, la) in SITES.items():
            i0 = max(nearest(lon, lo) - 3, 0)
            i1 = min(nearest(lon, lo) + 3, len(lon) - 1)
            j0 = max(nearest(lat, la) - 3, 0)
            j1 = min(nearest(lat, la) + 3, len(lat) - 1)
            tan_w = ds.variables["tan"][
                :, surface_idx : surface_idx + 1, j0 : j1 + 1, i0 : i1 + 1
            ][:, 0, :, :]
            lon_w = lon[i0 : i1 + 1]
            lat_w = lat[j0 : j1 + 1]
            vals = [
                window_mean(tan_w[month], lon_w, lat_w, lo, la, 0.75)
                for month in range(12)
            ]
            rows.append(vals)
    out_rows = []
    for month in range(12):
        out_rows.append([f"{month + 1:02d}", rows[0][month], rows[1][month]])
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["月", "珠海_表层温度_degC", "陵水_表层温度_degC"])
        w.writerows(out_rows)
    print(f"WOA18 monthly climatology -> {out.name} ({len(out_rows)} rows)")


def main() -> None:
    annual_profiles()
    monthly_surface()


if __name__ == "__main__":
    main()
