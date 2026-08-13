"""Convert downloaded HKO tide and NOAA SST data into tidy CSV files."""

from __future__ import annotations

import csv
import re
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
ZHUHAI = (113.75, 22.25)   # (lon, lat) near Zhuhai
LINGSHUI = (110.00, 18.50)  # (lon, lat) near Lingshui, Hainan
BOX = dict(lon=(108.0, 118.0), lat=(15.0, 25.0))


def parse_hko_hourly_tide(html_path: Path, csv_path: Path) -> None:
    """Parse HKO hourly predicted tide height tables (MM/DD + 24 hourly heights)."""
    text = html_path.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[str, str, float]] = []
    pattern = re.compile(
        r"<TR><TD>(\d{2})</TD><TD>(\d{2})</TD><TD></TD>" + r"(<TD>[\s\-0-9.]*</TD>)" * 24
    )
    for m in pattern.finditer(text):
        month, day = m.group(1), m.group(2)
        raw = "".join(m.groups()[2:])
        heights = [float(x) for x in re.findall(r">\s*(-?[0-9.]+)\s*<", raw)]
        for hour, h in enumerate(heights, start=1):
            rows.append((f"2026-{month}-{day}", hour, h))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "hour", "tide_height_m"])
        w.writerows(rows)
    print(f"HKO tide: {len(rows)} hourly records -> {csv_path.name}")


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def read_ersst_sst(nc_path: Path) -> tuple[dict[str, float], float]:
    """Return {point_name: sst} for the two sites and the box mean SST."""
    from netCDF4 import Dataset

    with Dataset(nc_path) as ds:
        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]
        sst = ds.variables["sst"][:]
    sst = np.squeeze(sst)
    fill = -999.0
    out: dict[str, float] = {}
    for name, (lo, la) in {"珠海": ZHUHAI, "陵水": LINGSHUI}.items():
        i = nearest_index(lon, lo)
        j = nearest_index(lat, la)
        v = float(sst[j, i]) if sst.ndim == 2 else float(sst[0, j, i])
        out[name] = v if v != fill else np.nan
    mask = (
        (lon >= BOX["lon"][0])
        & (lon <= BOX["lon"][1])
        & (lat[:, None] >= BOX["lat"][0])
        & (lat[:, None] <= BOX["lat"][1])
    )
    box = sst[mask]
    out["南海区域平均"] = float(np.nanmean(box[box != fill])) if box.size else np.nan
    return out, fill


def build_ersst_csv() -> None:
    folder = BASE / "海洋环境数据" / "ERSST_v5"
    ascii_folder = Path(tempfile.gettempdir()) / "cdata_ersst"
    ascii_folder.mkdir(exist_ok=True)
    out_rows = []
    for path in sorted(folder.glob("ersst.v5.20*.nc")):
        tmp = ascii_folder / path.name
        shutil.copy2(path, tmp)
        ym = path.stem.removeprefix("ersst.v5.")
        year, month = int(ym[:4]), int(ym[4:])
        vals, _ = read_ersst_sst(tmp)
        out_rows.append([f"{year}-{month:02d}", vals["珠海"], vals["陵水"], vals["南海区域平均"]])
    out = BASE / "海洋环境数据" / "ERSST_v5_2020-2021_南海站点SST.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["month", "珠海_SST_degC", "陵水_SST_degC", "南海区域平均_SST_degC"])
        w.writerows(out_rows)
    print(f"ERSST: {len(out_rows)} monthly rows -> {out.name}")


def build_oisst_csv() -> None:
    folder = BASE / "海洋环境数据" / "OISST_每日SST"
    files = sorted(folder.glob("oisst-avhrr-v02r01.2*.nc"))
    if not files:
        print("OISST: no complete files; skip")
        return
    ascii_folder = Path(tempfile.gettempdir()) / "cdata_oisst"
    ascii_folder.mkdir(exist_ok=True)
    out_rows = []
    for path in files:
        tmp = ascii_folder / path.name
        shutil.copy2(path, tmp)
        from netCDF4 import Dataset

        with Dataset(tmp) as ds:
            lon = ds.variables["lon"][:]
            lat = ds.variables["lat"][:]
            sst = np.squeeze(ds.variables["sst"][:])
        if sst.ndim == 3:
            sst = sst[0]
        fill = -999.0
        date = path.stem.removeprefix("oisst-avhrr-v02r01.")
        out = []
        for name, (lo, la) in {"珠海": ZHUHAI, "陵水": LINGSHUI}.items():
            i, j = nearest_index(lon, lo), nearest_index(lat, la)
            v = float(sst[j, i])
            out.append(v if v != fill else np.nan)
        mask = (
            (lon >= BOX["lon"][0])
            & (lon <= BOX["lon"][1])
            & (lat[:, None] >= BOX["lat"][0])
            & (lat[:, None] <= BOX["lat"][1])
        )
        box = sst[mask]
        out.append(float(np.nanmean(box[box != fill])) if box.size else np.nan)
        out_rows.append([f"{date[:4]}-{date[4:6]}-{date[6:]}", *out])
    out_csv = BASE / "海洋环境数据" / "OISST_2020-2021_每月中旬_南海站点SST.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "珠海_SST_degC", "陵水_SST_degC", "南海区域平均_SST_degC"])
        w.writerows(out_rows)
    print(f"OISST: {len(out_rows)} daily rows -> {out_csv.name}")


def build_woa18_csv(nc_path: Path) -> None:
    from netCDF4 import Dataset

    ascii_path = Path(tempfile.gettempdir()) / "woa18_decav_t00_01.nc"
    shutil.copy2(nc_path, ascii_path)
    with Dataset(ascii_path) as ds:
        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]
        depth = ds.variables["depth"][:]
        t_an = ds.variables["t_an"][:]
    t_an = np.squeeze(t_an)  # (depth, lat, lon) after squeeze
    fill = -999.0
    rows = []
    for name, (lo, la) in {"珠海": ZHUHAI, "陵水": LINGSHUI}.items():
        i = nearest_index(lon, lo)
        j = nearest_index(lat, la)
        for d, v in zip(depth, t_an[:, j, i]):
            rows.append([name, float(d), float(v) if v != fill else np.nan])
    out = BASE / "海洋环境数据" / "WOA18_南海温度剖面.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "深度_m", "温度_degC"])
        w.writerows(rows)
    print(f"WOA18: {len(rows)} profile records -> {out.name}")


def main() -> None:
    parse_hko_hourly_tide(
        BASE / "海洋环境数据" / "潮汐" / "HKO_ChekLapKokE_2026_每小时潮高.html",
        BASE / "海洋环境数据" / "潮汐" / "HKO_ChekLapKokE_2026_hourly_tide.csv",
    )
    build_ersst_csv()
    build_oisst_csv()
    woa = BASE / "海洋环境数据" / "WOA18" / "woa18_decav_t00_01.nc"
    if woa.exists() and woa.stat().st_size > 100_000_000:
        build_woa18_csv(woa)
    else:
        print("WOA18 file not complete yet; skip profile extraction")


if __name__ == "__main__":
    sys.exit(main())
