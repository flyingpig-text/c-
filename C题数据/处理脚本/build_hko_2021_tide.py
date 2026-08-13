# -*- coding: utf-8 -*-
"""解析香港天文台赤鱲角东 2021 年逐时潮汐预报并生成清洗 CSV。"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
HTML = BASE / "海洋环境数据" / "潮汐" / "HKO_ChekLapKokE_2021_hourly_tide.html"
RAW = BASE / "海洋环境数据" / "潮汐" / "HKO_ChekLapKokE_2021_hourly_tide.csv"
CLEAN = BASE / "清洗后数据" / "HKO_ChekLapKokE_2021_hourly_tide_clean.csv"
STATION = "赤鱲角东(ChekLapKokE)_2021天文潮预报"


def main() -> None:
    text = HTML.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"<TR><TD>(\d{2})</TD><TD>(\d{2})</TD><TD></TD>"
        + r"(<TD>[\s\-0-9.]*</TD>)" * 24
    )
    rows: list[list[str]] = []
    for m in pattern.finditer(text):
        month, day = m.group(1), m.group(2)
        raw = "".join(m.groups()[2:])
        heights = [float(x) for x in re.findall(r">\s*(-?[0-9.]+)\s*<", raw)]
        for hour, h in enumerate(heights, start=1):
            rows.append([f"2021-{month}-{day}", hour, h])

    with RAW.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "hour", "tide_height_m"])
        w.writerows(rows)

    clean_rows = []
    for date, hour, h in rows:
        if hour == 24:
            d = dt.date.fromisoformat(date) + dt.timedelta(days=1)
            clean_rows.append([STATION, f"{d.isoformat()} 00:00:00", h])
        else:
            clean_rows.append([STATION, f"{date} {hour:02d}:00:00", h])
    with CLEAN.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["站点", "datetime", "tide_height_m"])
        w.writerows(clean_rows)

    print(f"2021 HKO tide: {len(rows)} hourly records -> {CLEAN.name}")


if __name__ == "__main__":
    main()
