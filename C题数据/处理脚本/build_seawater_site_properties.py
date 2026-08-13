# -*- coding: utf-8 -*-
"""按站点盐度重算 20℃ 海水热物性。

盐度来源：WOA18 20℃ 等温线深度处的盐度剖面值。
公式：MIT SEAWATER v3.1.5（复用 build_seawater_properties.py）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).parent))
from build_seawater_properties import conductivity, density, specific_heat, viscosity  # noqa: E402


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
ISOTHERM = BASE / "清洗后数据" / "WOA18_20C等温线深度.csv"
OUT = BASE / "清洗后数据" / "海水热物性_20C_站点盐度_clean.csv"


def main() -> None:
    site_salinity = {}
    with ISOTHERM.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            site_salinity[r["站点"]] = float(r["该深度盐度_PSU"])

    rows = []
    for label, s in [("标准海水", 35.0), *[(f"{k}站20℃深度", v) for k, v in site_salinity.items()]]:
        t = 20.0
        p = 0.101325
        rho = density(t, s, p)
        cp = specific_heat(t, s, p)
        k = conductivity(t, s)
        mu = viscosity(t, s)
        rows.append({
            "口径": label,
            "盐度_g_kg": round(s, 3),
            "温度_degC": t,
            "压力_MPa": p,
            "密度_kg_m3": round(rho, 2),
            "比热容_J_kgK": round(cp, 1),
            "导热系数_W_mK": round(k, 4),
            "动力粘度_Pa_s": round(mu, 7),
            "运动粘度_m2_s": round(mu / rho, 9),
            "普朗特数": round(cp * mu / k, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
