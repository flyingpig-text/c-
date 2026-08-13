# -*- coding: utf-8 -*-
"""按 MIT SEAWATER v3.1.5 公式计算 35 g/kg 海水热物性表。

公式来源：SEAWATER_v3.1.5_07Aug24/seawater/MATLAB 下的
SW_Density.m、SW_SpcHeat.m、SW_Conductivity.m、SW_Viscosity.m、
SW_Kviscosity.m、SW_Prandtl.m。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path(r"D:\46884\Documents\数学建模\C题数据\清洗后数据\海水热物性_MIT_35gkg_clean.csv")


def density(t: float, s: float, p: float) -> float:
    """SW_Density(T,'C',S,'ppt',P,'MPa')，P 与 P0 均取 0.101325 MPa。"""
    p0 = 0.101325
    s_frac = s / 1000.0
    a = [9.9992293295e02, 2.0341179217e-02, -6.1624591598e-03,
         2.2614664708e-05, -4.6570659168e-08]
    b = [8.0200240891e02, -2.0005183488e00, 1.6771024982e-02,
         -3.0600536746e-05, -1.6132224742e-05]
    rho_w = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4
    d_rho = (b[0] * s_frac + b[1] * s_frac * t + b[2] * s_frac * t**2
             + b[3] * s_frac * t**3 + b[4] * s_frac**2 * t**2)
    rho_base = rho_w + d_rho
    c = [5.0792e-04, -3.4168e-06, 5.6931e-08, -3.7263e-10, 1.4465e-12,
         -1.7058e-15, -1.3389e-06, 4.8603e-09, -6.8039e-13]
    d = [-1.1077e-06, 5.5584e-09, -4.2539e-11, 8.3702e-09]
    f_p = (
        (p - p0)
        * (c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4
           + c[5] * t**5 + s * (d[0] + d[1] * t + d[2] * t**2))
        + 0.5 * (p**2 - p0**2) * (c[6] + c[7] * t + c[8] * t**3 + d[3] * s)
    )
    return rho_base * __import__("math").exp(f_p)


def specific_heat(t: float, s: float, p: float) -> float:
    """SW_SpcHeat(T,'C',S,'ppt',P,'MPa')。"""
    p0 = 0.101325
    t68 = 1.00024 * (t + 273.15)
    a = 5.328 - 9.76e-02 * s + 4.04e-04 * s**2
    b = -6.913e-03 + 7.351e-04 * s - 3.15e-06 * s**2
    c = 9.6e-06 - 1.927e-06 * s + 8.23e-09 * s**2
    d = 2.5e-09 + 1.666e-09 * s - 7.125e-12 * s**2
    cp_p0 = 1000.0 * (a + b * t68 + c * t68**2 + d * t68**3)
    c1, c2, c3, c4 = -3.1118, 0.0157, 5.1014e-05, -1.0302e-06
    c5, c6, c7, c8 = 0.0107, -3.9716e-05, 3.2088e-08, 1.0119e-09
    cp_p = (p - p0) * (c1 + c2 * t + c3 * t**2 + c4 * t**3
                       + s * (c5 + c6 * t + c7 * t**2 + c8 * t**3))
    return cp_p0 + cp_p


def conductivity(t: float, s: float) -> float:
    """SW_Conductivity(T,'C',S,'ppt')。"""
    t68 = 1.00024 * t
    s_p = s / 1.00472
    return 10.0 ** (
        __import__("math").log10(240.0 + 0.0002 * s_p)
        + 0.434 * (2.3 - (343.5 + 0.037 * s_p) / (t68 + 273.15))
        * (1 - (t68 + 273.15) / (647.3 + 0.03 * s_p)) ** (1 / 3)
        - 3
    )


def viscosity(t: float, s: float) -> float:
    """SW_Viscosity(T,'C',S,'ppt')。"""
    s_frac = s / 1000.0
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = (
        1.5700386464e-01, 6.4992620050e01, -9.1296496657e01,
        4.2844324477e-05, 1.5409136040e00, 1.9981117208e-02,
        -9.5203865864e-05, 7.9739318223e00, -7.5614568881e-02,
        4.7237011074e-04,
    )
    mu_w = a4 + 1.0 / (a1 * (t + a2) ** 2 + a3)
    a = a5 + a6 * t + a7 * t**2
    b = a8 + a9 * t + a10 * t**2
    return mu_w * (1 + a * s_frac + b * s_frac**2)


def main() -> None:
    rows = []
    for t in range(0, 41, 5):
        s = 35.0
        p = 0.101325
        rho = density(t, s, p)
        cp = specific_heat(t, s, p)
        k = conductivity(t, s)
        mu = viscosity(t, s)
        nu = mu / rho
        pr = cp * mu / k
        rows.append({
            "温度_degC": t,
            "盐度_g_kg": s,
            "压力_MPa": p,
            "密度_kg_m3": round(rho, 2),
            "比热容_J_kgK": round(cp, 1),
            "导热系数_W_mK": round(k, 4),
            "动力粘度_Pa_s": round(mu, 7),
            "运动粘度_m2_s": round(nu, 9),
            "普朗特数": round(pr, 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
