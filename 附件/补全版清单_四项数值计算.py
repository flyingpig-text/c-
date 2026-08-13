# -*- coding: utf-8 -*-
"""复算交付清单中四类可落成数值的项：
1) 问题 1 基准算例输出（读 Q1_结果.csv）；
2) 珠海/陵水 水深-温度两段连续线性模型；
3) 香港赤鱲角东 2021 逐时潮汐的标准分潮谐波拟合；
4) GODAS 2021 珠海/陵水月均海流流速。

依赖：numpy、pandas（bundled runtime 已具备）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\46884\Documents\数学建模")
CLEAN = ROOT / "C题数据" / "清洗后数据"


def q1_baseline() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "问题一" / "outputs" / "Q1_结果.csv", encoding="utf-8-sig")
    keys = [
        "壁厚 w", "有效散热面积 A_eff", "内壁换热面积 A_in",
        "舱内空气 h_air", "海水 h_sea", "综合 h_total",
        "最大散热量 Q_max", "散热理论上限 N_theory", "空间上限 N_space",
        "最终容量 N", "内壁温 T_wi", "外壁温 T_wo",
        "内部容积 V_inner", "单台体积 V_server",
    ]
    out = df[df["项目"].isin(keys)][["项目", "数值", "单位"]].copy()
    return out


def fit_two_segment(depth: np.ndarray, temp: np.ndarray) -> dict:
    """T(d)=c0+c1*d+c2*max(d-d1,0)，连续性由该形式保证。"""
    candidates = np.linspace(depth.min() + 1.0, depth.max() - 1.0, 200)
    best = None
    for d1 in candidates:
        x = np.column_stack([np.ones_like(depth), depth, np.maximum(depth - d1, 0.0)])
        coef, *_ = np.linalg.lstsq(x, temp, rcond=None)
        pred = x @ coef
        sse = float(np.sum((temp - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"d1": d1, "c0": coef[0], "c1": coef[1], "c2": coef[2],
                    "sse": sse, "pred": pred}
    resid = temp - best["pred"]
    ss_tot = float(np.sum((temp - temp.mean()) ** 2))
    r2 = 1.0 - best["sse"] / ss_tot
    return {
        "d1_m": best["d1"],
        "T0_degC": best["c0"],
        "k1_degC_per_m": best["c1"],
        "k2_degC_per_m": best["c1"] + best["c2"],
        "T_at_d1_degC": best["c0"] + best["c1"] * best["d1"],
        "R2": r2,
        "max_abs_resid_degC": float(np.abs(resid).max()),
        "rmse_degC": float(np.sqrt(best["sse"] / len(temp))),
    }


def temperature_piecewise() -> pd.DataFrame:
    df = pd.read_csv(CLEAN / "WOA18_南海温度剖面_clean.csv", encoding="utf-8-sig")
    rows = []
    for site in ["珠海", "陵水"]:
        sub = df[df["站点"] == site].sort_values("深度_m")
        res = fit_two_segment(sub["深度_m"].to_numpy(float),
                              sub["温度_degC"].to_numpy(float))
        res = {"站点": site, **res}
        rows.append(res)
    return pd.DataFrame(rows)


TIDAL_SPEEDS_DEG_PER_H = {
    "Q1": 13.3986609, "O1": 13.9430356, "P1": 14.9589314,
    "K1": 15.0410686, "N2": 28.4397295, "M2": 28.9841042,
    "S2": 30.0000000, "K2": 30.0821373, "M4": 57.9682084,
    "MS4": 58.9841042, "M6": 86.9523127, "Mf": 1.0980331,
    "Mm": 0.5443747,
}


def tidal_harmonic() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(CLEAN / "HKO_ChekLapKokE_2021_hourly_tide_clean.csv",
                     encoding="utf-8-sig")
    t = pd.to_datetime(df["datetime"])
    t0 = pd.Timestamp("2021-01-01 00:00:00")
    hours = (t - t0).dt.total_seconds().to_numpy(float) / 3600.0
    h = df["tide_height_m"].to_numpy(float)

    cols = [np.ones_like(h)]
    for name, speed in TIDAL_SPEEDS_DEG_PER_H.items():
        omega = math.radians(speed) * hours
        cols.append(np.cos(omega))
        cols.append(np.sin(omega))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, h, rcond=None)

    rows = []
    total_expl_var = 0.0
    names = list(TIDAL_SPEEDS_DEG_PER_H)
    for i, name in enumerate(names):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        amp = math.hypot(a, b)
        phase = math.degrees(math.atan2(b, a)) % 360.0
        speed = TIDAL_SPEEDS_DEG_PER_H[name]
        rows.append({"分潮": name, "角速度_deg_per_h": speed,
                     "周期_h": 360.0 / speed, "振幅_m": amp,
                     "相位_deg": phase, "方差贡献": amp * amp / 2.0})
        total_expl_var += amp * amp / 2.0
    result = pd.DataFrame(rows).sort_values("振幅_m", ascending=False).reset_index(drop=True)
    result["方差占比_pct"] = result["方差贡献"] / total_expl_var * 100.0

    pred = design @ coef
    resid = h - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((h - h.mean()) ** 2))
    summary = {
        "z0_m": float(coef[0]),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse_m": float(np.sqrt(ss_res / len(h))),
        "max_abs_resid_m": float(np.abs(resid).max()),
        "n_hours": len(h),
    }

    top6 = result.head(6)
    keep = ["Z0"] + [f"{name}_c" for name in top6["分潮"]] \
        + [f"{name}_s" for name in top6["分潮"]]
    keep_cols = [np.ones_like(h)]
    for name in top6["分潮"]:
        omega = math.radians(TIDAL_SPEEDS_DEG_PER_H[name]) * hours
        keep_cols.append(np.cos(omega))
        keep_cols.append(np.sin(omega))
    c6, *_ = np.linalg.lstsq(np.column_stack(keep_cols), h, rcond=None)
    pred6 = np.column_stack(keep_cols) @ c6
    r2_6 = 1.0 - float(np.sum((h - pred6) ** 2)) / ss_tot
    summary["r2_top6"] = r2_6
    return result, summary


def monthly_currents() -> pd.DataFrame:
    df = pd.read_csv(CLEAN / "GODAS_2021_南海站点海流_clean.csv", encoding="utf-8-sig")
    rows = []
    for month in sorted(df["month"].unique()):
        row = {"月份": month}
        for site, up_to, layer in (("珠海", 50.0, [45.0, 55.0]),
                                   ("陵水", 100.0, [95.0, 105.0])):
            sub = df[(df["站点"] == site) & (df["month"] == month)]
            near = sub[sub["深度_m"] <= up_to]
            dep = sub[sub["深度_m"].isin(layer)]
            row[f"{site}_0-{int(up_to)}m_流速"] = near["流速_m_s"].mean()
            row[f"{site}_部署层_流速"] = dep["流速_m_s"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    annual = {"月份": "全年平均"}
    for col in out.columns[1:]:
        annual[col] = out[col].mean()
    return pd.concat([out, pd.DataFrame([annual])], ignore_index=True)


def fmt_table(df: pd.DataFrame, digits: int = 4) -> str:
    lines = ["| " + " | ".join(df.columns) + " |",
             "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for col in df.columns:
            v = r[col]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    print("========== 1. 问题 1 基准算例 ==========")
    print(fmt_table(q1_baseline(), digits=4))
    print()

    print("========== 2. 水深-温度两段连续线性模型 ==========")
    print(fmt_table(temperature_piecewise(), digits=4))
    print()

    print("========== 3. 2021 潮汐谐波拟合 ==========")
    tide_df, tide_sum = tidal_harmonic()
    print(f"Z0={tide_sum['z0_m']:.4f} m，R2={tide_sum['r2']:.6f}，"
          f"RMSE={tide_sum['rmse_m']:.4f} m，max|resid|={tide_sum['max_abs_resid_m']:.4f} m")
    print(f"前 6 分潮模型 R2={tide_sum['r2_top6']:.6f}")
    print(fmt_table(tide_df, digits=4))
    print()

    print("========== 4. GODAS 2021 月均海流流速 ==========")
    print(fmt_table(monthly_currents(), digits=4))


if __name__ == "__main__":
    main()
