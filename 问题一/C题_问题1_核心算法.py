# -*- coding: utf-8 -*-
"""
C 题 问题 1 核心算法模块（独立函数版）

本文件只包含问题 1 的 5 步核心算法，不包含绘图与结果表，
因此不修改主脚本《C题_问题1_服务器散热_整数模型.py》。

计算顺序（交付清单统一口径）：
    步骤 1：圆柱外表面积 A = πDL
    步骤 2：自然对流 Nu 关联式 -> 换热系数 h
    步骤 3：Q_max = h A (T_max - T_∞)，N_theory = Q_max / q0
    步骤 4：空间上限 N_space = 内部容积 / 单台服务器体积
    步骤 5：N = floor(min(N_theory, N_space))，N_theory < 1 时输出 0

依赖：numpy；304 不锈钢导热系数优先从 清洗后数据 CSV 读取，
      仅当清洗后数据缺失时才回退到 C题数据 原始 HTML。
运行：python C题_问题1_核心算法.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 数据包定位与材料参数读取（优先使用 清洗后数据，禁止编造默认值）
# ---------------------------------------------------------------------------
def _find_data_dirs() -> tuple[Path, Path | None]:
    """自动定位 C题数据 与 清洗后数据（脚本同目录或上级目录）。"""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    data_dir = None
    clean_dir = None
    for _root in (here, here.parent, here.parent.parent):
        _cand = _root / "C题数据"
        if _cand.is_dir() and data_dir is None:
            data_dir = _cand
        _clean = _cand / "清洗后数据"
        if _clean.is_dir() and clean_dir is None:
            clean_dir = _clean
    return data_dir, clean_dir


DATA_DIR, CLEAN_DIR = _find_data_dirs()


def load_304_conductivity() -> float:
    """读取 304 不锈钢 20℃ 导热系数，W/(m·K)，优先使用清洗后 CSV。

    输入：无（自动读取 C题数据/清洗后数据）
    输出：float，304 不锈钢 20 ℃ 导热系数；数据文件缺失时报错，
          不使用编造的默认值。
    """
    if CLEAN_DIR is not None:
        p = CLEAN_DIR / "金属导热系数_EngineeringToolbox_clean.csv"
        if p.exists():
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("material") or "").strip() != "Steel - Stainless, Type 304":
                        continue
                    if abs(float(row["temperature_C"]) - 20.0) > 1e-9:
                        continue
                    return float(row["thermal_conductivity_W_per_mK"])
            raise ValueError("清洗后金属导热系数表中未找到 304 不锈钢 20℃ 数据")

    if DATA_DIR is not None:
        f = DATA_DIR / "材料数据" / "EngineeringToolbox_ThermalConductivity_Metals.html"
        if f.exists():
            raw = f.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))
            m = re.search(
                r"Steel\s*-\s*Stainless,\s*Type\s*304\s*([-0-9.]+)\s*([0-9.]+)",
                text,
            )
            if m:
                return float(m.group(2))

    raise FileNotFoundError(
        "未找到 C题数据/清洗后数据/金属导热系数_EngineeringToolbox_clean.csv，"
        "且无法回退到原始材料页，304 导热系数无法读取。"
    )


# ---------------------------------------------------------------------------
# 热物性（问题 1 采用 20 ℃ 附近工程近似；问题 3/4 建议调用完整海水物性库）
# ---------------------------------------------------------------------------
def air_props(T: float) -> dict[str, float]:
    """常压空气热物性近似，T 为摄氏温度。

    输入：T，摄氏温度 ℃
    输出：dict，含 rho(密度 kg/m^3)、cp(比热容 J/(kg·K))、
          k(导热系数 W/(m·K))、mu(动力粘度 Pa·s)、beta(膨胀系数 1/K)
    """
    Tk = T + 273.15
    rho = 101325.0 / (287.06 * Tk)
    cp = 1006.0
    k = 0.02439 + 0.0000792 * T
    mu = 1.72e-5 + 5.0e-8 * T
    beta = 1.0 / Tk
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "beta": beta}


def sea_props(T: float) -> dict[str, float]:
    """海水（S=35‰）热物性近似，T 为摄氏温度。

    输入：T，摄氏温度 ℃
    输出：dict，含 rho、cp、k、mu、beta（单位同上）
    """
    rho = 1027.0 - 0.24 * (T - 20.0)
    cp = 3985.0 + 0.35 * T
    k = 0.575 + 0.0016 * T
    mu = 0.00108 * np.exp(-0.019 * (T - 20.0))
    beta = 2.5e-4
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "beta": beta}


def h_horizontal_cylinder(D: float, dT: float, T_film: float,
                          props_fn) -> float:
    """水平圆柱自然对流换热系数，Churchill-Chu 关联式，W/(m^2·K)。

    输入：D 圆柱直径 m，dT 壁面与流体温差 ℃，T_film 膜温 ℃，
          props_fn 物性函数（air_props 或 sea_props）
    输出：float，自然对流换热系数 h，W/(m^2·K)
    """
    p = props_fn(T_film)
    g = 9.81
    nu = p["mu"] / p["rho"]
    alpha = p["k"] / (p["rho"] * p["cp"])
    Pr = nu / alpha
    Ra = g * p["beta"] * dT * D**3 / (nu * alpha)
    denom = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.60 + 0.387 * Ra ** (1.0 / 6.0) / denom) ** 2
    return Nu * p["k"] / D


# ---------------------------------------------------------------------------
# 核心算法：步骤 1-5 独立函数（输入 -> 处理 -> 输出）
# ---------------------------------------------------------------------------
def compute_areas(D_outer: float, L_outer: float, wall: float) -> dict[str, float]:
    """步骤 1：圆柱外表面积 A = πDL。

    输入：D_outer 外轮廓直径 m，L_outer 外轮廓长度 m，wall 壁厚 m
    输出：dict，含 D_inner/L_inner 内部尺寸 m，A_cyl 侧表面积 m^2，
          A_out 完整外表面积 m^2（含两端），A_in 完整内表面积 m^2
    """
    d_inner = D_outer - 2.0 * wall
    l_inner = L_outer - 2.0 * wall
    a_cyl = np.pi * D_outer * L_outer
    a_out = a_cyl + 2.0 * np.pi * D_outer**2 / 4.0
    a_in = np.pi * d_inner * l_inner + 2.0 * np.pi * d_inner**2 / 4.0
    return {"D_inner": d_inner, "L_inner": l_inner,
            "A_cyl": a_cyl, "A_out": a_out, "A_in": a_in}


def solve_heat_transfer(D_outer: float, L_outer: float,
                        D_inner: float, L_inner: float,
                        A_out: float, A_in: float,
                        k_wall: float, wall: float,
                        t_air_max: float, t_sea: float,
                        max_iter: int = 60, tol: float = 1e-6) -> dict[str, float]:
    """步骤 2：自然对流 Nu 关联式 -> 换热系数 h（热阻串联 + 壁温自洽迭代）。

    输入：D_outer/L_outer 外轮廓尺寸 m，D_inner/L_inner 内部尺寸 m，
          A_out/A_in 内外换热面积 m^2，k_wall 壁面导热系数 W/(m·K)，
          wall 壁厚 m，t_air_max 允许最高温度 ℃，t_sea 海水温度 ℃，
          max_iter 最大迭代次数，tol 收敛容差
    输出：dict，含 h_air/h_sea 空气/海水换热系数 W/(m^2·K)，
          h_total 综合换热系数 W/(m^2·K)，t_wi/t_wo 内/外壁温 ℃，
          r_total 总热阻 K/W，q_total 散热量 W
    """
    t_wi, t_wo = 40.0, 30.0
    for _ in range(max_iter):
        h_air = h_horizontal_cylinder(D_inner, t_air_max - t_wi,
                                      (t_air_max + t_wi) / 2.0, air_props)
        h_sea = h_horizontal_cylinder(D_outer, t_wo - t_sea,
                                      (t_wo + t_sea) / 2.0, sea_props)
        r_air = 1.0 / (h_air * A_in)
        r_wall = (np.log(D_outer / D_inner) / (2.0 * np.pi * k_wall * L_outer)
                  + 2.0 * wall / (k_wall * np.pi * D_inner**2 / 4.0))
        r_sea = 1.0 / (h_sea * A_out)
        r_total = r_air + r_wall + r_sea
        q_total = (t_air_max - t_sea) / r_total
        t_wi_new = t_air_max - q_total * r_air
        t_wo_new = t_sea + q_total * r_sea
        if abs(t_wi_new - t_wi) < tol and abs(t_wo_new - t_wo) < tol:
            t_wi, t_wo = t_wi_new, t_wo_new
            break
        t_wi, t_wo = t_wi_new, t_wo_new
    h_total = 1.0 / (r_total * A_out)
    return {"h_air": h_air, "h_sea": h_sea, "h_total": h_total,
            "t_wi": t_wi, "t_wo": t_wo,
            "r_total": r_total, "q_total": q_total}


def compute_thermal_capacity(h_total: float, a_eff: float,
                             t_air_max: float, t_sea: float,
                             q0: float) -> dict[str, float]:
    """步骤 3：最大总散热量 Q_max = h A ΔT 与散热理论上限 N_theory。

    输入：h_total 综合换热系数 W/(m^2·K)，a_eff 有效总散热面积 m^2，
          t_air_max 允许最高温度 ℃，t_sea 海水温度 ℃，q0 单台产热 W
    输出：dict，含 q_max 最大总散热量 W，n_theory 散热理论上限 台
    """
    q_max = h_total * a_eff * (t_air_max - t_sea)
    n_theory = q_max / q0
    return {"q_max": q_max, "n_theory": n_theory}


def compute_spatial_capacity(D_inner: float, L_inner: float,
                             w_server: float, h_server: float,
                             l_server: float) -> dict[str, float]:
    """步骤 4：空间上限 N_space = 舱内部容积 / 单台服务器体积。

    输入：D_inner/L_inner 内部尺寸 m，w_server/h_server/l_server 服务器尺寸 m
    输出：dict，含 v_inner 内部可用容积 m^3，v_server 单台体积 m^3，
          n_space 空间上限 台
    """
    v_inner = np.pi * D_inner**2 / 4.0 * L_inner
    v_server = w_server * h_server * l_server
    n_space = v_inner / v_server
    return {"v_inner": v_inner, "v_server": v_server, "n_space": n_space}


def compute_final_capacity(n_theory: float, n_space: float) -> dict[str, int]:
    """步骤 5：退化整数规划取整，N = floor(min(N_theory, N_space))。

    输入：n_theory 散热理论上限 台，n_space 空间上限 台
    输出：dict，含 n 最终可容纳服务器台数（整数）；
          异常判定：若 n_theory < 1，输出 0
    """
    if n_theory < 1.0:
        n = 0
    else:
        n = int(np.floor(min(n_theory, n_space)))
    return {"n": n}


def solve_q1(D_outer: float = 1.0, L_outer: float = 12.0, wall: float = 0.01,
             k_wall: float | None = None,
             t_air_max: float = 80.0, t_sea: float = 20.0, q0: float = 500.0,
             w_server: float = 0.4826, h_server: float = 0.04445,
             l_server: float = 0.525) -> dict:
    """问题 1 主流程：依次调用步骤 1-5，返回全部中间量与最终容量。

    输入：D_outer/L_outer 外轮廓尺寸 m，wall 壁厚 m（交付清单未给壁厚，
          按用户确认基准取 w=0.01 m，壁厚影响单独做灵敏度分析），
          k_wall 壁面导热系数 W/(m·K)（缺省时自动解析 304 不锈钢），
          t_air_max 允许最高温度 ℃，t_sea 海水温度 ℃，q0 单台产热 W，
          w_server/h_server/l_server 服务器尺寸 m
    输出：dict，含 D_inner/L_inner、A_cyl/A_out/A_in、h_air/h_sea/h_total、
          t_wi/t_wo、q_max、n_theory、v_inner、v_server、n_space、n
    """
    if k_wall is None:
        k_wall = load_304_conductivity()
    areas = compute_areas(D_outer, L_outer, wall)
    ht = solve_heat_transfer(D_outer, L_outer,
                             areas["D_inner"], areas["L_inner"],
                             areas["A_out"], areas["A_in"],
                             k_wall, wall, t_air_max, t_sea)
    thermal = compute_thermal_capacity(ht["h_total"], areas["A_out"],
                                       t_air_max, t_sea, q0)
    spatial = compute_spatial_capacity(areas["D_inner"], areas["L_inner"],
                                       w_server, h_server, l_server)
    final = compute_final_capacity(thermal["n_theory"], spatial["n_space"])
    return {**areas, **ht, **thermal, **spatial, **final, "k_wall": k_wall}


# ---------------------------------------------------------------------------
# 自检与命令行运行
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """手工校验示例：D=1、L=12、h=5、ΔT=60、q0=500、体积上限 800。"""
    q_max = 5.0 * (np.pi * 1.0 * 12.0) * (80.0 - 20.0)   # = 11309.73 W
    n_theory = q_max / 500.0                              # = 22.6195 台
    n_space = 800.0
    n = compute_final_capacity(n_theory, n_space)["n"]
    ok = (abs(q_max - 11309.73) < 0.01
          and abs(n_theory - 22.62) < 0.01
          and n == 22)
    print(f"自检：Q_max={q_max:.2f} W，N_theory={n_theory:.2f} 台，"
          f"N={n} 台 -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """先跑通简单基准算例，再运行完整模型。"""
    print("=" * 56)
    print("问题 1 核心算法（独立函数版）")
    print("=" * 56)
    print("0. 基准算例（先跑通，再扩展完整模型）")
    _self_check()

    print("1~5. 完整模型")
    r = solve_q1()
    print(f"步骤 1  A = πDL = {r['A_cyl']:.4f} m^2，"
          f"A_eff = {r['A_out']:.4f} m^2")
    print(f"步骤 2  h_air = {r['h_air']:.2f}，h_sea = {r['h_sea']:.1f}，"
          f"h_total = {r['h_total']:.2f} W/(m^2·K)")
    print(f"        壁温 T_wi = {r['t_wi']:.1f} ℃，"
          f"T_wo = {r['t_wo']:.1f} ℃")
    print(f"步骤 3  Q_max = {r['q_max']:.0f} W，"
          f"N_theory = {r['n_theory']:.2f} 台")
    print(f"步骤 4  N_space = {r['n_space']:.2f} 台")
    print(f"步骤 5  N = {r['n']} 台")
    print("=" * 56)


if __name__ == "__main__":
    main()
