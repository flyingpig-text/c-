# -*- coding: utf-8 -*-
"""
C 题 问题 2 核心算法模块（独立函数版）

本文件只包含问题 2 的核心算法：外形/翅片几何 -> 热阻网络（与问题 1
统一口径）-> 有效散热面积 -> 散热/空间上限 -> 最终容量。不包含
GA/SLSQP 优化与绘图，主脚本《C题_问题2_枚举shape_GA_SLSQP.py》
通过 import 调用本模块。

计算顺序（评审修正后的统一口径）：
    步骤 1：翅片/基体几何：
        A_fin = (2*Hf + df)*L（单根，含两侧 + 翅尖）
        A_base = A_侧 + A_端 - N_f*df*L
        A_eff = A_base + eta_f * N_f * A_fin
    步骤 2：热阻网络（壁温自洽迭代）：
        R_total = 1/(h_air*A_in) + R_wall + 1/(h_sea*A_eff)
    步骤 3：Q_max = h_total * A_eff * (T_max - T_inf)
            （等价于 (T_max - T_inf)/R_total，统一口径与问题 1 一致），
            N_theory = Q_max / q0
    步骤 4：N_space = V_inner / V_server（体积法上界）
    步骤 5：N = floor(min(N_theory, N_space))，N_theory < 1 时输出 0

依赖：numpy；304 不锈钢导热系数优先从 C题数据/清洗后数据 CSV 读取，
      缺失时回退工程默认值 14.4 W/(m·K)。
运行：python C题_问题2_核心算法.py（自检）
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ==================================================================
# 一、题目给定参数（全部来自题面 / 交付清单，勿改）
# ==================================================================
ENVELOPE_SIDE = 1.0      # 外形约束：横截面不超过 1 m（1m x 1m x 12m）
HULL_LENGTH = 12.0       # 舱体总长 L，m
Q0 = 500.0               # 单台服务器产热，W
T_MAX = 80.0             # 壳体允许最高温度，℃
T_INF = 20.0             # 问题 1 基准海水温度，℃

# 1U 服务器尺寸（宽 x 高 x 长），m
SERVER_W = 0.4826
SERVER_H = 0.04445
SERVER_L = 0.525


# ==================================================================
# 二、材料与海水热物性（数据来源见数据.md；与问题一模型保持同一套近似式）
# ==================================================================
K_FIN = 167.0            # 翅片材料导热系数（6061-T6 铝；MatWeb 167，MakeItFrom 170，取 167）
WALL = 0.01              # 壳体壁厚，m（沿用问题一默认值 10 mm）
K_WALL_DEFAULT = 14.4    # 壳体材料导热系数默认值（304 不锈钢 20 ℃）；优先读清洗后 CSV


def _find_project_root(start: Path | None = None) -> Path:
    """自动寻找工程根目录（含 outputs / C题数据 的上级目录）。"""
    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents]
    if start is not None:
        candidates.insert(0, Path(start).resolve())
    for p in candidates:
        if (p / "outputs").is_dir() or (p / "C题数据").is_dir():
            return p
    return here.parent


def air_props(T: float) -> dict:
    """常压空气热物性近似，T 为摄氏温度（与问题 1 核心算法同一套公式）。"""
    Tk = T + 273.15
    rho = 101325.0 / (287.06 * Tk)
    cp = 1006.0
    k = 0.02439 + 0.0000792 * T
    mu = 1.72e-5 + 5.0e-8 * T
    beta = 1.0 / Tk
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "beta": beta}


def sea_props(T: float) -> dict:
    """海水（S=35‰）热物性近似，T 为摄氏温度。

    与问题一脚本《C题_问题1_核心算法.py》保持同一套近似式：
        rho: 密度 kg/m^3
        cp : 比热容 J/(kg·K)
        k  : 导热系数 W/(m·K)
        mu : 动力粘度 Pa·s
        beta: 体膨胀系数 1/K
    """
    rho = 1027.0 - 0.24 * (T - 20.0)
    cp = 3985.0 + 0.35 * T
    k = 0.575 + 0.0016 * T
    mu = 0.00108 * np.exp(-0.019 * (T - 20.0))
    beta = 2.5e-4
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "beta": beta}


def h_horizontal_cylinder(D: float, dT: float, T_film: float,
                          props_fn=sea_props) -> float:
    """水平圆柱自然对流换热系数 h，Churchill-Chu 关联式，W/(m^2·K)。

    输入：
        D      : 特征长度（圆柱直径，或方柱等面积当量直径），m
        dT     : 壁面与流体温差，℃
        T_film : 膜温（取壁温与流体温度平均值），℃
        props_fn : 物性函数，默认 sea_props（海水），可传 air_props（舱内空气）
    公式：
        Ra = g*beta*dT*D^3/(nu*alpha)
        Nu = [0.60 + 0.387*Ra^(1/6)/(1+(0.559/Pr)^(9/16))^(8/27)]^2
        h  = Nu*k/D
    """
    p = props_fn(T_film)
    g = 9.81
    nu = p["mu"] / p["rho"]                     # 运动粘度
    alpha = p["k"] / (p["rho"] * p["cp"])       # 热扩散率
    Pr = nu / alpha
    Ra = g * p["beta"] * dT * D**3 / (nu * alpha)
    denom = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.60 + 0.387 * Ra ** (1.0 / 6.0) / denom) ** 2
    return Nu * p["k"] / D


# ==================================================================
# 三、翅片设计变量上下界（清单未给数值，以下为工程默认值【假设值】）
# ==================================================================
NF_MIN, NF_MAX = 1, 160          # 每根/每面翅片数（长方体 = 每面数量）
HF_MIN, HF_MAX = 0.005, 0.24     # 翅高上下界，m（0.24 保证基体仍有足够内空间）
DF_MIN, DF_MAX = 0.001, 0.01     # 翅厚上下界，m


# ==================================================================
# 四、翅片几何与传热模型（圆柱 / 长方体统一入口）
# ==================================================================
def _char_len(shape: str, base_side: float) -> float:
    """Churchill-Chu 特征长度：圆柱取直径；长方体取等面积当量直径 2a/sqrt(pi)。"""
    if shape == "cylinder":
        return base_side
    return 2.0 * base_side / np.sqrt(np.pi)


def _build_geometry(shape: str, Hf: float) -> dict:
    """由翅高 Hf 反推基体尺寸（翅尖刚好贴住 1m 外形约束）。

    几何约定：
        圆柱    D_base = 1 - 2*Hf，纵向矩形直翅沿圆周均布，翅长 = L；
        长方体  a_base = 1 - 2*Hf，四个侧面各布 nf 根纵向矩形直翅。
    这样翅片“向外延展不超 1m 限制”由构造自动满足。
    """
    if shape == "cylinder":
        D_base = ENVELOPE_SIDE - 2.0 * Hf
        return {
            "shape": shape,
            "char_len": _char_len(shape, D_base),     # 自然对流特征长度（直径）
            "base_side": D_base,
            "base_side_area": np.pi * D_base * HULL_LENGTH,   # 侧面积
            "base_end_area": 2.0 * np.pi * D_base**2 / 4.0,   # 两端封头面积
            "fin_banks": 1,                           # 圆柱只有 1 圈翅
            "perimeter": np.pi * D_base,              # 根部圆周
            "overlap_limit": np.pi * D_base,          # 翅根可布周长
            "v_cross": np.pi * D_base**2 / 4.0,       # 外轮廓横截面积
        }
    if shape == "cuboid":
        a = ENVELOPE_SIDE - 2.0 * Hf
        return {
            "shape": shape,
            "char_len": _char_len(shape, a),          # 等面积当量直径 2a/sqrt(pi)
            "base_side": a,
            "base_side_area": 4.0 * a * HULL_LENGTH,  # 四个侧面
            "base_end_area": 2.0 * a**2,              # 两个端面
            "fin_banks": 4,                           # 四个侧面各布翅
            "perimeter": 4.0 * a,                     # 横截面周长
            "overlap_limit": a,                       # 每面翅根可布宽度
            "v_cross": a**2,
        }
    return None


def _load_k_wall() -> float:
    """读取壳体材料（304 不锈钢）20 ℃ 导热系数，W/(m·K)。

    优先使用 C题数据/清洗后数据 CSV（与问题 1 同一数据源），
    数据缺失时回退到工程默认值 14.4 并打印提示。
    """
    try:
        root = _find_project_root()
        p = root / "C题数据" / "清洗后数据" \
            / "金属导热系数_EngineeringToolbox_clean.csv"
        if p.exists():
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("material") or "").strip() \
                            != "Steel - Stainless, Type 304":
                        continue
                    if abs(float(row["temperature_C"]) - 20.0) < 1e-9:
                        return float(row["thermal_conductivity_W_per_mK"])
    except Exception:
        pass
    print("[提示] 未找到 304 不锈钢导热系数 CSV，壳体 k_wall 使用默认值 %.2f W/(m·K)"
          % K_WALL_DEFAULT)
    return K_WALL_DEFAULT


_K_WALL_CACHE = {"value": None}


def _k_wall() -> float:
    """带缓存的壳体导热系数（GA/SLSQP 多次调用时只读一次 CSV）。"""
    if _K_WALL_CACHE["value"] is None:
        _K_WALL_CACHE["value"] = _load_k_wall()
    return _K_WALL_CACHE["value"]


def _inner_geometry(shape: str, geo: dict) -> dict:
    """由基体尺寸与壁厚计算内腔尺寸、内壁换热面积与内腔体积。"""
    inner_side = geo["base_side"] - 2.0 * WALL
    inner_len = HULL_LENGTH - 2.0 * WALL
    if shape == "cylinder":
        a_in = np.pi * inner_side * inner_len + 2.0 * np.pi * inner_side**2 / 4.0
        v_inner = np.pi * inner_side**2 / 4.0 * inner_len
    else:
        a_in = 4.0 * inner_side * inner_len + 2.0 * inner_side**2
        v_inner = inner_side**2 * inner_len
    return {"inner_side": inner_side, "inner_len": inner_len,
            "a_in": a_in, "v_inner": v_inner}


def _wall_resistance(shape: str, geo: dict, inner: dict, k_wall: float) -> float:
    """壳体壁导热热阻 K/W（与问题 1 口径一致：侧壁 + 两端封头）。"""
    if shape == "cylinder":
        d_o = geo["base_side"]
        d_i = inner["inner_side"]
        r_side = np.log(d_o / d_i) / (2.0 * np.pi * k_wall * HULL_LENGTH)
        r_caps = 2.0 * WALL / (k_wall * np.pi * d_i**2 / 4.0)
    else:
        a_o = geo["base_side_area"]
        a_i = 4.0 * inner["inner_side"] * inner["inner_len"]
        r_side = WALL / (k_wall * (a_o + a_i) / 2.0)
        r_caps = 2.0 * WALL / (k_wall * inner["inner_side"]**2)
    return r_side + r_caps


def _ra_number(D: float, dT: float, T_film: float, props_fn) -> float:
    """Churchill-Chu 使用的瑞利数 Ra = g*beta*dT*D^3/(nu*alpha)。"""
    p = props_fn(T_film)
    g = 9.81
    nu = p["mu"] / p["rho"]
    alpha = p["k"] / (p["rho"] * p["cp"])
    return g * p["beta"] * max(dT, 1e-9) * D**3 / (nu * alpha)


def _solve_thermal_network(shape: str, geo: dict, a_eff: float,
                           k_wall: float, t_air_max: float = T_MAX,
                           t_sea: float = T_INF,
                           max_iter: int = 60, tol: float = 1e-6) -> dict | None:
    """热阻网络求解：舱内空气 -> 壁导热 -> 海侧翅片，壁温自洽迭代。

    输入：
        shape/geo : 外形与基体几何
        a_eff     : 含翅片的有效外散热面积 m^2
        k_wall    : 壳体导热系数 W/(m·K)
    输出：
        dict，含 h_air/h_sea/h_total、t_wi/t_wo、r_total、q_total、
        ra_inner/ra_outer（适用域检查）、converged；
        内腔非法时返回 None。
    """
    inner = _inner_geometry(shape, geo)
    if inner["inner_side"] <= 0.0 or inner["inner_len"] <= 0.0:
        return None
    t_wi, t_wo = 40.0, 30.0
    d_in_char = _char_len(shape, inner["inner_side"])
    d_out_char = geo["char_len"]
    converged = False
    for _ in range(max_iter):
        h_air = h_horizontal_cylinder(
            d_in_char, max(t_air_max - t_wi, 1e-9),
            (t_air_max + t_wi) / 2.0, air_props)
        h_sea = h_horizontal_cylinder(
            d_out_char, max(t_wo - t_sea, 1e-9),
            (t_wo + t_sea) / 2.0, sea_props)
        r_air = 1.0 / (h_air * inner["a_in"])
        r_wall = _wall_resistance(shape, geo, inner, k_wall)
        r_sea = 1.0 / (h_sea * max(a_eff, 1e-9))
        r_total = r_air + r_wall + r_sea
        q_total = (t_air_max - t_sea) / r_total
        t_wi_new = t_air_max - q_total * r_air
        t_wo_new = t_sea + q_total * r_sea
        if abs(t_wi_new - t_wi) < tol and abs(t_wo_new - t_wo) < tol:
            t_wi, t_wo = t_wi_new, t_wo_new
            converged = True
            break
        t_wi, t_wo = t_wi_new, t_wo_new
    h_total = 1.0 / (r_total * max(a_eff, 1e-9))
    ra_inner = _ra_number(d_in_char, max(t_air_max - t_wi, 1e-9),
                          (t_air_max + t_wi) / 2.0, air_props)
    ra_outer = _ra_number(d_out_char, max(t_wo - t_sea, 1e-9),
                          (t_wo + t_sea) / 2.0, sea_props)
    return {"h_air": h_air, "h_sea": h_sea, "h_total": h_total,
            "t_wi": t_wi, "t_wo": t_wo, "r_total": r_total,
            "q_total": q_total, "v_inner": inner["v_inner"],
            "a_in": inner["a_in"], "inner_side": inner["inner_side"],
            "inner_len": inner["inner_len"],
            "ra_inner": ra_inner, "ra_outer": ra_outer,
            "converged": converged}


def _fin_efficiency(h_sea: float, Hf: float, df: float) -> float:
    """绝热翅尖直矩形翅效率：m=sqrt(2h/(k_fin*df))，eta=tanh(mHf)/(mHf)。"""
    m = np.sqrt(2.0 * h_sea / (K_FIN * df))
    mH = m * Hf
    return 1.0 if mH < 1e-12 else np.tanh(mH) / mH


def _solve_fin_and_network(shape: str, geo: dict, nf: int, Hf: float, df: float,
                           k_wall: float, max_outer: int = 20) -> dict | None:
    """eta_f 与热阻网络的固定点迭代（A_eff 依赖 eta_f，h_sea 依赖壁温）。"""
    a_fin_one = (2.0 * Hf + df) * HULL_LENGTH
    total_fins = nf * geo["fin_banks"]
    a_base = (geo["base_side_area"] + geo["base_end_area"]
              - total_fins * df * HULL_LENGTH)
    eta_f = 1.0
    net = None
    for _ in range(max_outer):
        a_eff = a_base + total_fins * eta_f * a_fin_one
        net = _solve_thermal_network(shape, geo, a_eff, k_wall)
        if net is None:
            return None
        eta_new = _fin_efficiency(net["h_sea"], Hf, df)
        if abs(eta_new - eta_f) < 1e-9:
            eta_f = eta_new
            break
        eta_f = eta_new
    return {"eta_f": eta_f, "a_fin_one": a_fin_one, "total_fins": total_fins,
            "a_base": a_base, "a_eff": a_base + total_fins * eta_f * a_fin_one,
            "net": net}


def _raw_metrics(shape: str, nf: int, Hf: float, df: float) -> dict | None:
    """只做几何/传热计算，不施加可行性门禁。

    该函数供 SLSQP 的数值梯度使用：若像 evaluate_design 那样在
    “Hf/df<3”等边界处直接返回 1e9 硬惩罚，中心差分会跨过不连续点，
    产生约 1e13 的假梯度；这里始终返回连续值，可行性交给约束函数
    g2/g3 与 L1 罚函数处理。
    """
    geo = _build_geometry(shape, Hf)
    if geo is None:
        return None
    inner = _inner_geometry(shape, geo)
    if inner["inner_side"] <= 0.0 or inner["inner_len"] <= 0.0:
        return None
    v_inner = inner["v_inner"]
    v_server = SERVER_W * SERVER_H * SERVER_L
    n_space = v_inner / v_server

    solved = _solve_fin_and_network(shape, geo, nf, Hf, df, _k_wall())
    if solved is None:
        return None
    net = solved["net"]
    # 统一口径：Q_max = h_total*A_eff*ΔT（与问题 1 compute_thermal_capacity 一致）
    q_max = net["h_total"] * solved["a_eff"] * (T_MAX - T_INF)
    n_theory = q_max / Q0

    return {
        "geo": geo,
        "base_side": geo["base_side"],
        "h": net["h_sea"],
        "h_air": net["h_air"],
        "h_total": net["h_total"],
        "t_wi": net["t_wi"],
        "t_wo": net["t_wo"],
        "r_total": net["r_total"],
        "q_max": q_max,
        "eta_f": solved["eta_f"],
        "a_fin_one": solved["a_fin_one"],
        "a_base": solved["a_base"],
        "a_eff": solved["a_eff"],
        "n_theory": n_theory,
        "n_space": n_space,
        "v_inner": v_inner,
        "ra_inner": net["ra_inner"],
        "ra_outer": net["ra_outer"],
        "converged": net["converged"],
    }


def evaluate_design(shape: str, nf: int, Hf: float, df: float,
                    collect_warnings: bool = True) -> dict:
    """计算一组翅片参数下的有效散热面积 A_eff 与装机台数 N。

    输入：
        shape : "cylinder" 或 "cuboid"
        nf    : 翅片数（长方体表示每个侧面的翅片数），须在 [NF_MIN, NF_MAX]
        Hf    : 翅高，m
        df    : 翅厚，m
    输出：
        dict，含 feasible(是否可行)、h(海侧 h_sea)、h_air、h_total、
        t_wi/t_wo、A_eff、eta_f、N_theory、N_space、N(整数)、warnings 等。

    鲁棒性约定：任何不满足几何/空间/温度约束的输入都不抛异常，
    而是返回 feasible=False，并给出中文警告。
    """
    warns = []

    # ---- 输入合理性检查 ----
    if shape not in ("cylinder", "cuboid"):
        warns.append("未知外形 shape=%r，仅支持 cylinder/cuboid" % (shape,))
        return _infeasible_result(warns)
    nf = int(round(float(nf)))
    if nf < 1:
        warns.append("翅片数 nf=%d < 1，无法形成有效翅片" % nf)
        return _infeasible_result(warns)
    if nf > NF_MAX:
        warns.append("翅片数 nf=%d 超出声明区间 [%d, %d]" % (nf, NF_MIN, NF_MAX))
        return _infeasible_result(warns)
    if not (HF_MIN - 1e-12 <= Hf <= HF_MAX + 1e-12):
        warns.append("翅高 Hf=%.5f 超出区间 [%.4f, %.4f] m" % (Hf, HF_MIN, HF_MAX))
        return _infeasible_result(warns)
    if not (DF_MIN - 1e-12 <= df <= DF_MAX + 1e-12):
        warns.append("翅厚 df=%.5f 超出区间 [%.4f, %.4f] m" % (df, DF_MIN, DF_MAX))
        return _infeasible_result(warns)
    if Hf / df < 3.0:
        warns.append("Hf/df=%.2f < 3，一维直翅假设不再可靠，按不可行处理" % (Hf / df))
        return _infeasible_result(warns)

    geo = _build_geometry(shape, Hf)
    if geo is None:
        warns.append("几何建模失败")
        return _infeasible_result(warns)

    # ---- 翅根间距检查：翅根厚度之和不能超过根部周长 ----
    overlap_limit = geo["overlap_limit"]
    if nf * df > overlap_limit + 1e-9:
        warns.append("nf*df=%.4f > 可布置周长 %.4f m，翅根重叠" % (nf * df, overlap_limit))
        return _infeasible_result(warns)

    # ---- 内部空间检查 ----
    base_side = geo["base_side"]
    inner = _inner_geometry(shape, geo)
    if inner["inner_side"] <= 0.0 or inner["inner_len"] <= 0.0:
        warns.append("壁厚使内部尺寸 <= 0，无可用容积")
        return _infeasible_result(warns)
    v_inner = inner["v_inner"]
    v_server = SERVER_W * SERVER_H * SERVER_L
    # 体积法空间上限（仅作上界，非真实机架布局数；散热上限通常会先约束 N）
    n_space = v_inner / v_server

    # ---- 热阻网络：eta_f 与壁温自洽迭代 ----
    solved = _solve_fin_and_network(shape, geo, nf, Hf, df, _k_wall())
    if solved is None:
        warns.append("热阻网络求解失败（内腔尺寸非法）")
        return _infeasible_result(warns)
    net = solved["net"]
    h = net["h_sea"]
    eta_f = solved["eta_f"]
    a_fin_one = solved["a_fin_one"]
    a_base = solved["a_base"]
    a_eff = solved["a_eff"]
    # 统一口径：Q_max = h_total*A_eff*ΔT（与问题 1 一致；网络 q_total 仅作迭代量）
    q_max = net["h_total"] * a_eff * (T_MAX - T_INF)
    if not net["converged"]:
        warns.append("壁温自洽迭代未完全收敛（已用末次结果）")
    if not (1e-5 <= net["ra_outer"] <= 1e12):
        warns.append("海侧 Ra=%.3e 超出 Churchill-Chu 适用域 [1e-5, 1e12]"
                     % net["ra_outer"])
    if not (1e-5 <= net["ra_inner"] <= 1e12):
        warns.append("舱内 Ra=%.3e 超出 Churchill-Chu 适用域 [1e-5, 1e12]"
                     % net["ra_inner"])

    # ---- 散热理论上限与空间上限 ----
    n_theory = q_max / Q0
    if n_theory < 1.0:
        warns.append("N_theory=%.2f < 1，散热能力不足以放置任何服务器" % n_theory)
        return _infeasible_result(warns)

    # ---- 最终容量：两个上限取小后向下取整 ----
    n_final = int(np.floor(min(n_theory, n_space)))
    if n_final < 1:
        warns.append("取整后 N=%d < 1，无可行装机方案" % n_final)
        return _infeasible_result(warns)

    return {
        "feasible": True,
        "shape": shape,
        "nf": nf,
        "Hf": float(Hf),
        "df": float(df),
        "base_side": float(base_side),
        "h": float(h),
        "h_air": float(net["h_air"]),
        "h_total": float(net["h_total"]),
        "t_wi": float(net["t_wi"]),
        "t_wo": float(net["t_wo"]),
        "r_total": float(net["r_total"]),
        "eta_f": float(eta_f),
        "a_fin_one": float(a_fin_one),
        "a_base": float(a_base),
        "a_eff": float(a_eff),
        "q_max": float(q_max),
        "n_theory": float(n_theory),
        "v_inner": float(v_inner),
        "n_space": float(n_space),
        "ra_inner": float(net["ra_inner"]),
        "ra_outer": float(net["ra_outer"]),
        "converged": bool(net["converged"]),
        "N": n_final,
        "warnings": warns if collect_warnings else [],
    }


def _infeasible_result(warns) -> dict:
    """不可行结果模板：保证调用方拿到统一结构，不会因缺键崩溃。"""
    return {
        "feasible": False,
        "shape": None, "nf": None, "Hf": None, "df": None,
        "base_side": None, "h": None, "eta_f": None, "a_fin_one": None,
        "h_air": None, "h_total": None, "t_wi": None, "t_wo": None,
        "r_total": None,
        "a_base": None, "a_eff": None, "q_max": None, "n_theory": None,
        "v_inner": None, "n_space": None, "N": 0,
        "ra_inner": None, "ra_outer": None, "converged": False,
        "warnings": warns,
    }


def solve_q2(shape: str, nf: int, Hf: float, df: float) -> dict:
    """问题 2 核心算法总入口：一组翅片参数 -> 全部中间量与最终容量。"""
    return evaluate_design(shape, nf, Hf, df)


# ==================================================================
# 自检与命令行运行
# ==================================================================
def _self_check() -> bool:
    """自检：问题 1 基准回归（裸圆柱 D=1、无翅片应复现 N=15、Q≈7761.5 W）。"""
    geo = _build_geometry("cylinder", 0.0)
    a_out = geo["base_side_area"] + geo["base_end_area"]
    net = _solve_thermal_network("cylinder", geo, a_out, _k_wall())
    q1_max = net["h_total"] * a_out * (T_MAX - T_INF)
    n_theory = q1_max / Q0
    n_space = net["v_inner"] / (SERVER_W * SERVER_H * SERVER_L)
    n = int(np.floor(min(n_theory, n_space)))
    ok = abs(q1_max - 7761.5) / 7761.5 < 0.05 and n == 15
    ok = ok and abs(q1_max - net["q_total"]) / q1_max < 1e-6
    print("自检（问题1基准回归）：Q_max=%.1f W，h_sea=%.1f W/(m^2·K)，"
          "N=%d 台 -> %s" % (q1_max, net["h_sea"], n, "PASS" if ok else "FAIL"))
    res = solve_q2("cylinder", 60, 0.10, 0.005)
    print("示例：圆柱 nf=60，Hf=0.10 m，df=0.005 m -> A_eff=%.3f m^2，"
          "N_theory=%.2f 台，N=%d 台" % (res["a_eff"], res["n_theory"], res["N"]))
    return ok


def main() -> None:
    """先跑问题 1 基准回归，再给一个翅片设计示例。"""
    print("=" * 56)
    print("问题 2 核心算法（独立函数版）")
    print("=" * 56)
    _self_check()


if __name__ == "__main__":
    main()
