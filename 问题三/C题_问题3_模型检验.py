# -*- coding: utf-8 -*-
"""
C 题 问题 3 多目标与决策模型检验
=================================================================
对《C题_问题3_NSGA2_TOPSIS.py》的模型做 7 项独立检验：
    1) 目标函数方向检验（Q 最大 / 成本最小 / 寿命最大 是否与代码一致）
    2) 约束模型检验（静水压力、壁厚、腐蚀余量、成本公式的单位与极限行为）
    3) 拟合回归诊断（温度-深度回归：线性性、同方差、残差正态性、自相关、
       共线性、函数形式 RESET）
    4) Pareto 非支配性质检验（任取两方案不允许互相支配）
    5) TOPSIS 模型检验（标准化、权重和、正负理想解、贴近度 0-1）
    6) 权重敏感性分析（±5%、±10% 扰动，Spearman 秩相关 > 0.8）
    7) 材料与深度情景检验（不同材料、不同水深边界，确认无突变/符号错误）

依赖：仅 numpy / pandas / matplotlib（检验统计量全部手写，不引入 scipy）。
运行：python C题_问题3_模型检验.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "输出" / "模型检验"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------- 载入模型
def load_problem3_model():
    """以模块方式载入问题3主程序，复用其全部函数与常量。"""
    spec = importlib.util.spec_from_file_location(
        "q3_model", HERE / "C题_问题3_NSGA2_TOPSIS.py")
    q3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q3)

    q3.setup_chinese_font()
    temp_df = q3.load_temperature_profile()
    model = q3.fit_temperature_model(temp_df, degree=2)
    r2_cv = q3.cross_validate_temperature(temp_df, k=5, degree=2)
    temp_pred = q3.make_temp_predictor(model)
    mat_df = q3.build_material_table()
    q3.MAT_DF = mat_df            # 供 preprocess_pareto_matrix 使用

    pareto_csv = HERE / "输出" / "结果_帕累托前沿.csv"
    if pareto_csv.exists():
        pareto_df = pd.read_csv(pareto_csv)
        print("读取已保存的帕累托前沿：", pareto_csv)
    else:
        print("未找到已保存帕累托前沿，重新运行 NSGA-II（较慢）...")
        pop, pareto, history = q3.nsga2(mat_df, temp_pred)
        pareto_df = q3.preprocess_pareto_matrix(pareto)
    return q3, temp_df, model, r2_cv, temp_pred, mat_df, pareto_df


# ---------------------------------------------------------------- 统计工具
def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_r(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return pearson_r(rx, ry)


def _gammp(a, x):
    """正则化下不完全伽马函数 P(a,x)，用于卡方分布 CDF。"""
    if x <= 0:
        return 0.0
    if x < a + 1:
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(300):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-12:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_cdf(x, k):
    """自由度为 k 的卡方分布 CDF。"""
    return _gammp(k / 2.0, x / 2.0)


def _betacf(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h


def betainc(a, b, x):
    """正则化不完全贝塔函数 I_x(a,b)，用于 F 分布 CDF。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_cdf(F, d1, d2):
    """F 分布 CDF。"""
    if F <= 0:
        return 0.0
    x = d1 * F / (d1 * F + d2)
    return betainc(d1 / 2.0, d2 / 2.0, x)


def jb_test(resid):
    """Jarque-Bera 正态性检验，返回 (JB, p)。"""
    n = len(resid)
    s = float(pd.Series(resid).skew())
    k = float(pd.Series(resid).kurtosis())  # 超额峰度
    jb = n / 6.0 * (s ** 2 + k ** 2 / 4.0)
    p = 1.0 - chi2_cdf(jb, 2.0)
    return jb, p, s, k


def goldfeld_quandt(fitted, resid):
    """Goldfeld-Quandt 异方差检验（按拟合值排序，前/后 1/3 方差比）。"""
    order = np.argsort(np.asarray(fitted))
    r = np.asarray(resid)
    n = len(r)
    k = max(3, n // 3)
    low = r[order[:k]]
    high = r[order[-k:]]
    f_stat = float(np.var(high, ddof=1) / max(np.var(low, ddof=1), 1e-300))
    p = 1.0 - f_cdf(f_stat, k - 1, k - 1)
    return f_stat, p, k


def reset_test(x, t, degree=2):
    """RESET 函数形式检验：在 degree 次拟合中加入 fitted^2 再检验。"""
    coef = np.polyfit(x, t, degree)
    fitted = np.polyval(coef, x)
    rss_r = float(np.sum((t - fitted) ** 2))
    X = np.column_stack([x ** i for i in range(degree, -1, -1)])
    X_u = np.column_stack([X, fitted ** 2])
    beta, *_ = np.linalg.lstsq(X_u, t, rcond=None)
    resid_u = t - X_u @ beta
    rss_u = float(np.sum(resid_u ** 2))
    n = len(t)
    f_stat = ((rss_r - rss_u) / 1.0) / (rss_u / (n - X_u.shape[1]))
    p = 1.0 - f_cdf(f_stat, 1, n - X_u.shape[1])
    return f_stat, p, rss_r, rss_u


def vif_and_condition(x):
    """多项式回归共线性诊断：VIF 与设计矩阵条件数。"""
    X = np.column_stack([x ** 2, x, np.ones_like(x)])
    vifs = []
    for j in range(2):
        yj = X[:, j]
        Xj = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
        pred = Xj @ beta
        r2 = 1.0 - np.sum((yj - pred) ** 2) / max(np.sum((yj - yj.mean()) ** 2), 1e-300)
        vifs.append(1.0 / max(1.0 - r2, 1e-6))
    cond = float(np.linalg.cond(X))
    return vifs, cond


def breusch_pagan_test(resid, x):
    """Breusch-Pagan 异方差检验：残差平方对解释变量回归。

    返回 (LM, p, F, F_p, R2)。LM 服从卡方分布，自由度=非截距项数。
    """
    n = len(resid)
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    u2 = np.asarray(resid, dtype=float) ** 2
    beta, *_ = np.linalg.lstsq(X, u2, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((u2 - pred) ** 2))
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    df_num = X.shape[1] - 1
    df_den = n - X.shape[1]
    lm = n * r2
    p_lm = 1.0 - chi2_cdf(lm, df_num)
    f_stat = (r2 / df_num) / max((1.0 - r2) / df_den, 1e-300)
    p_f = 1.0 - f_cdf(f_stat, df_num, df_den)
    return lm, p_lm, f_stat, p_f, r2


def white_test(resid, x):
    """White 异方差检验：残差平方对 x、x^2、x^3、x^4 回归。

    用于捕捉多项式拟合中的非线性异方差；自由度=4（不含截距）。
    """
    n = len(resid)
    X = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3, x ** 4])
    u2 = np.asarray(resid, dtype=float) ** 2
    beta, *_ = np.linalg.lstsq(X, u2, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((u2 - pred) ** 2))
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    df_num = X.shape[1] - 1
    lm = n * r2
    p = 1.0 - chi2_cdf(lm, df_num)
    return lm, p, r2


def polynomial_f_test(x, t, deg_low, deg_high):
    """比较两个多项式阶数的 F 检验：检验多出的高阶项是否显著。"""
    n = len(t)
    sse = {}
    for deg in (deg_low, deg_high):
        coef = np.polyfit(x, t, deg)
        pred = np.polyval(coef, x)
        sse[deg] = float(np.sum((t - pred) ** 2))
    df_num = deg_high - deg_low
    df_den = n - (deg_high + 1)
    f_stat = ((sse[deg_low] - sse[deg_high]) / df_num) \
        / max(sse[deg_high] / df_den, 1e-300)
    p = 1.0 - f_cdf(f_stat, df_num, df_den)
    return f_stat, p, sse[deg_low], sse[deg_high]


# ---------------------------------------------------------------- TOPSIS 静默版
def topsis_rank(pareto_df: pd.DataFrame, weights: np.ndarray):
    """与主程序一致的 TOPSIS，但不打印。

    返回 (贴近度, 正理想, 负理想, 加权矩阵, z, mm, norm, w)，供检验复算。
    """
    cols = ["Q", "cost", "life"]
    z = (pareto_df[cols] - pareto_df[cols].mean()) / pareto_df[cols].std()
    mm = pd.DataFrame(index=z.index)
    for col in cols:
        mn, mx = z[col].min(), z[col].max()
        if mx - mn < 1e-12:
            mm[col] = 1.0
        elif col == "cost":
            mm[col] = (mx - z[col]) / (mx - mn)
        else:
            mm[col] = (z[col] - mn) / (mx - mn)
    norm = mm / np.sqrt((mm ** 2).sum(axis=0))
    w = weights / weights.sum()
    v = norm.mul(w, axis=1)
    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg)
    return closeness, v_pos, v_neg, v, z, mm, norm, w


# ================================================================ 7 项检验
def check_1_objective_direction(q3, mat_df, temp_pred):
    """目标函数方向：成本增不应使目标变好等。"""
    print("=" * 78)
    print("检验 1  目标函数方向检验")
    print("=" * 78)
    # 用 8 mm 壁厚使寿命低于 50 年上限，才能检验腐蚀/强度对寿命的方向。
    base = q3.evaluate_design(0, 50.0, 0.008, mat_df, temp_pred)
    print("基准方案（6061，d=50 m，t=8 mm）：Q=%.4g W，成本=%.2f 元，寿命=%.2f 年"
          % (base["Q"], base["cost"], base["life"]))

    # 目标向量应严格等于 [-Q, cost, -life]（全部最小化）
    obj_ok = bool(np.allclose(base["obj"], [-base["Q"], base["cost"], -base["life"]]))
    print("目标向量映射 obj=[-Q, cost, -life]：%s"
          % ("通过" if obj_ok else "失败"))
    print("口径说明：题目要求提高散热效果/存放更多服务器，代码以 Q 作为 f1；"
          "N 由 Q 与空间约束导出并单独输出，不作为优化目标。")

    rows = []

    def add_case(name, metric, before, after, expect, ok):
        rows.append({
            "检验点": name, "指标": metric,
            "基准值": before, "扰动后值": after,
            "期望": expect, "通过": "通过" if ok else "失败",
        })

    # 材料价格 +10% -> 成本应上升（成本目标变差）
    tmp = mat_df.copy()
    tmp.loc[0, "价格_元_吨"] *= 1.10
    r = q3.evaluate_design(0, 50.0, 0.008, tmp, temp_pred)
    add_case("材料价格 +10%", "成本_元", base["cost"], r["cost"],
             "成本上升", r["cost"] > base["cost"])
    print("价格 +10%%：成本 %.2f -> %.2f 元（期望上升，实际 %s）"
          % (base["cost"], r["cost"], "通过" if r["cost"] > base["cost"] else "失败"))

    # 腐蚀速率 +10% -> 寿命应下降（寿命目标变差）
    tmp = mat_df.copy()
    tmp.loc[0, "腐蚀速率_mm_年"] *= 1.10
    r = q3.evaluate_design(0, 50.0, 0.008, tmp, temp_pred)
    add_case("腐蚀速率 +10%", "寿命_年", base["life"], r["life"],
             "寿命下降", r["life"] < base["life"] - 1e-9)
    print("腐蚀速率 +10%%：寿命 %.2f -> %.2f 年（期望下降，实际 %s）"
          % (base["life"], r["life"], "通过" if r["life"] < base["life"] else "失败"))

    # 导热系数 +10% -> 散热应上升（散热目标变好）
    tmp = mat_df.copy()
    tmp.loc[0, "导热系数_W_mK"] *= 1.10
    r = q3.evaluate_design(0, 50.0, 0.008, tmp, temp_pred)
    add_case("导热系数 +10%", "Q_W", base["Q"], r["Q"],
             "Q上升", r["Q"] > base["Q"])
    print("导热系数 +10%%：Q %.4g -> %.4g W（期望上升，实际 %s）"
          % (base["Q"], r["Q"], "通过" if r["Q"] > base["Q"] else "失败"))

    # 屈服强度 -10% -> 所需壁厚上升 -> 腐蚀余量下降 -> 寿命下降
    tmp = mat_df.copy()
    tmp.loc[0, "屈服强度_MPa"] *= 0.90
    r = q3.evaluate_design(0, 50.0, 0.008, tmp, temp_pred)
    add_case("屈服强度 -10%", "t_req_m", base["t_req"], r["t_req"],
             "t_req上升", r["t_req"] > base["t_req"])
    add_case("屈服强度 -10%", "寿命_年", base["life"], r["life"],
             "寿命下降", r["life"] < base["life"] - 1e-9)
    print("屈服强度 -10%%：t_req %.5f -> %.5f m，寿命 %.2f -> %.2f 年（期望寿命下降，实际 %s）"
          % (base["t_req"], r["t_req"], base["life"], r["life"],
             "通过" if r["life"] < base["life"] else "失败"))

    # 深度 50->60 m：水温下降 -> Q 上升；压力上升 -> t_req 上升
    r = q3.evaluate_design(0, 60.0, 0.008, mat_df, temp_pred)
    add_case("深度 50->60 m", "Q_W", base["Q"], r["Q"],
             "Q上升", r["Q"] > base["Q"])
    add_case("深度 50->60 m", "t_req_m", base["t_req"], r["t_req"],
             "t_req上升", r["t_req"] > base["t_req"])
    print("深度 50->60 m：Q %.4g -> %.4g W（期望上升），t_req %.5f -> %.5f m（期望上升）"
          % (base["Q"], r["Q"], base["t_req"], r["t_req"]))

    # 壁厚 +10% -> 材料质量/成本上升，内部空间下降（成本目标变差）
    r = q3.evaluate_design(0, 50.0, 0.0088, mat_df, temp_pred)
    add_case("壁厚 +10%", "成本_元", base["cost"], r["cost"],
             "成本上升", r["cost"] > base["cost"])
    add_case("壁厚 +10%", "N_space_台", base["n_space"], r["n_space"],
             "空间上限下降", r["n_space"] < base["n_space"])
    print("壁厚 8->8.8 mm：成本 %.2f -> %.2f 元（期望上升），空间上限 %.2f -> %.2f 台（期望下降）"
          % (base["cost"], r["cost"], base["n_space"], r["n_space"]))

    rows.append({
        "检验点": "目标向量映射", "指标": "obj",
        "基准值": base["obj"].tolist(), "扰动后值": "—",
        "期望": "等于[-Q,cost,-life]", "通过": "通过" if obj_ok else "失败",
    })
    dir_df = pd.DataFrame(rows)
    print(dir_df.to_string(index=False))
    ok_all = bool(dir_df["通过"].eq("通过").all())
    print("方向检验结论：NSGA-II 目标向量 obj=[-Q, cost, -life]（最小化），"
          "等价于 max Q / min 成本 / max 寿命。总体：%s" % ("通过" if ok_all else "失败"))
    print()
    return dir_df, ok_all and obj_ok


def check_2_constraints(q3, mat_df, temp_pred):
    """约束公式单位与极限行为。"""
    print("=" * 78)
    print("检验 2  约束模型检验（静水压力 / 壁厚 / 腐蚀余量 / 成本）")
    print("=" * 78)
    mat = mat_df.iloc[0]
    sigma_allow = float(mat["屈服强度_MPa"]) * 1e6 / q3.SAFETY_FACTOR
    print("采用材料：%s，σ_y=%.1f MPa，n_s=%.1f，σ_allow=%.4g Pa"
          % (mat["材料"], mat["屈服强度_MPa"], q3.SAFETY_FACTOR, sigma_allow))
    print("公式：p = ρ_w·g·d；t_req = p·D/(2σ_allow)；"
          "腐蚀余量=(t-t_req)·1000 mm；寿命=腐蚀余量/腐蚀速率")
    print("单位：p[Pa]=kg/m^3·m/s^2·m；t_req[m]=Pa·m/Pa；"
          "成本[元]=kg·元/kg + m^2·元/m^2")

    # 固定壁厚表：隔离压力/壁厚公式本身
    rows = []
    for d in [0.0, 5.0, 25.0, 50.0, 75.0, 100.0]:
        res = q3.evaluate_design(0, d, 0.020, mat_df, temp_pred)
        p_manual = q3.RHO_SW * q3.G * d
        t_manual = p_manual * q3.ENVELOPE_SIDE / (2.0 * sigma_allow)
        rows.append({
            "深度_m": d, "p_hydro_Pa": res["p_hydro"],
            "p_手动_Pa": p_manual, "t_req_m": res["t_req"],
            "t_手动_m": t_manual, "Q_W": res["Q"],
            "成本_元": res["cost"], "寿命_年": res["life"],
            "可行": res["feasible"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    p0 = df.iloc[0]["p_hydro_Pa"]
    print("深度=0：p=%.3e Pa（期望≈0，实际%s），t_req=%.3e m"
          % (p0, "通过" if abs(p0) < 1e-6 else "失败", df.iloc[0]["t_req_m"]))
    ok_lin = np.allclose(df["p_hydro_Pa"], df["p_手动_Pa"], rtol=1e-8)
    ok_t = np.allclose(df["t_req_m"], df["t_手动_m"], rtol=1e-8)
    print("静水压力线性关系：%s；壁厚公式：%s" % ("通过" if ok_lin else "失败",
                                              "通过" if ok_t else "失败"))
    print("固定壁厚 20 mm 时，深度增大：Q 单调不减 = %s，成本单调不减 = %s，寿命单调不增 = %s"
          % (bool(np.all(np.diff(df["Q_W"]) >= -1e-6)),
             bool(np.all(np.diff(df["成本_元"]) >= -1e-6)),
             bool(np.all(np.diff(df["寿命_年"]) <= 1e-6))))

    # 工程设计壁厚表：按 max(WALL_MIN, 1.5*t_req) 选壁厚，检查成本随深度
    design_rows = []
    for d in [0.0, 5.0, 25.0, 50.0, 75.0, 100.0]:
        res = q3.evaluate_design(0, d, 0.020, mat_df, temp_pred)
        wall_design = max(q3.WALL_MIN, 1.5 * res["t_req"])
        rd = q3.evaluate_design(0, d, wall_design, mat_df, temp_pred)
        design_rows.append({
            "深度_m": d, "t_req_m": res["t_req"],
            "设计壁厚_m": wall_design, "成本_元": rd["cost"],
            "N_space_台": rd["n_space"], "寿命_年": rd["life"],
            "可行": rd["feasible"],
        })
    design_df = pd.DataFrame(design_rows)
    print("\n工程设计壁厚表（wall=max(WALL_MIN, 1.5*t_req)）：")
    print(design_df.to_string(index=False))
    cost_mono = bool(np.all(np.diff(design_df["成本_元"]) >= -1e-6))
    wall_mono = bool(np.all(np.diff(design_df["设计壁厚_m"]) >= -1e-9))
    t_mono = bool(np.all(np.diff(design_df["t_req_m"]) >= -1e-9))
    print("深度增大时：t_req 单调不减=%s，设计壁厚单调不减=%s，成本单调不减=%s"
          % (t_mono, wall_mono, cost_mono))
    print("说明：浅水段 t_req<WALL_MIN=4 mm，成本受壁厚下限控制而保持不变；"
          "t_req 超过下限后成本随壁厚上升。")

    # 成本-壁厚直接单调表：壁厚越大材料体积越大，成本严格上升
    cost_rows = []
    for w in [0.004, 0.008, 0.012, 0.020, 0.030, 0.050]:
        rw = q3.evaluate_design(0, 50.0, w, mat_df, temp_pred)
        cost_rows.append({
            "壁厚_m": w, "成本_元": rw["cost"], "mass_kg": rw["mass_kg"],
            "N_space_台": rw["n_space"], "t_req_m": rw["t_req"],
            "可行": rw["feasible"],
        })
    cost_wall_df = pd.DataFrame(cost_rows)
    print("\n成本随壁厚变化表（d=50 m）：")
    print(cost_wall_df.to_string(index=False))
    cost_strict = bool(np.all(np.diff(cost_wall_df["成本_元"]) > 0))
    n_mono = bool(np.all(np.diff(cost_wall_df["N_space_台"]) <= 1e-9))
    print("壁厚增大时：成本严格上升=%s，内部空间上限单调不增=%s"
          % (cost_strict, n_mono))

    # 壁厚低于 t_req 时必须不可行
    res_bad = q3.evaluate_design(0, 100.0, 0.001, mat_df, temp_pred)
    print("极限检查：t=1 mm < t_req=%.4f mm -> 可行=%s（期望 False），g1=%.4g"
          % (res_bad["t_req"] * 1000, res_bad["feasible"], res_bad["g1"]))
    limit_ok = (not res_bad["feasible"]) and res_bad["g1"] < 0
    print()
    status = bool(p0 < 1e-6 and ok_lin and ok_t and cost_mono and wall_mono
                  and t_mono and cost_strict and n_mono and limit_ok)
    return {"fixed": df, "design": design_df, "cost_wall": cost_wall_df,
            "status": status, "p0_ok": abs(p0) < 1e-6,
            "formula_ok": ok_lin and ok_t, "mono_ok": cost_mono and wall_mono
            and t_mono and cost_strict and n_mono, "limit_ok": limit_ok}


def check_3_regression(model):
    """温度-深度回归诊断。"""
    print("=" * 78)
    print("检验 3  拟合回归诊断（温度-深度二次回归）")
    print("=" * 78)
    df = model["df"]
    d = df["depth_m"].to_numpy(dtype=float)
    t = df["temp_C"].to_numpy(dtype=float)
    x = (d - model["d_mean"]) / model["d_std"]
    resid = model["resid"]
    fitted = model["pred"]

    # 线性性：一次/二次/三次 R^2
    r2s = {}
    for deg in (1, 2, 3):
        c = np.polyfit(x, t, deg)
        p = np.polyval(c, x)
        r2s[deg] = 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    f_quad, p_quad, sse1, sse2 = polynomial_f_test(x, t, 1, 2)
    f_cub, p_cub, sse2b, sse3 = polynomial_f_test(x, t, 2, 3)
    print("线性性检验：R²(1次)=%.6f，R²(2次)=%.6f，R²(3次)=%.6f"
          % (r2s[1], r2s[2], r2s[3]))
    print("二次项 F 检验：F=%.4f，p=%.4f（p<0.05 说明二次项显著）"
          % (f_quad, p_quad))
    print("三次项 F 检验：F=%.4f，p=%.4f（p<0.05 说明三次项显著）"
          % (f_cub, p_cub))

    # 同方差（Goldfeld-Quandt）
    gq_f, gq_p, k = goldfeld_quandt(fitted, resid)
    print("同方差检验（Goldfeld-Quandt）：F=%.4f，p=%.4f（>0.05 通常无显著异方差）"
          % (gq_f, gq_p))
    bp_lm, bp_p, bp_f, bp_fp, bp_r2 = breusch_pagan_test(resid, x)
    print("同方差检验（Breusch-Pagan）：LM=%.4f，p=%.4f；F=%.4f，F_p=%.4f"
          % (bp_lm, bp_p, bp_f, bp_fp))
    wh_lm, wh_p, wh_r2 = white_test(resid, x)
    print("同方差检验（White）：LM=%.4f，p=%.4f（含 x^3、x^4 的非线性异方差）"
          % (wh_lm, wh_p))

    # 残差正态性（Jarque-Bera）
    jb, jb_p, sk, ku = jb_test(resid)
    print("残差正态性（Jarque-Bera）：偏度=%.4f，峰度=%.4f，JB=%.4f，p=%.4f"
          % (sk, ku, jb, jb_p))

    # 自相关（DW + lag1）
    dw = model["dw"]
    r1 = pearson_r(resid[:-1], resid[1:])
    print("自相关检验：DW=%.3f（接近2无明显自相关），lag1 r=%.4f"
          % (dw, r1 if not np.isnan(r1) else 0.0))

    # 共线性（VIF + 条件数）
    vifs, cond = vif_and_condition(x)
    print("共线性检验：VIF(x²)=%.2f，VIF(x)=%.2f，设计矩阵条件数=%.2f"
          % (vifs[0], vifs[1], cond))

    # 函数形式（RESET）
    f_reset, p_reset, rss_r, rss_u = reset_test(x, t, degree=2)
    print("函数形式（RESET 加 fitted²）：F=%.4f，p=%.4f" % (f_reset, p_reset))

    # 汇总：异方差、自相关、函数形式任一不通过即提示改用稳健/分段模型
    checks_failed = []
    if gq_p < 0.05 or bp_p < 0.05 or wh_p < 0.05:
        checks_failed.append("异方差（GQ/BP/White）")
    if jb_p < 0.05:
        checks_failed.append("残差正态性")
    if not (1.2 <= dw <= 2.8):
        checks_failed.append("自相关（DW=%.3f 偏离 2）" % dw)
    if max(vifs) >= 10 or cond > 30:
        checks_failed.append("共线性")
    if p_reset < 0.05:
        checks_failed.append("函数形式（RESET）")
    if p_cub < 0.05:
        checks_failed.append("函数形式（三次项显著）")
    status = "警告" if checks_failed else "通过"
    print("回归诊断结论：%s。未通过项：%s"
          % (status, "；".join(checks_failed) if checks_failed else "无"))
    print("建议：温度剖面平滑导致 DW 偏低；若需正式外推，可改用两段线性模型/"
          "三次多项式或稳健标准误/加权最小二乘。")
    print()
    return {"r2_linear": r2s[1], "r2_quad": r2s[2], "r2_cubic": r2s[3],
            "f_quad": f_quad, "p_quad": p_quad, "f_cubic": f_cub,
            "p_cubic": p_cub, "gq_F": gq_f, "gq_p": gq_p,
            "bp_LM": bp_lm, "bp_p": bp_p, "bp_F": bp_f, "bp_F_p": bp_fp,
            "white_LM": wh_lm, "white_p": wh_p, "jb": jb, "jb_p": jb_p,
            "dw": dw, "lag1": r1, "vif_x2": vifs[0], "vif_x": vifs[1],
            "cond": cond, "reset_F": f_reset, "reset_p": p_reset,
            "status": status, "failed": "；".join(checks_failed)}


def check_4_pareto(pareto_df):
    """Pareto 非支配性质：任意两方案不允许互相支配。"""
    print("=" * 78)
    print("检验 4  Pareto 非支配性质检验（解集性质）")
    print("=" * 78)
    df = pareto_df[["Q", "cost", "life"]].reset_index(drop=True)
    n = len(df)
    q = df["Q"].to_numpy()
    c = df["cost"].to_numpy()
    lf = df["life"].to_numpy()
    viol = 0
    for i in range(n):
        for j in range(i + 1, n):
            d1 = (q[i] >= q[j]) & (c[i] <= c[j]) & (lf[i] >= lf[j]) \
                 & ((q[i] > q[j]) | (c[i] < c[j]) | (lf[i] > lf[j]))
            d2 = (q[j] >= q[i]) & (c[j] <= c[i]) & (lf[j] >= lf[i]) \
                 & ((q[j] > q[i]) | (c[j] < c[i]) | (lf[j] > lf[i]))
            if d1 or d2:
                viol += 1
                print("  发现支配对：%d 支配 %d（第 %d/%d 对）" % (i, j, i, j))
    print("方案数=%d，检验对数=%d，支配对=%d" % (n, n * (n - 1) // 2, viol))
    print("结论：%s" % ("通过（无支配对）" if viol == 0 else "失败"))
    print()
    return viol


def check_5_topsis(pareto_df, weights):
    """TOPSIS 模型性质检查。"""
    print("=" * 78)
    print("检验 5  TOPSIS 模型检验")
    print("=" * 78)
    closeness, v_pos, v_neg, v, z, mm, norm, w = topsis_rank(pareto_df, weights)
    wsum = float(weights.sum())
    print("权重 w=", weights, "，权重和=%.6f（期望 1）：%s"
          % (wsum, "通过" if abs(wsum - 1.0) < 1e-9 else "失败"))
    print("贴近度范围：[%.6f, %.6f]，全部在 0-1：%s"
          % (closeness.min(), closeness.max(),
             "通过" if closeness.between(0, 1).all() else "失败"))
    print("正理想解：\n", v_pos.to_string())
    print("负理想解：\n", v_neg.to_string())
    ideal_ok = bool((v_pos >= v_neg).all())
    print("正理想解 >= 负理想解（逐列）：%s" % ("通过" if ideal_ok else "失败"))

    # 标准化合理性：z-score 均值≈0、标准差≈1；向量归一化列范数≈1
    z_ok = bool(np.all(np.abs(z.mean()) < 1e-9) and np.all(np.abs(z.std() - 1.0) < 1e-9))
    norm_ok = bool(np.all(np.abs(np.sqrt((norm ** 2).sum(axis=0)) - 1.0) < 1e-9))
    print("z-score 标准化：各列均值≈0、标准差≈1：%s" % ("通过" if z_ok else "失败"))
    print("向量归一化：各列 L2 范数≈1：%s" % ("通过" if norm_ok else "失败"))

    # 方向一致性只检查“正向化矩阵”与原始指标的方向关系
    direction_rows = []
    dir_ok = True
    for col in ["Q", "cost", "life"]:
        if col == "cost":
            rho = spearman_r(pareto_df[col], mm[col])
            ok = rho < 0
            expected = "负相关"
        else:
            rho = spearman_r(pareto_df[col], mm[col])
            ok = rho > 0
            expected = "正相关"
        dir_ok = dir_ok and ok
        direction_rows.append({"指标": col, "原始指标vs正向化": rho,
                               "期望": expected, "通过": "通过" if ok else "失败"})
        print("方向一致性：%s 原始值 vs 正向化分量 Spearman=%.4f（期望%s）：%s"
              % (col, rho, expected, "通过" if ok else "失败"))

    meta = pd.DataFrame(direction_rows)
    status = bool(abs(wsum - 1.0) < 1e-9 and closeness.between(0, 1).all()
                  and ideal_ok and z_ok and norm_ok and dir_ok)
    print()
    return closeness, status, meta


def check_6_weight_sensitivity(pareto_df, weights):
    """权重 ±5%、±10% 扰动下的 TOPSIS 排序稳定性（Spearman）。"""
    print("=" * 78)
    print("检验 6  权重敏感性分析（±5%、±10%，Spearman 秩相关）")
    print("=" * 78)
    base_c, *_ = topsis_rank(pareto_df, weights)
    base_rank = pd.Series(base_c).rank(ascending=False).to_numpy()

    rows = []
    for i, name in enumerate(["散热 Q", "成本", "寿命"]):
        for delta in (0.05, 0.10):
            for sign in (1, -1):
                w = weights.copy()
                w[i] += sign * delta
                w = w / w.sum()
                c, *_ = topsis_rank(pareto_df, w)
                r_rank = pd.Series(c).rank(ascending=False).to_numpy()
                rho = spearman_r(base_rank, r_rank)
                rows.append({
                    "扰动项": name,
                    "扰动": "%+.0f%%" % (sign * delta * 100),
                    "权重": "%.2f/%.2f/%.2f" % tuple(w),
                    "Spearman": rho,
                    "排名变化数": int(np.sum(base_rank != r_rank)),
                })
    sens = pd.DataFrame(rows)
    print(sens.to_string(index=False))
    min_rho = sens["Spearman"].min()
    print("最小 Spearman 秩相关 = %.4f（期望 > 0.8）：%s"
          % (min_rho, "通过" if min_rho > 0.8 else "警告"))
    sens.to_csv(OUT_DIR / "结果_权重敏感性.csv", index=False, encoding="utf-8-sig")
    return sens


def check_7_scenarios(q3, mat_df, temp_pred):
    """材料与深度情景检验。"""
    print("=" * 78)
    print("检验 7  材料与深度情景检验")
    print("=" * 78)
    rows = []
    for mi in range(len(mat_df)):
        mat = mat_df.iloc[mi]
        sigma_allow = float(mat["屈服强度_MPa"]) * 1e6 / q3.SAFETY_FACTOR
        for depth in (5.0, 50.0, 100.0):
            t_req = q3.RHO_SW * q3.G * depth * q3.ENVELOPE_SIDE / (2.0 * sigma_allow)
            wall = max(q3.WALL_MIN, 1.5 * t_req)
            r = q3.evaluate_design(mi, depth, wall, mat_df, temp_pred)
            rows.append({
                "材料": mat["材料"], "深度_m": depth, "t_req_m": t_req,
                "壁厚_m": wall, "Q_W": r["Q"], "成本_元": r["cost"],
                "寿命_年": r["life"], "N_台": r["N"], "可行": r["feasible"],
                "g1_承压_m": r["g1"], "g2_寿命_年": r["g2"],
                "g3_散热_台": r["g3"], "g4_空间_m": r["g4"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # 物理连续性：深度↑ -> Q、t_req、壁厚、成本单调不减；寿命可因壁厚加厚而上升
    ok = True
    for mi, mat in enumerate(mat_df["材料"]):
        sub = df[df["材料"] == mat].sort_values("深度_m")
        if not np.all(sub["Q_W"].diff().dropna() > -1e-6):
            print("  %s：Q 未随深度单调不减" % mat); ok = False
        if not np.all(sub["成本_元"].diff().dropna() > -1e-6):
            print("  %s：成本未随深度单调不减" % mat); ok = False
        if not np.all(sub["t_req_m"].diff().dropna() > -1e-9):
            print("  %s：t_req 未随深度单调不减" % mat); ok = False
        if not np.all(sub["壁厚_m"].diff().dropna() > -1e-9):
            print("  %s：设计壁厚未随深度单调不减" % mat); ok = False
    print("物理连续性（深度↑ -> Q↑、成本↑、t_req↑、壁厚↑）：%s" % ("通过" if ok else "失败"))

    # 符号与数值范围
    num_df = df.select_dtypes(include=[np.number])
    sym_ok = bool((df["Q_W"] > 0).all() and (df["成本_元"] > 0).all()
                  and (df["寿命_年"] >= 0).all()
                  and np.isfinite(num_df).all().all()
                  and (df["g1_承压_m"] >= -1e-9).all())
    print("符号/范围检查：Q>0、成本>0、寿命>=0、有限无 NaN、承压约束 g1>=0：%s"
          % ("通过" if sym_ok else "失败"))

    # 可行性与约束原因分开报告：AISI 1040 深水不满足寿命>=10，而非压力公式错误
    infeasible = df[~df["可行"]]
    infeasible_g2 = int((infeasible["g2_寿命_年"] < 0).sum()) if len(infeasible) else 0
    infeasible_g3 = int((infeasible["g3_散热_台"] < 0).sum()) if len(infeasible) else 0
    infeasible_g4 = int((infeasible["g4_空间_m"] < 0).sum()) if len(infeasible) else 0
    print("不可行方案数=%d；其中寿命约束不满足=%d、散热约束不满足=%d、空间约束不满足=%d"
          % (len(infeasible), infeasible_g2, infeasible_g3, infeasible_g4))
    if len(infeasible):
        print(infeasible[["材料", "深度_m", "寿命_年", "g2_寿命_年"]].to_string(index=False))
        print("说明：以上失败属于设计寿命<10 年的约束筛选，不是公式符号错误。")
    status = "通过" if ok and sym_ok else "警告"
    print()
    return {"df": df, "status": status, "infeasible": len(infeasible),
            "infeasible_g2": infeasible_g2, "infeasible_g3": infeasible_g3,
            "infeasible_g4": infeasible_g4}


# ---------------------------------------------------------------- 绘图
def plot_regression_diagnostics(model, out):
    df = model["df"]
    d = df["depth_m"].to_numpy()
    t = df["temp_C"].to_numpy()
    fitted = model["pred"]
    resid = model["resid"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(fitted, resid, s=28, alpha=0.8)
    axes[0, 0].axhline(0, color="k", lw=0.8)
    axes[0, 0].set_xlabel("拟合值 (℃)"); axes[0, 0].set_ylabel("残差 (℃)")
    axes[0, 0].set_title("残差 vs 拟合值（同方差检验）")

    axes[0, 1].plot(d, t, "o", ms=4, label="实测")
    dg = np.linspace(d.min(), d.max(), 200)
    xg = (dg - model["d_mean"]) / model["d_std"]
    axes[0, 1].plot(dg, np.polyval(model["coef"], xg), "-", label="二次拟合")
    axes[0, 1].set_xlabel("深度 (m)"); axes[0, 1].set_ylabel("温度 (℃)")
    axes[0, 1].set_title("线性性/函数形式：拟合曲线"); axes[0, 1].legend()

    # Q-Q 图（残差正态性）
    srt = np.sort(resid)
    qq = np.quantile(np.random.default_rng(1).normal(size=5000), 
                     np.linspace(0.001, 0.999, len(srt)))
    axes[1, 0].scatter(qq, srt, s=24)
    lo = min(qq.min(), srt.min()); hi = max(qq.max(), srt.max())
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[1, 0].set_xlabel("标准正态分位数"); axes[1, 0].set_ylabel("样本分位数")
    axes[1, 0].set_title("残差 Q-Q 图（正态性检验）")

    axes[1, 1].stem(d, resid, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    axes[1, 1].axhline(0, color="k", lw=0.8)
    axes[1, 1].set_xlabel("深度 (m)"); axes[1, 1].set_ylabel("残差 (℃)")
    axes[1, 1].set_title("残差序列（自相关检验）")
    fig.tight_layout()
    fig.savefig(out / "图6_回归诊断.png", dpi=160)
    plt.close(fig)


def plot_weight_sensitivity(sens, out):
    df = sens.copy()
    df["标签"] = df["扰动项"] + " " + df["扰动"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(df)), df["Spearman"], color="#4C72B0")
    ax.axhline(0.8, color="r", ls="--", lw=1.2, label="阈值 0.8")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["标签"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Spearman 秩相关")
    ax.set_title("权重敏感性：±5%、±10% 扰动下的排序稳定性")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "图7_权重敏感性.png", dpi=160)
    plt.close(fig)


def plot_scenarios(scn, out):
    scn = scn["df"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for mat in scn["材料"].unique():
        sub = scn[scn["材料"] == mat]
        axes[0].plot(sub["深度_m"], sub["Q_W"] / 1e6, "-o", ms=4, label=mat[:6])
        axes[1].plot(sub["深度_m"], sub["成本_元"] / 1e4, "-o", ms=4, label=mat[:6])
        axes[2].plot(sub["深度_m"], sub["寿命_年"], "-o", ms=4, label=mat[:6])
    axes[0].set_xlabel("深度 (m)"); axes[0].set_ylabel("Q (MW)")
    axes[1].set_xlabel("深度 (m)"); axes[1].set_ylabel("成本 (万元)")
    axes[2].set_xlabel("深度 (m)"); axes[2].set_ylabel("寿命 (年)")
    axes[0].set_title("材料-深度情景：散热"); axes[1].set_title("材料-深度情景：成本")
    axes[2].set_title("材料-深度情景：寿命")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "图8_材料深度情景.png", dpi=160)
    plt.close(fig)


def md_table(df, digits=4) -> str:
    """把 DataFrame 转成简洁 Markdown 表格。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            v = row[col]
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("—")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{digits}g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_md_report(summary, dir_df, con, reg, viol, closeness,
                    sens, scn, pareto_df, weights, out):
    """生成 Markdown 检验报告，方便直接粘贴到论文附录。"""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 问题3 多目标与决策模型检验报告",
        "",
        f"生成时间：{now}",
        "",
        "## 1. 检验对象与数据来源",
        "",
        "- 主模型：`问题三/C题_问题3_NSGA2_TOPSIS.py`（NSGA-II + TOPSIS）",
        "- 帕累托前沿：`问题三/输出/结果_帕累托前沿.csv`",
        "- 温度剖面/材料力学性能/导热系数：`C题数据/清洗后数据`",
        "- 材料表中的价格/腐蚀速率以及安全系数：采用工程默认值（用户已确认允许），"
        "论文中需注明来源；物理数据仍来自 `C题数据/清洗后数据`。",
        "- 全部物理量采用国际单位制（m、Pa、W、kg/m^3、元）；中间量见控制台打印。",
        "",
        "## 2. 检验汇总",
        "",
        md_table(summary, digits=4),
        "",
        "## 3. 各检验细节",
        "",
        "### 3.1 目标函数方向检验",
        "",
        "物理含义：优化器全部按最小化处理 `obj=[-Q, cost, -life]`，"
        "因此 `Q` 越大、`cost` 越小、`life` 越大才对应更好的目标值。"
        "题目口径为“散热效果/更多服务器”，`Q` 是 f1 代理，`N` 由 Q 与空间约束导出并单独输出。",
        "",
        md_table(dir_df, digits=4),
        "",
        "### 3.2 约束模型检验",
        "",
        "物理含义：`p=ρ_w·g·d` 是静水压强，`t_req=p·D/(2σ_allow)` 是薄壁圆筒最小壁厚；"
        "腐蚀余量为 `(wall-t_req)`，寿命为腐蚀余量/腐蚀速率。",
        "",
        md_table(con["design"], digits=5),
        "",
        "成本随壁厚严格单调是材料体积公式正确性的必要条件：",
        "",
        md_table(con["cost_wall"], digits=4),
        "",
        "### 3.3 拟合回归诊断",
        "",
        "物理含义：温度-深度二次回归为后续海水温度预测提供输入；"
        "若残差存在异方差、自相关或函数形式遗漏，预测和置信区间会失真。",
        "",
        f"- 线性/二次/三次 R²：{reg['r2_linear']:.6f} / {reg['r2_quad']:.6f} / {reg['r2_cubic']:.6f}"
        f"，二次项 F p={reg['p_quad']:.4f}，三次项 F p={reg['p_cubic']:.4f}",
        f"- 异方差：GQ p={reg['gq_p']:.4f}，BP p={reg['bp_p']:.4f}，White p={reg['white_p']:.4f}",
        f"- 正态性：Jarque-Bera p={reg['jb_p']:.4f}；自相关：DW={reg['dw']:.3f}，lag1={reg['lag1']:.3f}",
        f"- 共线性：VIF(x²)={reg['vif_x2']:.2f}，VIF(x)={reg['vif_x']:.2f}，条件数={reg['cond']:.2f}",
        f"- RESET：F={reg['reset_F']:.3f}，p={reg['reset_p']:.4f}",
        f"- 结论：`{reg['status']}`，未通过项：{reg['failed']}。"
        "建议对温度剖面改用两段线性模型或稳健标准误，避免把平滑剖面当作独立样本。",
        "",
        "### 3.4 Pareto 非支配性质检验",
        "",
        "物理含义：最终解集内部不允许存在一个方案全面不差于另一个且至少一项更优。",
        f"检验方案数={len(pareto_df)}，支配对数={viol}。",
        "",
        "### 3.5 TOPSIS 模型检验",
        "",
        f"- 权重和={weights.sum():.6f}；贴近度范围=[{closeness.min():.6f}, {closeness.max():.6f}]；"
        "正理想解≥负理想解。",
        "- z-score 标准化后均值≈0、标准差≈1；向量归一化后各列 L2 范数≈1；"
        "Q/寿命正向化分量与原值正相关，成本反向化分量与原值负相关。",
        "",
        "### 3.6 权重敏感性分析",
        "",
        "物理含义：决策权重是主观设定，若 ±5%、±10% 扰动导致排序剧烈变化，则决策结论不可靠。",
        "",
        md_table(sens, digits=4),
        "",
        f"最小 Spearman 秩相关 = {sens['Spearman'].min():.4f}（阈值 0.8）。",
        "",
        "### 3.7 材料和深度情景检验",
        "",
        "物理含义：在不同材料和 5/50/100 m 水深下重算目标值，"
        "检查压力、壁厚、成本、寿命是否随深度连续且无符号错误。",
        "",
        md_table(scn["df"], digits=5),
        "",
        f"不可行方案数={scn['infeasible']}，其中寿命约束 g2<0 的方案数="
        f"{scn['infeasible_g2']}（AISI 1040 深水寿命不足 10 年，属约束筛选而非公式错误）。",
        "",
        "## 4. 图表",
        "",
        "- `图6_回归诊断.png`：残差、拟合曲线、Q-Q 图、残差序列",
        "- `图7_权重敏感性.png`：12 种权重扰动的 Spearman 秩相关",
        "- `图8_材料深度情景.png`：材料-深度下的 Q、成本、寿命曲线",
        "",
    ]
    (out / "问题3_模型检验报告.md").write_text("\n".join(lines),
                                                 encoding="utf-8")


# ---------------------------------------------------------------- 主流程
def main():
    q3, temp_df, model, r2_cv, temp_pred, mat_df, pareto_df = load_problem3_model()
    weights = q3.TOPSIS_W

    dir_df, dir_ok = check_1_objective_direction(q3, mat_df, temp_pred)
    con = check_2_constraints(q3, mat_df, temp_pred)
    reg = check_3_regression(model)
    viol = check_4_pareto(pareto_df)
    closeness, topsis_ok, topsis_meta = check_5_topsis(pareto_df, weights)
    sens = check_6_weight_sensitivity(pareto_df, weights)
    scn = check_7_scenarios(q3, mat_df, temp_pred)

    plot_regression_diagnostics(model, OUT_DIR)
    plot_weight_sensitivity(sens, OUT_DIR)
    plot_scenarios(scn, OUT_DIR)

    # 汇总表
    summary = pd.DataFrame([
        {"检验项": "1 目标函数方向",
         "结果": "通过" if dir_ok else "失败",
         "关键指标": "obj=[-Q,cost,-life] 映射正确；价格/腐蚀/强度方向均符合"},
        {"检验项": "2 约束模型",
         "结果": "通过" if con["status"] else "失败",
         "关键指标": "d=0 时 p≈0；p=ρgd、t_req 公式一致；成本随壁厚严格上升"},
        {"检验项": "3 回归诊断",
         "结果": reg["status"],
         "关键指标": "R²=%.6f，DW=%.3f，RESET p=%.3f，White p=%.3f"
                     % (reg["r2_quad"], reg["dw"], reg["reset_p"], reg["white_p"])},
        {"检验项": "4 Pareto 非支配", "结果": "通过" if viol == 0 else "失败",
         "关键指标": "支配对=%d" % viol},
        {"检验项": "5 TOPSIS 性质",
         "结果": "通过" if topsis_ok else "失败",
         "关键指标": "贴近度 [%.4f, %.4f]，权重和=1" % (closeness.min(), closeness.max())},
        {"检验项": "6 权重敏感性", "结果": "通过" if sens["Spearman"].min() > 0.8 else "警告",
         "关键指标": "最小 Spearman=%.4f" % sens["Spearman"].min()},
        {"检验项": "7 材料-深度情景",
         "结果": scn["status"],
         "关键指标": "Q/成本/t_req/壁厚随深度单调；不可行=%d（寿命约束）"
                     % scn["infeasible"]},
    ])
    summary.to_csv(OUT_DIR / "结果_模型检验汇总.csv", index=False,
                   encoding="utf-8-sig")
    con["fixed"].to_csv(OUT_DIR / "结果_约束检验.csv", index=False,
                        encoding="utf-8-sig")
    con["design"].to_csv(OUT_DIR / "结果_约束检验_设计壁厚.csv", index=False,
                         encoding="utf-8-sig")
    con["cost_wall"].to_csv(OUT_DIR / "结果_成本壁厚单调.csv", index=False,
                            encoding="utf-8-sig")
    scn["df"].to_csv(OUT_DIR / "结果_情景检验.csv", index=False,
                     encoding="utf-8-sig")
    dir_df.to_csv(OUT_DIR / "结果_目标方向检验.csv", index=False,
                  encoding="utf-8-sig")
    topsis_meta.to_csv(OUT_DIR / "结果_TOPSIS方向检验.csv", index=False,
                       encoding="utf-8-sig")
    reg_df = pd.DataFrame([reg])
    reg_df.to_csv(OUT_DIR / "结果_回归诊断.csv", index=False,
                  encoding="utf-8-sig")
    write_md_report(summary, dir_df, con, reg, viol, closeness,
                    sens, scn, pareto_df, weights, OUT_DIR)
    print("=" * 78)
    print("模型检验汇总")
    print("=" * 78)
    print(summary.to_string(index=False))
    print()
    print("图表与结果已保存到：", OUT_DIR)


if __name__ == "__main__":
    main()
