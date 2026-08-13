# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import warnings
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ==================================================================
# 核心算法（独立模块：C题_问题2_核心算法.py）
# 函数输入/输出说明见《算法设计_问题2_核心算法.md》
# ==================================================================
from C题_问题2_核心算法 import (
    ENVELOPE_SIDE, HULL_LENGTH, Q0, T_MAX, T_INF,
    SERVER_W, SERVER_H, SERVER_L,
    K_FIN, WALL, K_WALL_DEFAULT,
    NF_MIN, NF_MAX, HF_MIN, HF_MAX, DF_MIN, DF_MAX,
    air_props, sea_props, h_horizontal_cylinder,
    _char_len, _load_k_wall, _k_wall,
    _inner_geometry, _wall_resistance, _ra_number,
    _solve_thermal_network, _fin_efficiency,
    _solve_fin_and_network, _build_geometry, _raw_metrics,
    evaluate_design, _infeasible_result, _find_project_root,
    solve_q2,
)


# ==================================================================
# 三、优化器参数区（GA / SLSQP；翅片设计变量上下界见核心算法模块）
# ==================================================================
# ---- GA 遗传算子参数（清单未给数值，以下为工程默认值【假设值】）----
GA_POP = 80          # 种群规模
GA_GEN = 60          # 迭代代数
GA_PC = 0.85         # 交叉概率
GA_PM = 0.10         # 变异概率
GA_ELITE = 2         # 精英保留数
GA_SEED = 20260813   # 随机种子（保证可复现）

# ---- SLSQP 局部寻优参数 ----
SLSQP_MAX_ITER = 100
SLSQP_TOL = 1e-6


# ==================================================================
# 四点五、可视化：最优结构横截面温度云图
# ==================================================================
def _fin_temp_profile(Hf: float, df: float, h: float, s):
    """绝热翅尖一维直翅解：T(s)=Tinf+(Tmax-Tinf)*cosh(m(Hf-s))/cosh(mHf)。

    与代码主模型中的 eta_f = tanh(mHf)/(mHf) 使用同一套一维直翅假设，
    保证云图颜色与数值计算的散热口径一致。
    """
    m = np.sqrt(2.0 * h / (K_FIN * df))
    mH = m * Hf
    s = np.asarray(s, dtype=float)
    if mH < 1e-12:
        return np.full_like(s, T_MAX, dtype=float)
    return T_INF + (T_MAX - T_INF) * np.cosh(m * (Hf - np.clip(s, 0.0, Hf))) / np.cosh(mH)


def _find_project_root(start: Path | None = None) -> Path:
    """自动寻找工程根目录（含 outputs / C题数据 的上级目录）。

    使脚本不依赖固定的绝对路径，可在任意工作目录下直接运行；
    若找不到标志目录，则回退到脚本所在目录的上级。
    """
    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents]
    if start is not None:
        candidates.insert(0, Path(start).resolve())
    for p in candidates:
        if (p / "outputs").is_dir() or (p / "C题数据").is_dir():
            return p
    return here.parent


def _steady_conduction_field(shape: str, nf: int, Hf: float, df: float,
                             h: float, n_grid: int = 1501,
                             tol: float = 1e-6):
    """有限体积 + 共轭梯度求解横截面二维稳态导热场。

    内舱节点固定为 T_MAX（允许最高温度边界），固体外露面对海水用 Robin
    对流边界：-k*dT/dn = h*(T-T_inf)。只对固体节点建立稀疏邻接，
    避免在整张网格上做慢速 SOR，运行时间约 1 秒量级。材料分区与主模型
    一致：基体壁为壳体材料 K_WALL，翅片为 K_FIN，节点间导热用调和平均。
    """
    x = np.linspace(-0.5, 0.5, n_grid)
    dx = x[1] - x[0]
    XX, YY = np.meshgrid(x, x)

    # ---- 几何：基体 + 翅片为固体，内舱为 Dirichlet 区域，其余为海水 ----
    if shape == "cuboid":
        a = ENVELOPE_SIDE - 2.0 * Hf
        half = a / 2.0
        inner = half - WALL
        interior = (np.abs(XX) < inner) & (np.abs(YY) < inner)
        solid = np.maximum(np.abs(XX), np.abs(YY)) <= half
        spacing = a / nf
        pos_y = (YY + half) / spacing
        k_y = np.floor(pos_y).astype(int)
        d_y = np.minimum(
            np.abs(YY - (-half + spacing * (k_y + 0.5))),
            np.abs(YY - (-half + spacing * (k_y + 1.5))),
        )
        pos_x = (XX + half) / spacing
        k_x = np.floor(pos_x).astype(int)
        d_x = np.minimum(
            np.abs(XX - (-half + spacing * (k_x + 0.5))),
            np.abs(XX - (-half + spacing * (k_x + 1.5))),
        )
        solid |= (XX >= half) & (d_y <= df / 2.0)
        solid |= (XX <= -half) & (d_y <= df / 2.0)
        solid |= (YY >= half) & (d_x <= df / 2.0)
        solid |= (YY <= -half) & (d_x <= df / 2.0)
        solid &= ~interior
        extra = (half, inner)
    else:
        R_base = 0.5 - Hf
        RR = np.hypot(XX, YY)
        interior = RR < R_base - WALL
        solid = RR <= R_base
        solid &= ~interior
        dtheta = 2.0 * np.pi / nf
        phi = np.arctan2(YY, XX)
        idx = np.round(phi / dtheta)
        ang = phi - idx * dtheta
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        tt = RR * ang
        solid |= (RR >= R_base) & (RR <= 0.5) & (np.abs(tt) <= df / 2.0)
        solid &= ~interior
        extra = (R_base,)

    outside = ~(solid | interior)
    active = solid.copy()

    # ---- 材料分区：基体壁为壳体材料 K_WALL，翅片为 K_FIN ----
    k_grid = np.where(solid, K_FIN, 0.0)
    if shape == "cuboid":
        ring = (np.abs(XX) <= half) & (np.abs(YY) <= half)
    else:
        ring = RR <= R_base
    k_grid[ring & ~interior] = _k_wall()

    # ---- 稀疏邻接：每个固体节点最多 4 个固体邻居 ----
    idx_flat = np.arange(solid.size).reshape(solid.shape)
    local_idx = np.full(solid.size, -1, dtype=np.int64)
    act_flat = np.flatnonzero(active)
    local_idx[act_flat] = np.arange(act_flat.size, dtype=np.int64)

    def _shift(a, axis, back):
        out = np.full(solid.shape, -1, dtype=np.int64)
        if axis == 1:
            if back:
                out[:, 1:] = a[:, :-1]
            else:
                out[:, :-1] = a[:, 1:]
        else:
            if back:
                out[1:, :] = a[:-1, :]
            else:
                out[:-1, :] = a[1:, :]
        return out

    nbs = []
    for axis, back in ((1, False), (1, True), (0, False), (0, True)):
        nb_flat = _shift(idx_flat, axis, back)[active]
        nbs.append(np.where(nb_flat >= 0, local_idx[nb_flat], -1))
    nbs = np.stack(nbs, axis=1)

    iE = np.zeros_like(interior); iE[:, :-1] = interior[:, 1:]
    iW = np.zeros_like(interior); iW[:, 1:] = interior[:, :-1]
    iN = np.zeros_like(interior); iN[:-1, :] = interior[1:, :]
    iS = np.zeros_like(interior); iS[1:, :] = interior[:-1, :]
    oE = np.zeros_like(outside); oE[:, :-1] = outside[:, 1:]
    oW = np.zeros_like(outside); oW[:, 1:] = outside[:, :-1]
    oN = np.zeros_like(outside); oN[:-1, :] = outside[1:, :]
    oS = np.zeros_like(outside); oS[1:, :] = outside[:-1, :]

    n_dir = (iE + iW + iN + iS)[active].astype(float)
    n_exp = (oE + oW + oN + oS)[active].astype(float)
    k_act = k_grid[active]
    k_nb = []
    for d in range(4):
        nb = nbs[:, d]
        k_j = np.where(nb >= 0, k_act[nb], 0.0)
        k_nb.append(np.where(k_j > 0.0,
                             2.0 * k_act * k_j / (k_act + k_j), 0.0))
    k_nb = np.stack(k_nb, axis=1)

    diag = k_nb.sum(axis=1) + k_act * n_dir + h * dx * n_exp
    b = k_act * T_MAX * n_dir + h * dx * T_INF * n_exp

    u = np.full(act_flat.size, 60.0)
    r = b - diag * u
    for d in range(4):
        nb = nbs[:, d]
        r = r + k_nb[:, d] * np.where(nb >= 0, u[nb], 0.0)
    p = r.copy()
    rr = float(r @ r)
    for _ in range(20000):
        ap = diag * p
        for d in range(4):
            nb = nbs[:, d]
            ap = ap - k_nb[:, d] * np.where(nb >= 0, p[nb], 0.0)
        pap = float(p @ ap)
        if pap <= 0.0:
            break
        alpha = rr / pap
        u = u + alpha * p
        r = r - alpha * ap
        rr_new = float(r @ r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new
        if rr < tol * tol:
            break

    T = np.full(solid.shape, T_INF, dtype=float)
    T[interior] = T_MAX
    T.flat[act_flat] = u

    # ---- 海水侧热边界层示意：固体表面温度指数衰减到 T_INF ----
    # 物理量纲：k_sea≈0.6 W/(m·K)（海水 60 ℃ 左右），h 单位 W/(m²·K)，
    # 热边界层厚度 delta_t ≈ k_sea/h ≈ 0.6/980 ≈ 0.0006 m（约 0.6 mm）。
    # 旧版固定取 6 mm 会把整片海水染成高温，视觉上“温度变化不合理”，
    # 故改为按 k_sea/h 估算的薄边界层，并至少保留 2 个网格宽度以便成图。
    k_sea = 0.6          # W/(m·K)，海水导热系数近似值
    delta_m = max(2.0 * dx, k_sea / max(h, 1e-6))   # 热边界层厚度，m
    delta_cells = delta_m / dx
    iterations = int(np.ceil(delta_cells)) + 2
    dist = np.full(solid.shape, 1e12, dtype=float)
    t_near = np.full(solid.shape, T_INF, dtype=float)
    dist[solid] = 0.0
    t_near[solid] = T[solid]
    for axis in (1, 0):
        for back in (False, True):
            cand_d = np.full_like(dist, 1e12)
            cand_t = np.full_like(t_near, T_INF)
            if axis == 1:
                if back:
                    cand_d[:, 1:] = dist[:, :-1] + 1.0
                    cand_t[:, 1:] = t_near[:, :-1]
                else:
                    cand_d[:, :-1] = dist[:, 1:] + 1.0
                    cand_t[:, :-1] = t_near[:, 1:]
            else:
                if back:
                    cand_d[1:, :] = dist[:-1, :] + 1.0
                    cand_t[1:, :] = t_near[:-1, :]
                else:
                    cand_d[:-1, :] = dist[1:, :] + 1.0
                    cand_t[:-1, :] = t_near[1:, :]
            m = cand_d < dist
            dist[m] = cand_d[m]
            t_near[m] = cand_t[m]
    sea_layers = outside & (dist <= delta_cells)
    T[sea_layers] = T_INF + (t_near[sea_layers] - T_INF) * np.exp(
        -dist[sea_layers] / delta_cells)
    return T, dx, interior, solid, outside, extra


def _draw_shape_schematic(ax, res: dict, is_best: bool) -> None:
    """左两图：圆柱 / 长方体结构示意（翅片厚度与高度为示意放大）。"""
    from matplotlib.patches import Circle, Polygon, Rectangle

    shape = res["shape"]
    nf = int(res["nf"])
    Hf = float(res["Hf"])
    df = float(res["df"])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.58, 0.58)
    # 最优结构用浅金色底框突出，对照结构用浅灰底框淡化，一眼区分最优解
    if is_best:
        bg_color = "#fff8e1"
        frame_color = "#b8860b"
        frame_lw = 2.6
    else:
        bg_color = "#f0f0f0"
        frame_color = "#9e9e9e"
        frame_lw = 1.2
    # 注意：axis("off") 会隐藏坐标轴背景色，故用显式矩形绘制底色/边框
    ax.add_patch(Rectangle((-0.58, -0.58), 1.16, 1.16,
                           facecolor=bg_color, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False,
                           edgecolor=frame_color, lw=frame_lw, zorder=1))
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)
        spine.set_linewidth(frame_lw)

    half_s = 0.34
    Hf_s = 0.5 - half_s
    df_s = 0.014
    if shape == "cylinder":
        ax.add_patch(Circle((0, 0), half_s, facecolor="#c9cdd4",
                            edgecolor="#333333", lw=1.5))
        ax.add_patch(Circle((0, 0), half_s - 0.02, facecolor="#ffffff",
                            edgecolor="#666666", lw=1.0))
        nf_vis = 24
        for k in range(nf_vis):
            th = 2.0 * np.pi * k / nf_vis
            e_r = np.array([np.cos(th), np.sin(th)])
            e_t = np.array([-np.sin(th), np.cos(th)])
            hw = df_s / 2.0
            corners = [
                e_r * half_s + e_t * (-hw),
                e_r * half_s + e_t * hw,
                e_r * (half_s + Hf_s) + e_t * hw,
                e_r * (half_s + Hf_s) + e_t * (-hw),
            ]
            ax.add_patch(Polygon(corners, closed=True, facecolor="#aab2bd",
                                 edgecolor="#444444", lw=0.8))
    else:
        ax.add_patch(Rectangle((-half_s, -half_s), 2.0 * half_s, 2.0 * half_s,
                               facecolor="#c9cdd4", edgecolor="#333333",
                               lw=1.5))
        ax.add_patch(Rectangle((-half_s + 0.02, -half_s + 0.02),
                               2.0 * (half_s - 0.02), 2.0 * (half_s - 0.02),
                               facecolor="#ffffff", edgecolor="#666666",
                               lw=1.0))
        nf_vis = 20
        spacing_s = 2.0 * half_s / nf_vis
        centers = -half_s + spacing_s * (np.arange(nf_vis) + 0.5)
        for yc in centers:
            ax.add_patch(Rectangle((half_s, yc - df_s / 2.0), Hf_s, df_s,
                                   facecolor="#aab2bd", edgecolor="#444444",
                                   lw=0.6))
            ax.add_patch(Rectangle((-half_s - Hf_s, yc - df_s / 2.0),
                                   Hf_s, df_s, facecolor="#aab2bd",
                                   edgecolor="#444444", lw=0.6))
            ax.add_patch(Rectangle((yc - df_s / 2.0, half_s), df_s, Hf_s,
                                   facecolor="#aab2bd", edgecolor="#444444",
                                   lw=0.6))
            ax.add_patch(Rectangle((yc - df_s / 2.0, -half_s - Hf_s),
                                   df_s, Hf_s, facecolor="#aab2bd",
                                   edgecolor="#444444", lw=0.6))

    shape_cn = "圆柱" if shape == "cylinder" else "长方体"
    suffix = "（最优）★" if is_best else "（对照）"
    title = "%s%s\nN=%d 台\nnf=%d" % (shape_cn, suffix, res["N"], nf)
    if shape == "cuboid":
        title += "/面"
    title += "，Hf=%.1f mm，df=%.1f mm" % (Hf * 1000.0, df * 1000.0)
    ax.set_title(title, fontsize=9)
    ax.text(0.0, -0.57, "翅片尺寸为示意放大",
            ha="center", va="top", fontsize=8, color="#555555")


def plot_optimal_cross_section(best: dict,
                               shape_results: dict | None = None) -> Path | None:
    """输出“外形对比 + 最优长方体横截面温度云图”（PNG）。

    左两图为圆柱 / 长方体结构示意，突出最优解为长方体；右图为有限体积 +
    共轭梯度求解的横截面二维稳态导热场。内壁取 T_MAX，外露面对海水取
    Robin 对流边界，云图只用于温度梯度展示，N 仍以主模型为准。
    """
    if not best.get("feasible", False):
        return None
    shape = best["shape"]
    if shape not in ("cylinder", "cuboid"):
        return None
    nf = int(best["nf"])
    Hf = float(best["Hf"])
    df = float(best["df"])
    res = evaluate_design(shape, nf, Hf, df, collect_warnings=False)
    if not res["feasible"]:
        return None

    # 结果输出到脚本所在目录（问题二文件夹）下的 outputs/，与问题一目录习惯一致
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        print("[警告] 未安装 matplotlib，跳过问题2云图输出（不影响数值结果）。")
        return None

    for _font in ("Microsoft YaHei", "SimHei", "SimSun",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        try:
            font_manager.findfont(_font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    h = res["h"]
    n_grid = 1501
    T, dx, interior, solid, outside, extra = _steady_conduction_field(
        shape, nf, Hf, df, h, n_grid=n_grid)
    if shape == "cuboid":
        half, inner = extra
        R_base = None
    else:
        half = inner = None
        R_base = extra[0]
    x_fine = np.linspace(-0.5, 0.5, n_grid)
    XX, YY = np.meshgrid(x_fine, x_fine)
    t_solid_min = float(np.min(T[solid]))

    # ---- 数量级校验：取关键位置温度（SI 单位 ℃），便于人工核对 ----
    def _T(px: float, py: float) -> float:
        ix = int(round((px + 0.5) / dx))
        iy = int(round((py + 0.5) / dx))
        return float(T[iy, ix])

    if shape == "cuboid":
        spacing = (2.0 * half) / nf
        fin_y = -half + spacing * (nf // 2 + 0.5)    # 取右面一根翅片中心线
        t_wall = _T(half - 0.001, 0.0)               # 外壁温度，℃
        t_fin_base = _T(half + 0.0008, fin_y)        # 翅根温度，℃
        t_fin_tip = _T(half + Hf - 0.0002, fin_y)    # 翅尖温度，℃
    else:
        fin_y = 0.0
        t_wall = _T(R_base - 0.001, 0.0)
        t_fin_base = _T(R_base + 0.0008, 0.0)
        t_fin_tip = _T(0.5 - 0.0002, 0.0)
    print("[量纲校验] 温度场（SI 单位 ℃）：内壁 T_max=%.1f ℃（固定边界）→ "
          "外壁≈%.1f ℃ → 翅根≈%.1f ℃ → 翅尖≈%.1f ℃ → 海水 T_inf=%.1f ℃；"
          "固体最低≈%.1f ℃，h=%.0f W/(m^2·K)"
          % (T_MAX, t_wall, t_fin_base, t_fin_tip, T_INF, t_solid_min, h))

    cmap = LinearSegmentedColormap.from_list(
        "q2_thermal",
        ["#003f5c", "#7a5195", "#ef5675", "#ffa600", "#ffd700"],
    )
    norm = plt.Normalize(T_INF, T_MAX)

    fig, (ax_cyl, ax_box, ax) = plt.subplots(
        1, 3, figsize=(15.0, 6.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})
    results = shape_results or {}
    cyl_res = results.get("cylinder")
    box_res = results.get("cuboid")
    n_cyl = (cyl_res["N"] if cyl_res is not None
             and cyl_res.get("feasible", False) else 0)
    n_box = (box_res["N"] if box_res is not None
             and box_res.get("feasible", False) else 0)
    fig.suptitle("问题2 结构对比：圆柱 N=%d 台（对照） ｜ 长方体 N=%d 台（最优）★"
                 % (n_cyl, n_box), fontsize=13)
    if cyl_res is not None and cyl_res.get("feasible", False):
        _draw_shape_schematic(ax_cyl, cyl_res, False)
    else:
        ax_cyl.axis("off")
    if box_res is not None and box_res.get("feasible", False):
        _draw_shape_schematic(ax_box, box_res, True)
    else:
        ax_box.axis("off")

    # ---- 主云图：对 1501 网格降采样绘制，速度与清晰度兼顾 ----
    step = 2
    cf = ax.contourf(XX[::step, ::step], YY[::step, ::step],
                     T[::step, ::step],
                     levels=np.linspace(T_INF, T_MAX, 120),
                     cmap=cmap, extend="both")
    if shape == "cuboid":
        ax.add_patch(Rectangle((-half, -half), 2.0 * half, 2.0 * half,
                               fill=False, edgecolor="white", lw=2.0))
        ax.add_patch(Rectangle((-inner, -inner), 2.0 * inner, 2.0 * inner,
                               fill=False, edgecolor="white", lw=1.2, ls="--"))
    else:
        ax.add_patch(Circle((0, 0), R_base, fill=False,
                            edgecolor="white", lw=2.0))
        ax.add_patch(Circle((0, 0), R_base - WALL, fill=False,
                            edgecolor="white", lw=1.2, ls="--"))
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False,
                           edgecolor="#333333", lw=1.0))

    # 主云图用局部放大窗口：让内壁→外壁→翅尖→海水的温度梯度清晰可见
    vx0, vx1 = 0.28, 0.505
    vy0, vy1 = -0.1125, 0.1125

    # ---- 关键温度标注：直接说明“温度变化”的合理性 ----
    def _lbl(txt, px, py, tx, ty):
        ax.annotate(txt, xy=(px, py), xytext=(tx, ty), fontsize=9,
                    color="black", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="#333333",
                                    lw=1.0))

    if shape == "cuboid":
        _lbl("内壁 80 ℃", inner - 0.005, 0.0, 0.325, 0.025)
        _lbl("外壁≈%.0f ℃" % t_wall, half - 0.001, 0.0,
             0.325, 0.048)
        _lbl("翅尖≈%.0f ℃" % t_fin_tip, half + Hf - 0.0004, fin_y,
             0.432, 0.100)
    else:
        _lbl("内壁 80 ℃", R_base - WALL - 0.005, 0.0,0.325, 0.025)
        _lbl("外壁≈%.0f ℃" % t_wall, R_base - 0.001, 0.0,
             0.325, 0.048)
        _lbl("翅尖≈%.0f ℃" % t_fin_tip, 0.5 - 0.0004, 0.0,
             0.432, 0.100)
    _lbl("海水远场 20 ℃（k_sea/h≈0.6 mm）", 0.4985, 0.0,
         0.345, 0.096)

    # ---- 翅簇放大：用同一 2D 求解场，不另画解析近似 ----
    if shape == "cuboid":
        zx0, zx1 = inner - 0.004, 0.506
        zy0, zy1 = -0.036, 0.036
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                               fill=False, edgecolor="white", lw=1.3, ls="--"))
        axins = ax.inset_axes([0.58, 0.08, 0.40, 0.40])
        rows = np.where((x_fine >= zx0) & (x_fine <= zx1))[0]
        cols = np.where((x_fine >= zy0) & (x_fine <= zy1))[0]
        axins.pcolormesh(XX[np.ix_(rows, cols)], YY[np.ix_(rows, cols)],
                         T[np.ix_(rows, cols)], cmap=cmap, norm=norm,
                         shading="auto", rasterized=True)
        spacing = (2.0 * half) / nf
        centers = -half + spacing * (np.arange(nf) + 0.5)
        for yc in centers:
            if yc < zy0 - df or yc > zy1 + df:
                continue
            axins.add_patch(Rectangle((half, yc - df / 2.0), Hf, df,
                                      fill=False, edgecolor="white",
                                      lw=0.7, zorder=3))
        axins.add_patch(Rectangle((inner, zy0), half - inner, zy1 - zy0,
                                  fill=False, edgecolor="black",
                                  lw=0.8, zorder=2))
        axins.axvline(0.5, color="black", lw=1.0, zorder=2)
        axins.set_xlim(zx0, zx1)
        axins.set_ylim(zy0, zy1)
        axins.set_aspect("equal")
        axins.set_facecolor("#d9eef7")
        axins.tick_params(labelsize=8)
        ax.text(0.55, 0.45, "翅簇放大（2D 求解场）", fontsize=9, 
        transform=ax.transAxes, ha='center', va='bottom')
    else:
        zx0, zx1 = 0.30, 0.506
        zy0, zy1 = -0.10, 0.10
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                               fill=False, edgecolor="white", lw=1.3, ls="--"))
        axins = ax.inset_axes([0.58, 0.08, 0.40, 0.40])
        rows = np.where((x_fine >= zx0) & (x_fine <= zx1))[0]
        cols = np.where((x_fine >= zy0) & (x_fine <= zy1))[0]
        axins.pcolormesh(XX[np.ix_(rows, cols)], YY[np.ix_(rows, cols)],
                         T[np.ix_(rows, cols)], cmap=cmap, norm=norm,
                         shading="auto", rasterized=True)
        axins.add_patch(Circle((0, 0), R_base, fill=False,
                               edgecolor="white", lw=1.2))
        axins.set_xlim(zx0, zx1)
        axins.set_ylim(zy0, zy1)
        axins.set_aspect("equal")
        axins.set_facecolor("#d9eef7")
        axins.tick_params(labelsize=8)  
        axins.set_title("右端扇区放大（2D 求解场）", fontsize=9)

    shape_cn = "长方体" if shape == "cuboid" else "圆柱"
    ax.set_aspect("equal")
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(
        "最优结构：%s（N=%d 台）— 右侧翅片局部放大（2D 稳态导热）\n"
        "范围 x∈[%.3f, %.3f] m、y∈[%.3f, %.3f] m；\n"
        "nf=%d，Hf=%.1f mm，df=%.1f mm，h=%.0f W/(m^2·K)\n"
        "温度变化：内舱 80 ℃ → 外壁≈%.1f ℃ → 翅尖≈%.1f ℃ → 海水 20 ℃"
        % (shape_cn, best["N"], vx0, vx1, vy0, vy1,
           nf, Hf * 1000.0, df * 1000.0, h, t_wall, t_fin_tip),
        fontsize=10.5,y=1.05
    )
    cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("温度 / ℃")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png = out_dir / "问题2_最优结构横截面温度云图.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("问题2云图已保存：%s" % png)
    return png


# ==================================================================
# 五、手写 GA 遗传算法
# ==================================================================
@dataclass
class GAResult:
    best_chrom: list
    best_fitness: float
    history: list          # [(代数, 当代最优适应度), ...]
    warnings: list


class SimpleGA:
    """实数/整数混合编码遗传算法。

    染色体 = [nf(整数), Hf(实数), df(实数)]
    适应度 = N（越大越好）；不可行个体给 -1e9 惩罚值。
    """

    def __init__(self, bounds, pop_size=GA_POP, generations=GA_GEN,
                 pc=GA_PC, pm=GA_PM, elite=GA_ELITE, seed=GA_SEED):
        self.bounds = bounds                      # [(low, high), ...] 每基因
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.elite = elite
        self.rng = np.random.default_rng(seed)
        self.warnings = []

    def _rand_chrom(self) -> list:
        """随机生成一个染色体：nf 取整数，Hf/df 取连续值。"""
        chrom = []
        for i, (lo, hi) in enumerate(self.bounds):
            v = self.rng.uniform(lo, hi)
            chrom.append(int(round(v)) if i == 0 else float(v))
        return chrom

    def _clip(self, chrom: list) -> list:
        """把染色体拉回变量区间，防止变异越界。"""
        out = []
        for i, (lo, hi) in enumerate(self.bounds):
            v = float(chrom[i])
            v = min(max(v, lo), hi)
            out.append(int(round(v)) if i == 0 else v)
        return out

    @staticmethod
    def _tournament(pop, fits, k=3, rng=None):
        """二元锦标赛（k 元变体）：随机抽 k 个，取适应度最高者。"""
        idx = rng.integers(0, len(pop), size=k)
        best = idx[0]
        for j in idx[1:]:
            if fits[j] > fits[best]:
                best = j
        return pop[best]

    @staticmethod
    def _crossover(p1, p2, pc, rng):
        """均匀交叉：每个基因以 0.5 概率交换；整体以 pc 概率执行。"""
        if rng.random() >= pc:
            return p1[:], p2[:]
        c1, c2 = [], []
        for g1, g2 in zip(p1, p2):
            if rng.random() < 0.5:
                c1.append(g2)
                c2.append(g1)
            else:
                c1.append(g1)
                c2.append(g2)
        return c1, c2

    def _mutate(self, chrom, pm):
        """变异：nf 整数随机重置；Hf/df 高斯扰动后截断到区间。"""
        out = chrom[:]
        for i, (lo, hi) in enumerate(self.bounds):
            if self.rng.random() >= pm:
                continue
            if i == 0:
                out[i] = int(round(self.rng.uniform(lo, hi)))
            else:
                sigma = 0.08 * (hi - lo)          # 变异步长为区间宽度的 8%
                out[i] = float(out[i]) + self.rng.normal(0.0, sigma)
        return self._clip(out)

    def run(self, fitness) -> GAResult:
        """主循环：初始化 -> 逐代选择/交叉/变异 -> 精英保留。"""
        pop = [self._rand_chrom() for _ in range(self.pop_size)]
        fits = [fitness(c) for c in pop]
        history = []

        for gen in range(self.generations):
            # ---- 精英保留：直接进入下一代 ----
            order = np.argsort(fits)[::-1]
            new_pop = [pop[i] for i in order[:self.elite]]
            new_fits = [fits[i] for i in order[:self.elite]]

            # ---- 交叉 + 变异生成其余个体 ----
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits, rng=self.rng)
                p2 = self._tournament(pop, fits, rng=self.rng)
                c1, c2 = self._crossover(p1, p2, self.pc, self.rng)
                c1, c2 = self._mutate(c1, self.pm), self._mutate(c2, self.pm)
                new_pop.append(c1)
                new_fits.append(fitness(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
                    new_fits.append(fitness(c2))

            pop, fits = new_pop, new_fits
            history.append((gen + 1, float(max(fits))))

        best_idx = int(np.argmax(fits))
        return GAResult(
            best_chrom=pop[best_idx],
            best_fitness=float(fits[best_idx]),
            history=history,
            warnings=self.warnings,
        )


def make_fitness(shape: str):
    """适应度函数：主项 = 装机台数 N，N 越大适应度越高。

    同一 N 存在大量平局时，附加 1e-9*N_theory 的极小次级项，
    让 GA 在“都能放同样多服务器”的设计中偏向散热裕度更大的翅片，
    不影响 N 的主导地位，也便于 SLSQP 从更合理的初值出发。
    """
    def fitness(chrom):
        nf = int(round(chrom[0]))
        res = evaluate_design(shape, nf, chrom[1], chrom[2], collect_warnings=False)
        if not res["feasible"]:
            return -1e9                             # 不可行惩罚
        return float(res["N"]) + 1e-9 * res["n_theory"]
    return fitness


# ==================================================================
# 六、手写 SLSQP：BFGS + 主动集 QP + Armijo 线搜索
# ==================================================================
def _numeric_grad(f, x, fd_step=1e-5):
    """中心差分数值梯度。"""
    g = np.empty_like(x, dtype=float)
    for i in range(x.size):
        step = max(1e-7, fd_step * max(1.0, abs(float(x[i]))))
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        g[i] = (f(xp) - f(xm)) / (2.0 * step)
    return g


def _solve_kkt(H, g, A, c):
    """求解 KKT 方程：
        [H  -A^T] [d ]   [-g]
        [A   0  ] [λ ] = [-c]
    其中 A 为激活约束梯度行，c 为激活约束当前值（要求 A d = -c）。
    """
    n = H.shape[0]
    m = A.shape[0]
    M = np.zeros((n + m, n + m))
    rhs = np.zeros(n + m)
    M[:n, :n] = H
    M[:n, n:] = -A.T
    M[n:, :n] = A
    rhs[:n] = -g
    rhs[n:] = -c
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]   # 奇异时最小二乘兜底
    if not np.all(np.isfinite(sol)):
        return None
    return sol[:n], sol[n:]


def _solve_qp(H, g, A, c, tol=1e-7):
    """主动集法求解 QP 子问题：
        min  0.5*d' H d + g' d
        s.t. A d + c >= 0
    约束个数少（本问题 m<=2），直接枚举所有激活集合并取可行最优。
    """
    m = A.shape[0]
    best = None
    for mask in range(1 << m):
        idx = [i for i in range(m) if (mask >> i) & 1]
        if not idx:
            d = -np.linalg.solve(H, g)
            lam = np.zeros(m)
        else:
            sol = _solve_kkt(H, g, A[idx], c[idx])
            if sol is None:
                continue
            d, lam_act = sol
            lam = np.zeros(m)
            lam[idx] = lam_act
            if np.any(lam_act < -tol):
                continue
        # 所有约束都必须满足线性化可行性
        if np.any(A @ d + c < -tol):
            continue
        val = 0.5 * d @ H @ d + g @ d
        if best is None or val < best[0]:
            best = (float(val), d.copy(), lam.copy())
    return best


def slsqp_local(objective, constraints, x0, bounds,
                max_iter=SLSQP_MAX_ITER, tol=SLSQP_TOL) -> dict:
    """手写 SLSQP 局部寻优。

    输入：
        objective  : 待最小化目标 f(x)
        constraints: 不等式约束 g_i(x) >= 0 的函数列表
        x0         : 初始点（GA 最优个体）
        bounds     : 每维下界/上界
    输出：
        dict，含 x、fval、feasible、iterations、warnings。
    """
    x = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bounds)],
                 dtype=float)
    n = x.size
    B = np.eye(n)                                  # 逆 Hessian 近似（BFGS）
    g_obj = _numeric_grad(objective, x)

    def all_c(xx):
        return np.array([ci(xx) for ci in constraints], dtype=float)

    def merit(xx):
        """L1 罚函数：目标 + 罚参数 * 约束违和。"""
        return objective(xx) + mu * float(np.sum(np.maximum(0.0, -all_c(xx))))

    mu = 10.0
    warnings_collected = []
    best_x = x.copy()
    best_f = objective(x)
    best_viol = float(np.sum(np.maximum(0.0, -all_c(x))))

    for it in range(max_iter):
        c = all_c(x)
        A = np.vstack([_numeric_grad(ci, x) for ci in constraints]) \
            if constraints else np.empty((0, n))
        qp = _solve_qp(B, g_obj, A, c)
        if qp is not None:
            _, d, lam = qp
            # L1 罚参数取乘子量级 10 倍，避免目标量级远大于罚项
            mu = max(mu, 1.0, 10.0 * float(np.max(np.abs(lam))))
        else:
            # QP 子问题退化：改用罚函数（含约束违和）的负梯度方向恢复可行
            d = -_numeric_grad(merit, x)
            warnings_collected.append("QP 子问题退化，改用罚函数梯度方向")
        if qp is not None and np.linalg.norm(d) < tol and np.all(c >= -tol):
            break                                    # 收敛：QP 步长足够小且可行
        if np.linalg.norm(d) < tol:
            d = -_numeric_grad(merit, x)

        # ---- Armijo 线搜索（L1 罚函数）----
        m0 = merit(x)
        eps_dir = 1e-6
        x_plus = np.clip(x + eps_dir * d, [b[0] for b in bounds],
                         [b[1] for b in bounds])
        x_minus = np.clip(x - eps_dir * d, [b[0] for b in bounds],
                          [b[1] for b in bounds])
        der0 = (merit(x_plus) - merit(x_minus)) / (2.0 * eps_dir)
        # 若方向不是罚函数下降方向，则改用罚函数梯度方向
        if der0 >= 0.0:
            d = -_numeric_grad(merit, x)
            x_plus = np.clip(x + eps_dir * d, [b[0] for b in bounds],
                             [b[1] for b in bounds])
            x_minus = np.clip(x - eps_dir * d, [b[0] for b in bounds],
                              [b[1] for b in bounds])
            der0 = (merit(x_plus) - merit(x_minus)) / (2.0 * eps_dir)

        t = 1.0
        accepted = False
        for _ in range(30):
            xt = np.clip(x + t * d, [b[0] for b in bounds],
                         [b[1] for b in bounds])
            if merit(xt) <= m0 + 1e-4 * t * der0:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # 极小步长仍不下降：若方向已被边界完全截断，属于正常到边；
            # 否则才提示线搜索失败。
            if np.linalg.norm(
                    np.clip(x + d, [b[0] for b in bounds],
                            [b[1] for b in bounds]) - x) < 1e-10:
                break
            warnings_collected.append("线搜索未找到下降步，提前停止")
            break

        x_new = np.clip(x + t * d, [b[0] for b in bounds],
                        [b[1] for b in bounds])

        # ---- BFGS 逆 Hessian 更新 ----
        g_new = _numeric_grad(objective, x_new)
        s = x_new - x
        y = g_new - g_obj
        sy = float(s @ y)
        if sy > 1e-8:
            rho = 1.0 / sy
            V = np.eye(n) - rho * np.outer(s, y)
            B = V @ B @ V.T + rho * np.outer(s, s)
            # 对称化与正则化，防止数值漂移
            B = (B + B.T) / 2.0
            B += 1e-10 * np.eye(n)

        x, g_obj = x_new, g_new

        f_now = objective(x)
        viol_now = float(np.sum(np.maximum(0.0, -all_c(x))))
        if viol_now < best_viol - 1e-9 or \
           (abs(viol_now - best_viol) < 1e-9 and f_now < best_f):
            best_x, best_f, best_viol = x.copy(), f_now, viol_now

    # 最终可行判定以“返回点本身”为准，避免跟踪变量与末点不同步
    feasible = bool(np.all(all_c(best_x) >= -1e-6))
    if not feasible:
        warnings_collected.append(
            "SLSQP 未收敛到可行点，尝试约束恢复阶段")
        x_rest = best_x.copy()
        for _ in range(50):
            viol_grad = np.zeros_like(x_rest)
            for ci in constraints:
                val = float(ci(x_rest))
                if val < 0.0:
                    viol_grad += -_numeric_grad(ci, x_rest)
            if np.linalg.norm(viol_grad) < 1e-12:
                break
            x_rest = np.clip(x_rest - 0.01 * viol_grad,
                             [b[0] for b in bounds], [b[1] for b in bounds])
        viol_rest = float(np.sum(np.maximum(0.0, -all_c(x_rest))))
        if viol_rest < best_viol:
            best_x, best_viol = x_rest.copy(), viol_rest
        feasible = bool(np.all(all_c(best_x) >= -1e-6))

    return {
        "x": best_x,
        "fval": float(objective(best_x)),
        "feasible": feasible,
        "iterations": it + 1,
        "warnings": warnings_collected,
    }


# ==================================================================
# 七、外层枚举：分别跑圆柱、长方体两套 GA + SLSQP
# ==================================================================
def _ga_bounds(shape: str) -> list:
    """GA 变量上下界：[nf, Hf, df]。长方体每面 nf 根，总数 4*nf。"""
    if shape == "cylinder":
        nf_max = NF_MAX
    else:
        # 每面翅根总宽 <= 边长（a_min = 1-2*HF_MAX ≈ 0.52 m）
        nf_max = min(NF_MAX, int((ENVELOPE_SIDE - 2.0 * HF_MAX) / DF_MIN))
        nf_max = max(nf_max, 1)
    return [(NF_MIN, nf_max), (HF_MIN, HF_MAX), (DF_MIN, DF_MAX)]


def _slsqp_objective(shape: str, nf: int):
    """SLSQP 目标：最小化 -min(N_theory, N_space)（连续 N 的松弛形式）。

    说明：N = floor(min(N_theory, N_space))，直接用整数台阶做目标会让
    梯度失效；SLSQP 阶段用“连续 min”作代理目标，最终报告仍按清单取整。
    """
    def obj(x):
        # 数值微分可能把试探点送到边界外，先投影回区间，
        # 避免“边界外侧不可行 -> 惩罚 1e9”污染梯度。
        Hf = min(max(float(x[0]), HF_MIN), HF_MAX)
        df = min(max(float(x[1]), DF_MIN), DF_MAX)
        raw = _raw_metrics(shape, nf, Hf, df)
        if raw is None or raw["n_theory"] <= 0.0 or raw["n_space"] <= 0.0:
            return 1e9
        return -min(raw["n_theory"], raw["n_space"])
    return obj


def _slsqp_constraints(shape: str, nf: int):
    """SLSQP 不等式约束 g(x) >= 0：
        g3 = (Hf - 3*df)/Hf                       （一维直翅假设 Hf/df >= 3）
        g2 = (根部可布周长 - nf*df)/根部可布周长    （翅根不重叠）
    空间、温度两个上限已并入目标 min(N_theory, N_space)，
    不再重复设约束，避免“热容量恒大于空间容量”导致的伪不可行。
    """
    def g3(x):
        Hf = min(max(float(x[0]), HF_MIN), HF_MAX)
        df = min(max(float(x[1]), DF_MIN), DF_MAX)
        return (Hf - 3.0 * df) / max(Hf, 1e-9)

    def g2(x):
        Hf = min(max(float(x[0]), HF_MIN), HF_MAX)
        df = min(max(float(x[1]), DF_MIN), DF_MAX)
        geo = _build_geometry(shape, Hf)
        if geo is None:
            return -1.0
        limit = geo["overlap_limit"]
        return (limit - nf * df) / max(limit, 1e-9)
    return [g3, g2]


def solve_shape(shape: str, verbose: bool = True) -> dict:
    """一套完整流程：GA -> SLSQP -> 最终评估。"""
    bounds = _ga_bounds(shape)
    fitness = make_fitness(shape)
    ga = SimpleGA(bounds, seed=GA_SEED + (0 if shape == "cylinder" else 1))
    ga_result = ga.run(fitness)

    if ga_result.best_fitness <= -1e8:
        out = {
            "shape": shape,
            "feasible": False,
            "N": 0,
            "nf": None, "Hf": None, "df": None,
            "a_eff": None, "n_theory": None, "n_space": None,
            "warnings": ["GA 未找到可行个体，跳过 SLSQP"],
        }
        if verbose:
            print("[警告] 外形 %s：GA 无可行解，返回 N=0。" % shape)
        return out

    nf_ga, Hf_ga, df_ga = (int(round(ga_result.best_chrom[0])),
                           float(ga_result.best_chrom[1]),
                           float(ga_result.best_chrom[2]))
    res_ga = evaluate_design(shape, nf_ga, Hf_ga, df_ga)

    # ---- SLSQP 局部寻优：固定 nf，微调 (Hf, df) ----
    x0 = [Hf_ga, df_ga]
    slsqp_bounds = [(HF_MIN, HF_MAX), (DF_MIN, DF_MAX)]
    obj = _slsqp_objective(shape, nf_ga)
    cons = _slsqp_constraints(shape, nf_ga)
    local = slsqp_local(obj, cons, x0, slsqp_bounds)

    Hf_loc, df_loc = float(local["x"][0]), float(local["x"][1])
    res_loc = evaluate_design(shape, nf_ga, Hf_loc, df_loc)

    if not res_loc["feasible"]:
        out = {
            "shape": shape,
            "feasible": False,
            "N": 0,
            "nf": nf_ga, "Hf": Hf_loc, "df": df_loc,
            "a_eff": None, "n_theory": None, "n_space": None,
            "warnings": res_loc["warnings"] + local["warnings"],
        }
        if verbose:
            print("[警告] 外形 %s：SLSQP 后仍无可行解。" % shape)
            for w in out["warnings"]:
                print("    -", w)
        return out

    if verbose:
        print("=" * 70)
        print("外形：%s" % ("圆柱" if shape == "cylinder" else "长方体"))
        print("=" * 70)
        print("GA 最优个体      : nf=%d, Hf=%.4f m, df=%.4f m -> N=%d"
              % (nf_ga, Hf_ga, df_ga, res_ga["N"]))
        print("SLSQP 局部寻优后 : nf=%d, Hf=%.4f m, df=%.4f m -> N=%d"
              % (nf_ga, Hf_loc, df_loc, res_loc["N"]))
        print("    h_air  = %.3f W/(m^2·K)，h_sea = %.1f W/(m^2·K)"
              % (res_loc["h_air"], res_loc["h"]))
        print("    h_total= %.4f W/(m^2·K)，T_wi=%.2f ℃，T_wo=%.2f ℃"
              % (res_loc["h_total"], res_loc["t_wi"], res_loc["t_wo"]))
        print("    eta_f  = %.4f" % res_loc["eta_f"])
        print("    A_eff  = %.2f m^2（A_base=%.2f m^2）"
              % (res_loc["a_eff"], res_loc["a_base"]))
        print("    N_theory = %.2f 台，N_space = %.2f 台"
              % (res_loc["n_theory"], res_loc["n_space"]))
        if res_loc["n_theory"] < res_loc["n_space"]:
            print("    提示：当前瓶颈为散热上限 N_theory，"
                  "外翅片收益受舱内空气侧热阻制约")
        else:
            print("    提示：当前瓶颈为空间上限 N_space（体积法上界），"
                  "翅片参数主要提供散热裕度")
        if local["warnings"]:
            print("    SLSQP 提示:")
            for w in local["warnings"]:
                print("      -", w)

    return {
        "shape": shape,
        "feasible": True,
        "N": res_loc["N"],
        "nf": nf_ga,
        "Hf": Hf_loc,
        "df": df_loc,
        "a_eff": res_loc["a_eff"],
        "n_theory": res_loc["n_theory"],
        "n_space": res_loc["n_space"],
        "warnings": local["warnings"],
    }


def write_q2_results_csv(cyl: dict, box: dict, best: dict) -> Path:
    """把圆柱 / 长方体与全局最优的数值结果写入 问题二/outputs/问题2_结果.csv。"""
    fields = ["外形", "nf", "Hf_m", "df_m", "N_台", "N_theory", "N_space",
              "A_base_m2", "A_eff_m2", "eta_f", "h_air_W_m2K", "h_sea_W_m2K",
              "h_total_W_m2K", "T_wi_C", "T_wo_C", "Q_max_W"]

    def _num(v):
        return None if v is None else round(float(v), 6)

    def _row(label: str, r: dict) -> dict:
        return {
            "外形": label,
            "nf": None if r.get("nf") is None else int(r["nf"]),
            "Hf_m": _num(r.get("Hf")),
            "df_m": _num(r.get("df")),
            "N_台": None if r.get("N") is None else int(r["N"]),
            "N_theory": _num(r.get("n_theory")),
            "N_space": _num(r.get("n_space")),
            "A_base_m2": _num(r.get("a_base")),
            "A_eff_m2": _num(r.get("a_eff")),
            "eta_f": _num(r.get("eta_f")),
            "h_air_W_m2K": _num(r.get("h_air")),
            "h_sea_W_m2K": _num(r.get("h")),
            "h_total_W_m2K": _num(r.get("h_total")),
            "T_wi_C": _num(r.get("t_wi")),
            "T_wo_C": _num(r.get("t_wo")),
            "Q_max_W": _num(r.get("q_max")),
        }

    def _full(r: dict) -> dict:
        """solve_shape 只返回部分字段，完整指标用 evaluate_design 补全。"""
        if not r.get("feasible") or r.get("h_total") is not None:
            return r
        return evaluate_design(r["shape"], r["nf"], r["Hf"], r["df"],
                               collect_warnings=False)

    rows = [_row("圆柱", _full(cyl)), _row("长方体", _full(box))]
    best_label = "全局最优（%s）" % ("圆柱" if best["shape"] == "cylinder"
                                    else "长方体")
    rows.append(_row(best_label, _full(best)))

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "问题2_结果.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("问题2数值结果已保存：%s" % path)
    return path


def main() -> None:
    """主流程：自检 -> 圆柱 -> 长方体 -> 全局最优。"""
    print("=" * 70)
    print("C 题 问题 2：枚举 shape + GA + SLSQP")
    print("=" * 70)
    print("题目给定：D/L=1/12 m，外形上限 1x1x12 m，q0=500 W，")
    print("          Tmax=80 ℃，Tinf=20 ℃，1U=482.6x44.45x525 mm")
    print("工程默认（清单未给数值，见代码参数区）: "
          "nf∈[%d,%d]，Hf∈[%.3f,%.3f] m，df∈[%.3f,%.3f] m"
          % (NF_MIN, NF_MAX, HF_MIN, HF_MAX, DF_MIN, DF_MAX))
    print("GA: pop=%d, gen=%d, pc=%.2f, pm=%.2f；SLSQP: 手写 BFGS+QP"
          % (GA_POP, GA_GEN, GA_PC, GA_PM))
    print()
    print("口径说明：散热按问题1热阻网络（舱内空气 -> 壁导热 -> 海侧翅片）计算；")
    print("          长方体 Churchill-Chu 特征长度取等面积当量直径；")
    print("          N_space 为体积法上界，不代表真实机架布局数。")
    print("参数来源核对：D/L/Tmax/Tinf/q0/服务器尺寸 = 题面给定；")
    print("          WALL=10 mm（问题1基准，假设值）；")
    print("          K_FIN=6061-T6 铝 167 W/(m·K)（MatWeb，MakeItFrom 为 170）；")
    print("          K_WALL=304 不锈钢 20 ℃（清洗后 CSV，缺失时默认 14.4）。")
    print()

    # ---- 0. 代码校验：同一组翅片参数，手算 A_eff 与 N ----
    self_check_ok = run_self_check()
    print()

    # ---- 1/2. 圆柱与长方体各跑一套 GA + SLSQP ----
    cyl = solve_shape("cylinder")
    print()
    box = solve_shape("cuboid")
    print()

    # ---- 3. 外层枚举：对比两个外形的 N，输出全局最优结构 ----
    print("=" * 70)
    print("全局最优对比")
    print("=" * 70)
    if not cyl["feasible"] and not box["feasible"]:
        print("[警告] 圆柱与长方体均无可行解，最终输出 N=0。")
        return

    candidates = [c for c in (cyl, box) if c["feasible"]]
    candidates.sort(key=lambda c: (c["N"], c.get("n_theory") or 0.0),
                    reverse=True)
    best = candidates[0]
    shape_name = "圆柱" if best["shape"] == "cylinder" else "长方体"
    print("最优结构：%s" % shape_name)
    print("    nf=%d，Hf=%.4f m，df=%.4f m" % (best["nf"], best["Hf"], best["df"]))
    print("    A_eff=%.2f m^2，N_theory=%.2f 台，N_space=%.2f 台"
          % (best["a_eff"], best["n_theory"], best["n_space"]))
    print("    最终装机台数 N = %d 台" % best["N"])
    if best["n_theory"] < best["n_space"]:
        print("    提示：当前瓶颈为散热上限 N_theory，"
              "外翅片收益受舱内空气侧热阻制约")
    else:
        print("    提示：当前瓶颈为空间上限 N_space（体积法上界），"
              "翅片参数主要提供散热裕度")
    if cyl["feasible"] and box["feasible"] and cyl["N"] == box["N"]:
        print("    说明：两者 N 并列，本代码以 N_theory 较高者为全局最优；"
              "如需并列展示可改比较规则。")
    print()
    write_q2_results_csv(cyl, box, best)
    plot_optimal_cross_section(best, {"cylinder": cyl, "cuboid": box})
    print("自检结果：%s" % ("PASS" if self_check_ok else "FAIL"))


# ==================================================================
# 八、手算校验：同一组翅片参数核对 A_eff 与 N
# ==================================================================
def run_self_check() -> bool:
    """评审修正后的自检：几何、翅片效率、热阻网络、问题1基准回归。

    校验算例（全部为手算可复算的数值）：
        圆柱：nf=60，Hf=0.10 m，df=0.005 m
        基体直径 D_base = 1 - 2*0.10 = 0.80 m
        问题1回归：裸圆柱 D=1（无翅片）应复现 N=15、Q≈7761 W。
    """
    print("=" * 70)
    print("代码校验：几何 + 热阻网络 + 问题1基准回归")
    print("=" * 70)

    nf, Hf, df = 60, 0.10, 0.005
    res = evaluate_design("cylinder", nf, Hf, df)
    if not res["feasible"]:
        print("自检结果：FAIL（设计不可行）")
        return False
    D_base = ENVELOPE_SIDE - 2.0 * Hf
    print("输入：圆柱 nf=%d，Hf=%.3f m，df=%.3f m，D_base=%.2f m"
          % (nf, Hf, df, D_base))
    print("几何：A_base=%.4f m^2，A_fin,单根=%.4f m^2，eta_f=%.4f，A_eff=%.4f m^2"
          % (res["a_base"], res["a_fin_one"], res["eta_f"], res["a_eff"]))
    print("网络：h_air=%.3f，h_sea=%.1f，h_total=%.4f W/(m^2·K)"
          % (res["h_air"], res["h"], res["h_total"]))
    print("壁温：T_wi=%.2f ℃，T_wo=%.2f ℃；Q_max=%.1f W"
          % (res["t_wi"], res["t_wo"], res["q_max"]))
    print("容量：N_theory=%.2f 台，N_space=%.2f 台，N=%d 台"
          % (res["n_theory"], res["n_space"], res["N"]))

    # ---- 一致性断言 ----
    ok = True
    ok = ok and res["a_eff"] > 0.0 and 0.0 < res["eta_f"] <= 1.0
    # 统一口径：Q_max = h_total*A_eff*ΔT（精确）；再与网络 ΔT/R_total 交叉验证
    dT = T_MAX - T_INF
    ok = ok and abs(res["q_max"] - res["h_total"] * res["a_eff"]
                    * dT) < 1e-9
    ok = ok and abs(res["q_max"] - dT / res["r_total"]) < 1e-3
    ok = ok and res["N"] == int(np.floor(min(res["n_theory"], res["n_space"])))
    ok = ok and 1e-5 <= res["ra_outer"] <= 1e12

    # ---- 问题1基准回归：裸圆柱 D=1（无翅片）应回到 N=15 ----
    geo = _build_geometry("cylinder", 0.0)
    a_out = geo["base_side_area"] + geo["base_end_area"]
    net = _solve_thermal_network("cylinder", geo, a_out, _k_wall())
    q1_max = net["h_total"] * a_out * dT
    n_theory_q1 = q1_max / Q0
    n_space_q1 = net["v_inner"] / (SERVER_W * SERVER_H * SERVER_L)
    n_q1 = int(np.floor(min(n_theory_q1, n_space_q1)))
    print("问题1基准回归（裸圆柱 D=1，无翅片）：Q_max=%.1f W，h_sea=%.1f，"
          "N_theory=%.2f，N=%d 台（基准 N=15）"
          % (q1_max, net["h_sea"], n_theory_q1, n_q1))
    ok = ok and abs(q1_max - 7761.5) / 7761.5 < 0.05
    ok = ok and abs(q1_max - net["q_total"]) / q1_max < 1e-6
    ok = ok and n_q1 == 15
    print("自检结果：%s" % ("PASS" if ok else "FAIL"))
    if not ok:
        warnings.warn("自检未通过，请检查公式或数据。")
    return ok


# ==================================================================
# 九、无可行解鲁棒性演示（不抛异常，只给警告）
# ==================================================================
def demo_warning_path() -> bool:
    """鲁棒性演示：非法输入应返回 feasible=False 与中文警告，不抛异常。

    覆盖 4 类情况：
        1) 声明区间内翅根重叠（nf*df > 可布置周长）
        2) 一维翅片假设不满足（Hf/df < 3）
        3) 翅片数超出声明区间（nf > NF_MAX）
        4) 未知外形
    """
    print("鲁棒性演示：非法输入均返回 feasible=False 与中文警告，不抛异常")
    cases = [
        ("翅根重叠（nf*df > 可布置周长）", "cuboid", 160, 0.24, 0.01),
        ("一维翅片假设不满足（Hf/df < 3）", "cylinder", 60, 0.005, 0.01),
        ("翅片数超出声明区间（nf > NF_MAX）", "cylinder", 200, 0.10, 0.005),
        ("未知外形（仅支持 cylinder/cuboid）", "sphere", 60, 0.10, 0.005),
    ]
    ok = True
    for name, shape, nf, Hf, df in cases:
        res = evaluate_design(shape, nf, Hf, df)
        valid = (not res["feasible"]) and len(res["warnings"]) > 0
        ok = ok and valid
        print("  [%s] %s" % ("PASS" if valid else "FAIL", name))
        print("    输入：shape=%s，nf=%d，Hf=%.3f m，df=%.4f m"
              % (shape, nf, Hf, df))
        print("    输出：feasible=%s，N=%d" % (res["feasible"], res["N"]))
        for w in res["warnings"]:
            print("    警告：", w)
    print("鲁棒性演示：%s" % ("全部 PASS" if ok else "存在 FAIL"))
    return ok


if __name__ == "__main__":
    main()
    print()
    demo_warning_path()
