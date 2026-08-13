# -*- coding: utf-8 -*-
"""
C 题 问题 4：潮汐与季节影响下的动态散热 + 联合优化
=================================================================
方法：
    1) 季节影响：WOA18 月均表层海温（清洗后数据）+ 余弦拟合逐时化，
       按月计算海温温差 dT 与海水物性（MIT 35 g/kg 清洗表插值），
       输出每月散热能力与可放服务器数；
    2) 潮汐影响：HKO 赤鱲角东 2026 天文潮逐时预报（清洗后数据），
       水位变化改变浸没深度；潮汐+月均海流（GODAS 2021 清洗表）引入
       强制对流，水平圆柱用 Churchill-Bernstein 关联式；
    3) 混合对流判据 Gr/Re^2：>10 自然对流主导、<0.1 强制对流主导、
       其余为混合对流；逐时输出 h_nat、h_forced、h_mixed 与散热量；
    4) FFT + 13 个标准分潮调和分析识别 M2/S2/K1/O1 等主分潮，
       量化大小潮周期、潮位极值与逐日潮差包络；
    5) RK4 四阶龙格库塔解集总参数动态方程，二分搜索最大服务器数 N；
    6) 联合优化：NSGA-II 嵌套 RK4（每个个体内部用二分+RK4 求最大 N
       与全年温度约束），输出帕累托前沿与折衷解；
    7) 每个模型结束做灵敏度分析；全部中间量打印单位与物理含义；
    8) 明确说明：所用潮位为天文潮预报，不含风暴潮与余水位。

依赖：numpy / pandas / matplotlib（bundled runtime 已具备）。
运行：python C题_问题4_RK4_NSGA2.py
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

# 时变传热模型六项检验：能量残差、边界合理性、关联式适用域、极限情景、
# 参数敏感性与周期稳定性（结果见 输出/结果_时变模型检验.csv 与 图8）。

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "输出"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ==================================================================
# 0. 中文绘图字体与工作区自动定位
# ==================================================================
def setup_chinese_font() -> str:
    """选择系统可用中文字体，避免 matplotlib 中文显示为方框。"""
    candidates = ["Microsoft YaHei", "SimHei", "KaiTi",
                  "Arial Unicode MS", "Noto Sans CJK SC", "DengXian"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("[警告] 未找到中文字体，图中中文可能显示为方框。")
    return "DejaVu Sans"


def find_workspace_root() -> Path:
    """自动向上查找工作区根目录（含 C题数据/清洗后数据）。"""
    here = Path(__file__).resolve().parent
    for root in (here, *here.parents):
        if (root / "C题数据" / "清洗后数据").is_dir():
            return root
    raise FileNotFoundError("自动查找失败：未找到 C题数据/清洗后数据 目录。")


WORKSPACE = find_workspace_root()
CLEAN_DIR = WORKSPACE / "C题数据" / "清洗后数据"


# ==================================================================
# 1. 题目给定参数（来源：题目 C题.pdf；附件交付清单）
# ==================================================================
D_EXT = 1.0          # 圆柱外径 D，m（问题 1）
L_EXT = 12.0         # 圆柱长度 L，m（问题 1）
Q0 = 500.0           # 单台服务器产热，W（问题 1）
T_MAX = 80.0         # 允许最高温度，degC（问题 1）
SERVER_W, SERVER_H, SERVER_L = 0.4826, 0.04445, 0.525  # 1U 服务器，m
V_SERVER = SERVER_W * SERVER_H * SERVER_L              # 单台体积，m^3

# 工程假设（附件交付清单 3.6/3.8 已确认）：寿命与成本基准
MIN_LIFE = 10.0      # 最低使用寿命，年
LIFE_CAP = 50.0      # 寿命上限，年
COATING_PRICE = 150.0  # 防腐涂层单价，元/m^2
G = 9.81             # 重力加速度，m/s^2

# 优化变量范围（附件交付清单：DEPTH in [5,100] m，WALL in [0.004,0.05] m）
DEPTH_MIN, DEPTH_MAX = 5.0, 100.0
WALL_MIN, WALL_MAX = 0.004, 0.050

# NSGA-II 超参数（沿用问题 3 口径）
NSGA_POP = 16
NSGA_GEN = 8
NSGA_PC = 0.90
NSGA_PM = 0.10
NSGA_ETA_C = 15.0
NSGA_ETA_M = 20.0
NSGA_SEED = 20260813

# 潮汐流速幅值：清洗后数据无逐时海流，属于显式工程假设（灵敏度扫描覆盖）
U_TIDE_AMP_BASE = 0.15   # m/s，潮汐流速峰值（模型假设，论文需注明）

# 站点：题目背景为珠海高栏港；可选 陵水
SITE = "珠海"


# ==================================================================
# 2. 数据读取（只从 C题数据/清洗后数据 与 附件 读取，禁止编造）
# ==================================================================
def load_woa18_monthly_sst() -> pd.DataFrame:
    """WOA18 1981-2010 月均表层海温，12 个月。"""
    f = CLEAN_DIR / "WOA18_1981-2010_月均表层温度_南海站点_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df.sort_values("月").reset_index(drop=True)
    print("=" * 76)
    print("第 1 步  读取 WOA18 月均表层海温：", f.name)
    print("=" * 76)
    print("shape:", df.shape, "| 缺失值:", int(df.isna().sum().sum()))
    print(df.head(4).to_string(index=False))
    print("量级检查：珠海月均海温范围 = %.2f ~ %.2f degC"
          % (df["珠海_表层温度_degC"].min(), df["珠海_表层温度_degC"].max()))
    print()
    return df


def load_ersst_monthly_sst() -> pd.DataFrame:
    """ERSST v5 2020-2021 月均 SST，24 个月，用于季节模型校核。"""
    f = CLEAN_DIR / "ERSST_v5_2020-2021_南海站点SST_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    print("第 1 步  读取 ERSST 2020-2021 月均 SST：", f.name,
          "| shape:", df.shape, "| 缺失值:", int(df.isna().sum().sum()))
    print("量级检查：珠海 ERSST 范围 = %.2f ~ %.2f degC"
          % (df["珠海_SST_degC"].min(), df["珠海_SST_degC"].max()))
    print()
    return df


def load_tide_2026() -> pd.DataFrame:
    """HKO 赤鱲角东 2026 天文潮逐时预报，8760 点（清洗后数据）。"""
    f = CLEAN_DIR / "HKO_ChekLapKokE_2026_hourly_tide_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print("第 2 步  读取 2026 天文潮逐时预报：", f.name)
    print("=" * 76)
    print("shape:", df.shape, "| 时间范围:", df["datetime"].min(),
          "->", df["datetime"].max())
    print("潮位范围: %.3f ~ %.3f m | 均值: %.4f m"
          % (df["tide_height_m"].min(), df["tide_height_m"].max(),
             df["tide_height_m"].mean()))
    print("[重要说明] 本数据为天文潮预报，不含风暴潮与余水位；"
          "珠海/陵水无本地潮位站，采用香港赤鱲角东代理（附件 3.8）。")
    print()
    return df


def load_seawater_props_table() -> pd.DataFrame:
    """MIT 海水物性表（35 g/kg，0-40 degC，清洗后数据）。"""
    f = CLEAN_DIR / "海水热物性_MIT_35gkg_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df.sort_values("温度_degC").reset_index(drop=True)
    print("第 3 步  读取海水热物性表（MIT, S=35 g/kg）：", f.name,
          "| shape:", df.shape)
    print("温度网格:", df["温度_degC"].tolist())
    print("20 degC 校验：rho=%.2f kg/m^3, k=%.4f W/(m.K), "
          "nu=%.3e m^2/s, Pr=%.3f"
          % (df.loc[df["温度_degC"] == 20, "密度_kg_m3"].iloc[0],
             df.loc[df["温度_degC"] == 20, "导热系数_W_mK"].iloc[0],
             df.loc[df["温度_degC"] == 20, "运动粘度_m2_s"].iloc[0],
             df.loc[df["温度_degC"] == 20, "普朗特数"].iloc[0]))
    print()
    return df


def load_metal_cp() -> dict[str, float]:
    """金属比热容表（EngineeringToolbox 清洗表），单位 J/(kg.K)。"""
    f = CLEAN_DIR / "金属比热容_EngineeringToolbox_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    mapping = {
        "6061 铝合金": "Aluminum",
        "304 不锈钢": "Carbon Steel",     # 附件注明不锈钢按碳钢近似
        "316L 不锈钢": "Carbon Steel",
        "TC4 钛合金": "Titanium",
        "AISI 1040 碳钢": "Carbon Steel",
        "T2 紫铜": "Copper",
    }
    out = {}
    for mat, eng in mapping.items():
        row = df[df["材料"].astype(str).str.strip() == eng]
        if row.empty:
            raise ValueError("比热容表中未找到 %s" % eng)
        out[mat] = float(row["比热容_J_kgK"].iloc[0])
    print("第 3 步  金属比热容（J/(kg.K)）：", out)
    print()
    return out


def load_godas_currents() -> pd.DataFrame:
    """GODAS 2021 月均海流（珠海/陵水，各深度层），清洗后数据。"""
    f = CLEAN_DIR / "GODAS_2021_南海站点海流_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    print("第 4 步  读取 GODAS 2021 月均海流：", f.name,
          "| shape:", df.shape, "| 缺失值:", int(df.isna().sum().sum()))
    sub = df[df["站点"] == SITE]
    print("站点 %s 月均流速范围: %.4f ~ %.4f m/s"
          % (SITE, sub["流速_m_s"].min(), sub["流速_m_s"].max()))
    print("[说明] 数据包无 2026 逐时海流，采用 GODAS 2021 月均作为背景流；"
          "潮汐流速为显式工程假设。")
    print()
    return df


def load_material_table() -> pd.DataFrame:
    """候选材料表：密度/屈服/抗拉强度来自附件清洗表，导热系数来自
    金属导热系数清洗表；价格与腐蚀速率为工程默认值（用户已确认）。"""
    f_attach = CLEAN_DIR / "海洋材料性能_C题附件_clean.csv"
    f_k = CLEAN_DIR / "金属导热系数_EngineeringToolbox_clean.csv"
    dfm = pd.read_csv(f_attach, encoding="utf-8-sig")
    dfk = pd.read_csv(f_k, encoding="utf-8-sig")
    dfk["material"] = dfk["material"].astype(str).str.strip().str.lower()

    def mech(key: str) -> tuple[float, float, float]:
        hit = dfm[dfm["材料"].astype(str).str.strip() == key]
        if hit.empty:
            raise ValueError("附件材料表未找到牌号 %s" % key)
        row = hit.iloc[0]
        rho_lb = float(row["密度"])
        rho = rho_lb * 27679.9047  # lb/in^3 -> kg/m^3
        yld = float(str(row["屈服强度_ksi"]).split()[0]) * 6.894757  # ksi -> MPa
        rmg = float(str(row["抗拉强度_ksi"]).split()[0]) * 6.894757
        return rho, yld, rmg

    def k_20c(eng_key: str | None) -> float | None:
        if eng_key is None:
            return None
        hit = dfk[(dfk["material"] == eng_key)
                  & (np.isclose(dfk["temperature_C"], 20.0))]
        if not hit.empty:
            return float(hit["thermal_conductivity_W_per_mK"].iloc[0])
        return None

    raw = [
        ("6061 铝合金", "Aluminum alloy 6061", "aluminum alloy 6061"),
        ("304 不锈钢", "304 Stainless Steel", "steel - stainless, type 304"),
        ("316L 不锈钢", "316 Stainless Steel", None),
        ("TC4 钛合金", "Titanium 6Al-4V", None),
        ("AISI 1040 碳钢", "AISI 1040 Steel", "steel - carbon, 0.5% c"),
        ("T2 紫铜", "Copper", "copper"),
    ]
    rows = []
    for name, brand, kkey in raw:
        rho, yld, rmg = mech(brand)
        k = k_20c(kkey)
        rows.append((name, brand, rho, yld, rmg, k))
    mat = pd.DataFrame(rows, columns=["材料", "牌号", "密度_kg_m3",
                                      "屈服强度_MPa", "抗拉强度_MPa",
                                      "导热系数_W_mK"])
    # 口径与交付清单一致：6061=167、304=14.4、316L=14.9、TC4=6.7
    default_k = {"6061 铝合金": 167.0, "304 不锈钢": 14.4,
                 "316L 不锈钢": 14.9, "TC4 钛合金": 6.7,
                 "AISI 1040 碳钢": 54.0, "T2 紫铜": 401.0}
    for i, r in mat.iterrows():
        if pd.isna(r["导热系数_W_mK"]):
            mat.loc[i, "导热系数_W_mK"] = default_k[r["材料"]]
            print("[警告] %s 导热系数用工程默认值 %.1f W/(m.K)（数据.md 引用值）"
                  % (r["材料"], default_k[r["材料"]]))
    price = {"6061 铝合金": 22000.0, "304 不锈钢": 15000.0,
             "316L 不锈钢": 22000.0, "TC4 钛合金": 320000.0,
             "AISI 1040 碳钢": 5200.0, "T2 紫铜": 68000.0}
    corr = {"6061 铝合金": 0.150, "304 不锈钢": 0.020,
            "316L 不锈钢": 0.010, "TC4 钛合金": 0.001,
            "AISI 1040 碳钢": 0.300, "T2 紫铜": 0.080}
    mat["价格_元_吨"] = mat["材料"].map(price)
    mat["腐蚀速率_mm_年"] = mat["材料"].map(corr)
    print("第 5 步  候选材料表（密度/强度来自附件清洗表）：")
    print(mat.to_string(index=False))
    print("[说明] 价格与腐蚀速率采用附件交付清单/问题 3 工程默认值"
          "（已确认允许），正式论文需注明为估算。")
    print()
    return mat


def fit_two_segment(depth: np.ndarray, temp: np.ndarray) -> dict:
    """T(d) = c0 + c1*d + c2*max(d-d1,0)，两段连续线性模型。"""
    candidates = np.linspace(depth.min() + 1.0, depth.max() - 1.0, 200)
    best = None
    for d1 in candidates:
        x = np.column_stack([np.ones_like(depth), depth,
                             np.maximum(depth - d1, 0.0)])
        coef, *_ = np.linalg.lstsq(x, temp, rcond=None)
        sse = float(np.sum((temp - x @ coef) ** 2))
        if best is None or sse < best["sse"]:
            best = {"d1": d1, "c0": coef[0], "c1": coef[1],
                    "c2": coef[2], "sse": sse}
    ss_tot = float(np.sum((temp - temp.mean()) ** 2))
    return {"d1": best["d1"], "T0": best["c0"],
            "k1": best["c1"], "k2": best["c1"] + best["c2"],
            "R2": 1.0 - best["sse"] / ss_tot}


def make_depth_temp_predictor(site: str) -> Callable[[np.ndarray], np.ndarray]:
    """由 WOA18 温度剖面构造深度-温度预测器（两段连续线性）。"""
    f = CLEAN_DIR / "WOA18_南海温度剖面_clean.csv"
    df = pd.read_csv(f, encoding="utf-8-sig")
    df = df[df["站点"] == site].sort_values("深度_m")
    dep = df["深度_m"].to_numpy(float)
    tmp = df["温度_degC"].to_numpy(float)
    model = fit_two_segment(dep, tmp)
    print("第 6 步  WOA18 温度剖面两段线性拟合（%s）：d1=%.2f m, "
          "T0=%.3f degC, k1=%.5f, k2=%.5f degC/m, R2=%.4f"
          % (site, model["d1"], model["T0"], model["k1"], model["k2"],
             model["R2"]))

    return DepthTempPredictor(model)


def fit_seasonal_cosine(monthly: pd.DataFrame, site: str) -> dict:
    """用 WOA18 12 个月均海温做余弦拟合：
    T_surf(t) = Tm + A*cos(2*pi*(t - t_peak)/8760)，t 为小时。"""
    t_mid = (monthly["月"].to_numpy(float) - 0.5) * 730.0  # 月中小时刻
    y = monthly["%s_表层温度_degC" % site].to_numpy(float)
    omega = 2.0 * math.pi / 8760.0
    x = np.column_stack([np.ones_like(t_mid), np.cos(omega * t_mid),
                         np.sin(omega * t_mid)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    tm = coef[0]
    a = math.hypot(coef[1], coef[2])
    t_peak = math.atan2(coef[2], coef[1]) / omega
    t_peak %= 8760.0
    pred = x @ coef
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / float(
        np.sum((y - y.mean()) ** 2))
    print("第 6 步  季节余弦拟合（%s）：Tm=%.3f degC, A=%.3f degC, "
          "峰值时刻=%.2f h, R2=%.4f"
          % (site, tm, a, t_peak, r2))
    print("  与附件交付清单对比：珠海 Tm=25.118, A=4.540（R2=0.980）；"
          "陵水 Tm=26.674, A=2.865（R2=0.939）")
    return {"Tm": tm, "A": a, "t_peak": t_peak, "omega": omega, "R2": r2}


def validate_season_with_ersst(ersst: pd.DataFrame, sst_model: dict,
                               site: str = SITE) -> pd.DataFrame:
    """用 ERSST 2020-2021 月均 SST 校核 WOA18 季节余弦模型。"""
    rows = []
    preds: list[float] = []
    obs: list[float] = []
    for _, r in ersst.iterrows():
        month = int(str(r["month"]).strip()[-2:])
        t_mid = (month - 0.5) * 730.0
        pred = sst_model["Tm"] + sst_model["A"] * math.cos(
            sst_model["omega"] * (t_mid - sst_model["t_peak"]))
        observed = float(r["%s_SST_degC" % site])
        rows.append({"月份": month, "ERSST_degC": observed,
                     "WOA18模型_degC": pred, "残差_degC": observed - pred})
        preds.append(pred)
        obs.append(observed)
    out = pd.DataFrame(rows)
    pred_arr = np.asarray(preds, dtype=float)
    obs_arr = np.asarray(obs, dtype=float)
    rmse = float(np.sqrt(np.mean((obs_arr - pred_arr) ** 2)))
    bias = float(np.mean(obs_arr - pred_arr))
    r2 = 1.0 - float(np.sum((obs_arr - pred_arr) ** 2)
                     / np.sum((obs_arr - obs_arr.mean()) ** 2))
    print("=" * 76)
    print("第 6b 步  ERSST 2020-2021 校核 WOA18 季节余弦模型")
    print("=" * 76)
    print("RMSE=%.3f degC | 平均偏差=%.3f degC | R2=%.3f"
          % (rmse, bias, r2))
    print(out.round(3).to_string(index=False))
    print()
    return out


# ==================================================================
# 3. 海水物性与换热关联式（全部 SI 单位）
# ==================================================================
class SeaPropsInterpolator:
    """海水物性插值器：rho/cp/k/nu/Pr/beta，beta 由密度表差分求得。
    写成模块级类（而非闭包）以便 Windows 多进程池可序列化。"""

    def __init__(self, table: pd.DataFrame):
        self.T = table["温度_degC"].to_numpy(float)
        self.rho = table["密度_kg_m3"].to_numpy(float)
        self.cp = table["比热容_J_kgK"].to_numpy(float)
        self.k = table["导热系数_W_mK"].to_numpy(float)
        self.nu = table["运动粘度_m2_s"].to_numpy(float)
        self.pr = table["普朗特数"].to_numpy(float)
        self.beta = np.gradient(-np.log(np.maximum(self.rho, 1.0)), self.T)

    def at(self, tt: float) -> tuple[float, float, float, float, float, float]:
        t = float(np.clip(tt, self.T[0], self.T[-1]))
        return (float(np.interp(t, self.T, self.rho)),
                float(np.interp(t, self.T, self.cp)),
                float(np.interp(t, self.T, self.k)),
                float(np.interp(t, self.T, self.nu)),
                float(np.interp(t, self.T, self.pr)),
                float(np.interp(t, self.T, self.beta)))

    def arrays(self, tarr: np.ndarray) -> dict[str, np.ndarray]:
        t = np.clip(tarr, self.T[0], self.T[-1])
        return {"rho": np.interp(t, self.T, self.rho),
                "cp": np.interp(t, self.T, self.cp),
                "k": np.interp(t, self.T, self.k),
                "nu": np.interp(t, self.T, self.nu),
                "Pr": np.interp(t, self.T, self.pr),
                "beta": np.interp(t, self.T, self.beta)}

    def __call__(self, tt: float) -> tuple[float, float, float, float, float, float]:
        """兼容旧接口：props_at(T) 直接返回物性元组。"""
        return self.at(tt)


class DepthTempPredictor:
    """WOA18 温度剖面两段连续线性预测器（模块级类，可序列化）。"""

    def __init__(self, model: dict):
        self.model = model

    def __call__(self, depth) -> np.ndarray:
        d = np.asarray(depth, dtype=float)
        m = self.model
        d1 = m["d1"]
        return np.where(d <= d1,
                        m["T0"] + m["k1"] * d,
                        m["T0"] + m["k1"] * d1 + m["k2"] * (d - d1))


class MonthlyCurrentModel:
    """GODAS 月均背景流模型：按小时所在月份取月均流速（可序列化）。"""

    def __init__(self, months_arr: np.ndarray, depths_arr: np.ndarray,
                 speeds_arr: np.ndarray, months_hourly: np.ndarray):
        self.months_arr = months_arr
        self.depths_arr = depths_arr
        self.speeds_arr = speeds_arr
        self.months_hourly = months_hourly

    def __call__(self, depth: float) -> np.ndarray:
        month_speeds = np.empty(12)
        for m in range(1, 13):
            mask = self.months_arr == m
            d = self.depths_arr[mask]
            s = self.speeds_arr[mask]
            if len(d) == 0:
                month_speeds[m - 1] = 0.0
                continue
            order = np.argsort(d)
            d, s = d[order], s[order]
            month_speeds[m - 1] = float(np.interp(depth, d, s))
        return month_speeds[self.months_hourly - 1]


def build_h_nat_grid(props_at) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预计算海水自然对流 h_nat(T_inf, dT) 二维表，供 RK4 快速插值。
    表头：T_inf 0-35 degC（步长 1），dT 0-30 K（步长 0.5）；
    覆盖全年海温与壁温温差范围，避免逐时重复计算 Gr/Ra/Nu。"""
    t_g = np.arange(0.0, 36.0, 1.0)
    dt_g = np.arange(0.0, 30.5, 0.5)
    table = np.empty((len(t_g), len(dt_g)))
    for i, t_inf in enumerate(t_g):
        for j, dT in enumerate(dt_g):
            t_film = t_inf + max(dT, 1e-6) / 2.0
            table[i, j] = h_natural_cylinder(
                D_EXT, max(dT, 1e-6), t_film, props_at(t_film))
    print("第 6 步  h_nat 查表预计算：T_inf %d 点 x dT %d 点，"
          "h 范围 %.1f ~ %.1f W/(m^2.K)"
          % (len(t_g), len(dt_g), float(table.min()), float(table.max())))
    return t_g, dt_g, table


def h_nat_lookup(t_inf: float, dT: float,
                 t_g: np.ndarray, dt_g: np.ndarray,
                 table: np.ndarray) -> float:
    """h_nat 二维双线性插值（纯 Python，快于逐次计算 Gr/Ra/Nu）。"""
    t0, dt0 = float(t_g[0]), float(dt_g[0])
    i = int((max(t_inf, t0) - t0) / (t_g[1] - t_g[0]))
    j = int((max(dT, dt0) - dt0) / (dt_g[1] - dt_g[0]))
    i = min(max(i, 0), len(t_g) - 2)
    j = min(max(j, 0), len(dt_g) - 2)
    fx = (t_inf - t_g[i]) / (t_g[i + 1] - t_g[i])
    fy = (dT - dt_g[j]) / (dt_g[j + 1] - dt_g[j])
    fx = min(max(fx, 0.0), 1.0)
    fy = min(max(fy, 0.0), 1.0)
    a = table[i, j]
    b = table[i + 1, j]
    c = table[i, j + 1]
    d = table[i + 1, j + 1]
    return float(a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
                 + c * (1 - fx) * fy + d * fx * fy)


def sea_props_interpolator(table: pd.DataFrame) -> SeaPropsInterpolator:
    """返回可调用的海水物性插值器（兼容旧接口 props_at(T)）。"""
    return SeaPropsInterpolator(table)


def air_props(T: float) -> dict[str, float]:
    """常压空气热物性近似（沿用问题 1 核心算法公式，T 为 degC）。"""
    tk = T + 273.15
    return {"rho": 101325.0 / (287.06 * tk), "cp": 1006.0,
            "k": 0.02439 + 0.0000792 * T,
            "mu": 1.72e-5 + 5.0e-8 * T, "beta": 1.0 / tk}


def h_natural_cylinder(D: float, dT: float, T_film: float,
                       p: tuple) -> float:
    """水平圆柱自然对流，Churchill-Chu 关联式，W/(m^2.K)。
    p = (rho, cp, k, nu, Pr, beta)。"""
    rho, cp, k, nu, pr, beta = p
    dT = max(dT, 1e-6)
    gr = G * beta * dT * D ** 3 / (nu ** 2)
    ra = gr * pr
    denom = (1.0 + (0.559 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    nu_corr = (0.60 + 0.387 * ra ** (1.0 / 6.0) / denom) ** 2
    return nu_corr * k / D


def h_forced_cylinder(D: float, U: float, p: tuple) -> float:
    """水平圆柱强制对流，Churchill-Bernstein 关联式，W/(m^2.K)。
    Nu = 0.3 + 0.62 Re^0.5 Pr^(1/3)/[1+(0.4/Pr)^(2/3)]^0.25
         * [1 + (Re/282000)^(5/8)]^(4/5)"""
    rho, cp, k, nu, pr, beta = p
    if U <= 1e-9:
        return 0.0
    re = U * D / nu
    base = (0.62 * re ** 0.5 * pr ** (1.0 / 3.0)
            / (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25)
    fin = (1.0 + (re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return (0.3 + base * fin) * k / D


def classify_mixed_conv(gr: float, re: float) -> tuple[str, float]:
    """Gr/Re^2 混合对流判据。返回 (流态, 比例)。"""
    if re <= 1e-12:
        return "自然对流主导", float("inf")
    ratio = gr / (re * re)
    if ratio >= 10.0:
        return "自然对流主导", ratio
    if ratio <= 0.1:
        return "强制对流主导", ratio
    return "混合对流", ratio


# ==================================================================
# 4. 几何、壳体热容与稳态网络
# ==================================================================
def geometry(wall: float, k_wall: float) -> dict[str, float]:
    """圆柱壳几何：内外径、内外长、面积、体积、壁体积。"""
    d_in = D_EXT - 2.0 * wall
    l_in = L_EXT - 2.0 * wall
    if d_in <= 0.0 or l_in <= 0.0:
        raise ValueError("壁厚过大，内腔不存在：wall=%.4f m" % wall)
    a_out = np.pi * D_EXT * L_EXT + 2.0 * np.pi * D_EXT ** 2 / 4.0
    a_in = np.pi * d_in * l_in + 2.0 * np.pi * d_in ** 2 / 4.0
    v_inner = np.pi * d_in ** 2 / 4.0 * l_in
    v_shell = np.pi * D_EXT ** 2 / 4.0 * L_EXT - v_inner
    r_wall = (np.log(D_EXT / d_in) / (2.0 * np.pi * k_wall * float(L_EXT))
              + 2.0 * wall / (k_wall * np.pi * d_in ** 2 / 4.0))
    return {"D_in": d_in, "L_in": l_in, "A_out": a_out, "A_in": a_in,
            "V_inner": v_inner, "V_shell": v_shell, "r_wall": r_wall}


def make_cfg(mat_idx: int, depth: float, wall: float,
             material_df: pd.DataFrame, cp_map: dict[str, float]) -> dict:
    """构造一个设计方案 cfg（几何+热容+压力/成本/寿命）。"""
    mat = material_df.iloc[int(mat_idx)]
    rho = float(mat["密度_kg_m3"])
    k_mat = float(mat["导热系数_W_mK"])
    sigma_y = float(mat["屈服强度_MPa"]) * 1e6
    sigma_u = float(mat["抗拉强度_MPa"]) * 1e6
    price = float(mat["价格_元_吨"])
    corr = float(mat["腐蚀速率_mm_年"])
    cp_shell = cp_map[mat["材料"]]

    geom = geometry(wall, k_mat)
    m_shell = rho * geom["V_shell"]
    mcp = m_shell * cp_shell

    # 允许应力按 GB/T 150.1-2024：min(Rm/2.7, ReL/1.5)
    sigma_allow = min(sigma_u / 2.7, sigma_y / 1.5)
    t_req = lambda p_hydro: p_hydro * D_EXT / (2.0 * sigma_allow)

    cost = (m_shell * price / 1000.0
            + geom["A_out"] * COATING_PRICE)
    return {"mat_idx": int(mat_idx), "材料": mat["材料"], "depth": depth,
            "wall": wall, "k_wall": k_mat, "rho": rho, "cp_shell": cp_shell,
            "mcp": mcp, "m_shell": m_shell, "geom": geom,
            "sigma_y": sigma_y, "sigma_u": sigma_u,
            "sigma_allow": sigma_allow, "price": price, "corr": corr,
            "t_req_fn": t_req, "cost": cost}


def calc_life(cfg: dict, p_hydro_max: float) -> tuple[float, float, float]:
    """寿命 = 腐蚀余量/腐蚀速率；返回 (life, t_req, corr_allow_mm)。"""
    t_req = cfg["t_req_fn"](p_hydro_max)
    corr_allow_mm = (cfg["wall"] - t_req) * 1000.0
    life = min(LIFE_CAP, corr_allow_mm / cfg["corr"]) if corr_allow_mm > 0 else 0.0
    return life, t_req, corr_allow_mm


def steady_capacity(cfg: dict, t_inf: float, u_mean: float = 0.0,
                    props_at=None, props_arrays=None, max_iter: int = 5,
                    tol: float = 1e-4) -> dict:
    """稳态散热能力：Q = h_total*A_out*(T_max - T_inf)。
    内部空气与海水换热系数通过壁温迭代求得，供月度/基准对比使用。"""
    geom = cfg["geom"]
    k_wall = cfg["k_wall"]
    t_wi, t_wo = T_MAX - 10.0, t_inf + 5.0
    for _ in range(max_iter):
        h_air = h_natural_cylinder(geom["D_in"], T_MAX - t_wi,
                                   (T_MAX + t_wi) / 2.0,
                                   (1.225, 1006.0, 0.0260, 1.5e-5, 0.71,
                                    1.0 / (300.0)))
        r_air = 1.0 / (h_air * geom["A_in"])
        h_sea_n = 0.0
        for _ in range(2):
            dT = t_wo - t_inf
            p = props_at((t_wo + t_inf) / 2.0)
            h_n = h_natural_cylinder(D_EXT, dT, (t_wo + t_inf) / 2.0, p)
            h_f = h_forced_cylinder(D_EXT, u_mean, props_at(t_inf))
            h_sea = (h_n ** 3 + h_f ** 3) ** (1.0 / 3.0)
            r_sea = 1.0 / (h_sea * geom["A_out"])
            r_tot = r_air + geom["r_wall"] + r_sea
            q = (T_MAX - t_inf) / r_tot
            t_wi_new = T_MAX - q * r_air
            t_wo_new = t_inf + q * r_sea
            if abs(t_wi_new - t_wi) < tol and abs(t_wo_new - t_wo) < tol:
                t_wi, t_wo = t_wi_new, t_wo_new
                break
            t_wi, t_wo = t_wi_new, t_wo_new
        h_sea_n = h_sea
        r_tot = r_air + geom["r_wall"] + r_sea
        q = (T_MAX - t_inf) / r_tot
        h_total = 1.0 / (r_tot * geom["A_out"])
    n_space = math.floor(geom["V_inner"] / V_SERVER)
    return {"h_air": h_air, "h_sea": h_sea_n, "h_total": h_total,
            "Q": q, "N": min(math.floor(q / Q0), n_space) if q >= Q0 else 0,
            "N_space": n_space, "T_wi": t_wi, "T_wo": t_wo}


# ==================================================================
# 5. 环境构建：季节 + 潮汐水位 + 背景流 + 潮汐流
# ==================================================================
def build_environment(ctx: dict, depth: float) -> dict:
    """构建全年逐时环境：T_inf(t)、U(t)、h_forced(t)。
    ctx 由 load_context 生成，包含潮位/月均海流/季节余弦/剖面等。"""
    tide = ctx["tide"]
    tide_mean = ctx["tide_mean"]
    t_hours = ctx["t_hours"]
    d_eff = depth + (tide - tide_mean)          # 浸没深度随潮位变化，m
    t_depth = ctx["T_depth_fn"](d_eff)          # 深度处年均温度，degC
    t_inf = t_depth + ctx["season_anom"]        # + 季节异常，degC

    u_base = ctx["U_base_by_month_depth"](depth)  # GODAS 月均背景流，m/s
    u_tide = ctx["U_TIDE_AMP"] * ctx["tide_deriv_norm"]  # 潮汐流，m/s
    u_hour = u_base + u_tide

    props = ctx["props_at"].arrays(t_inf)
    re = u_hour * D_EXT / props["nu"]
    term = (0.62 * np.sqrt(re) * props["Pr"] ** (1.0 / 3.0)
            / (1.0 + (0.4 / props["Pr"]) ** (2.0 / 3.0)) ** 0.25)
    h_forced = np.where(
        u_hour > 1e-9,
        (0.3 + term * (1.0 + (re / 282000.0) ** 0.625) ** 0.8)
        * props["k"] / D_EXT, 0.0)

    env = {"t_hours": t_hours, "T_inf": t_inf, "U": u_hour,
           "h_forced": h_forced, "d_eff": d_eff, "props": props,
           "hn_grid": ctx["hn_grid"]}
    return env


def load_context(woa18: pd.DataFrame, tide_df: pd.DataFrame,
                 currents_df: pd.DataFrame, seawater_df: pd.DataFrame,
                 site: str = SITE, u_tide_amp: float = U_TIDE_AMP_BASE) -> dict:
    """预加载全年公共环境量，供基准算例与 NSGA-II 复用。"""
    t = tide_df["datetime"]
    t0 = pd.Timestamp("2026-01-01 01:00:00")   # 数据第一行时刻
    t_hours = np.arange(len(tide_df), dtype=float)
    tide = tide_df["tide_height_m"].to_numpy(float)
    tide_mean = float(tide.mean())
    months = t.dt.month.to_numpy(int)

    props_at = sea_props_interpolator(seawater_df)
    T_depth_fn = make_depth_temp_predictor(site)
    sst_model = fit_seasonal_cosine(woa18, site)
    season_anom = sst_model["A"] * np.cos(
        sst_model["omega"] * (t_hours - sst_model["t_peak"]))

    # GODAS 月均背景流：按月份插值、按深度插值
    godas = currents_df[currents_df["站点"] == site].dropna(
        subset=["流速_m_s"]).copy()
    # GODAS 清洗表 month 列为 "2021-01" 字符串，取末尾月份数
    months_arr = np.array([int(str(x).strip()[-2:]) for x in godas["month"]])
    depths_arr = godas["深度_m"].to_numpy(float)
    speeds_arr = godas["流速_m_s"].to_numpy(float)

    deriv = np.gradient(tide, t_hours)          # 潮位变化率，m/h
    max_deriv = float(np.max(np.abs(deriv)))
    tide_deriv_norm = np.abs(deriv) / max_deriv if max_deriv > 0 else np.zeros_like(tide)

    ctx = {
        "site": site, "t_hours": t_hours, "tide": tide,
        "tide_mean": tide_mean, "tide_dt": t, "months": months,
        "season_anom": season_anom, "T_depth_fn": T_depth_fn,
        "props_at": props_at, "props_arrays": props_at.arrays,
        "U_TIDE_AMP": u_tide_amp,
        "U_base_by_month_depth": MonthlyCurrentModel(
            months_arr, depths_arr, speeds_arr, months),
        "tide_deriv_norm": tide_deriv_norm,
        "max_tide_deriv_m_per_h": max_deriv,
        "sst_model": sst_model,
    }
    ctx["hn_grid"] = build_h_nat_grid(props_at)
    return ctx


# ==================================================================
# 6. RK4 动态方程 + 二分搜索最大 N
# ==================================================================
def _conductance(cfg: dict, T: float, hourf: float, env: dict,
                 props_at) -> dict:
    """逐时混合对流换热与散热量求解（准稳态壁温 2 次迭代）。
    返回 Q、h_air、h_nat、h_forced、h_mixed、Gr/Re^2、流态、T_wo。"""
    n = len(env["T_inf"])
    if hourf >= n - 1:
        j = n - 2
        f = 1.0
    else:
        j = int(hourf)
        f = hourf - j
    t_inf = env["T_inf"][j] * (1.0 - f) + env["T_inf"][j + 1] * f
    hf = env["h_forced"][j] * (1.0 - f) + env["h_forced"][j + 1] * f
    u = env["U"][j] * (1.0 - f) + env["U"][j + 1] * f
    geom = cfg["geom"]
    t_g, dt_g, hn_table = env["hn_grid"]

    def sea_h_nat(t_inf_v: float, dT_v: float, t_wo_v: float) -> float:
        # 超出查表范围时直接计算，避免静默截断 dT>30 K 或海温>35 degC
        if t_inf_v < t_g[0] or t_inf_v > t_g[-1] or dT_v > dt_g[-1]:
            t_film_v = (t_wo_v + t_inf_v) / 2.0
            return h_natural_cylinder(D_EXT, dT_v, t_film_v,
                                      props_at(t_film_v))
        return h_nat_lookup(t_inf_v, dT_v, t_g, dt_g, hn_table)

    t_wo = t_inf + 0.5 * (T - t_inf)
    t_wi = T - 0.85 * (T - t_inf)
    h_air = 0.0
    for _ in range(2):
        h_air = h_natural_cylinder(geom["D_in"], max(T - t_wi, 1e-6),
                                   (T + t_wi) / 2.0,
                                   (1.225, 1006.0, 0.0260, 1.5e-5,
                                    0.71, 1.0 / 300.0))
        r_air = 1.0 / (h_air * geom["A_in"])
        for _ in range(1):
            dT_wo = max(t_wo - t_inf, 1e-6)
            hn = sea_h_nat(t_inf, dT_wo, t_wo)
            hm = (hn ** 3 + hf ** 3) ** (1.0 / 3.0)
            r_sea = 1.0 / (hm * geom["A_out"])
            r_tot = r_air + geom["r_wall"] + r_sea
            q = (T - t_inf) / r_tot
            t_wo = t_inf + q * r_sea
            t_wi = T - q * r_air
    p_final = props_at((t_wo + t_inf) / 2.0)
    hn = sea_h_nat(t_inf, max(t_wo - t_inf, 1e-6), t_wo)
    hm = (hn ** 3 + hf ** 3) ** (1.0 / 3.0)
    re = u * D_EXT / p_final[3]
    gr = G * p_final[5] * max(t_wo - t_inf, 1e-6) * D_EXT ** 3 / p_final[3] ** 2
    regime, ratio = classify_mixed_conv(gr, re)
    return {"Q": q, "h_air": h_air, "h_nat": hn, "h_forced": hf,
            "h_mixed": hm, "Q_cap": hm * geom["A_out"] * (T_MAX - t_inf),
            "Gr_Re2": ratio, "regime": regime,
            "T_inf": t_inf, "T_wo": t_wo, "U": u}


def simulate_year(cfg: dict, N: int, env: dict, props_at,
                  dt: float = 3600.0, warmup_h: int = 168,
                  progress_label: str = "", T0: float | None = None) -> dict:
    """全年 RK4 求解：
    m_shell*cp_shell*dT/dt = N*Q0 - h(t)*A_out*(T - T_inf(t))
    T0 可指定初温；不指定时沿用模型默认初值。
    丢弃前 warmup_h 小时（初值瞬态），其余用于温度/散热统计。"""
    n = len(env["T_inf"])
    mcp = cfg["mcp"]
    if T0 is None:
        t_inf0 = float(env["T_inf"][0])
        T_guess = t_inf0 + N * Q0 / (3.0 * cfg["geom"]["A_out"])
        # 初温不高于设计上限，避免高 N 时 T0 直接越界导致误判不可行
        T = float(min(max(T_guess, t_inf0), T_MAX))
    else:
        T = float(T0)
    T_arr = np.empty(n)
    Q_arr = np.empty(n)
    Q_eff_arr = np.empty(n)
    h_mix_arr = np.empty(n)
    h_nat_arr = np.empty(n)
    h_forced_arr = np.empty(n)
    q_cap_arr = np.empty(n)
    gr_re2_arr = np.empty(n)
    t_wo_arr = np.empty(n)
    regime_cnt = {"自然对流主导": 0, "强制对流主导": 0, "混合对流": 0}
    ok = True

    def rhs(temperature: float, hourf: float) -> tuple[float, dict]:
        c = _conductance(cfg, temperature, hourf, env, props_at)
        return (N * Q0 - c["Q"]) / mcp, c

    k1, c0 = rhs(T, 0)
    T_arr[0] = T
    Q_arr[0] = c0["Q"]
    Q_eff_arr[0] = c0["Q"]
    q_cap_arr[0] = c0["Q_cap"]
    h_mix_arr[0] = c0["h_mixed"]
    h_nat_arr[0] = c0["h_nat"]
    h_forced_arr[0] = c0["h_forced"]
    gr_re2_arr[0] = c0["Gr_Re2"]
    t_wo_arr[0] = c0["T_wo"]
    regime_cnt[c0["regime"]] += 1
    for i in range(n - 1):
        if not np.isfinite(T) or T > 250.0 or T < -50.0:
            ok = False
            break
        k1, c1 = rhs(T, i)
        k2, c2 = rhs(T + 0.5 * dt * k1, i + 0.5)
        k3, c3 = rhs(T + 0.5 * dt * k2, i + 0.5)
        k4, c4 = rhs(T + dt * k3, i + 1.0)
        T += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        T_arr[i + 1] = T
        Q_arr[i + 1] = c4["Q"]
        Q_eff_arr[i + 1] = (c1["Q"] + 2.0 * c2["Q"]
                            + 2.0 * c3["Q"] + c4["Q"]) / 6.0
        q_cap_arr[i + 1] = c4["Q_cap"]
        h_mix_arr[i + 1] = c4["h_mixed"]
        h_nat_arr[i + 1] = c4["h_nat"]
        h_forced_arr[i + 1] = c4["h_forced"]
        gr_re2_arr[i + 1] = c4["Gr_Re2"]
        t_wo_arr[i + 1] = c4["T_wo"]
        regime_cnt[c4["regime"]] += 1

    if not ok:
        return {"T_max": float("inf"), "T_mean": float("inf"),
                "Q_mean": float("inf"), "Q_min": float("inf"),
                "Q_max": float("inf"), "Q_cap_mean": float("inf"),
                "Q_cap_min": float("inf"), "Q_cap_max": float("inf"),
                "Q_fluct_pct": float("inf"), "T_arr": T_arr,
                "Q_arr": Q_arr, "Q_cap_arr": q_cap_arr,
                "Q_eff_arr": Q_eff_arr,
                "h_mix_arr": h_mix_arr, "h_nat_arr": h_nat_arr,
                "h_forced_arr": h_forced_arr, "regime_cnt": regime_cnt,
                "gr_re2_arr": gr_re2_arr, "t_wo_arr": t_wo_arr,
                "worst_idx": -1, "N": N, "ok": False}

    sl = slice(warmup_h, n)
    t_max = float(T_arr[sl].max())
    t_mean = float(T_arr[sl].mean())
    q_mean = float(Q_arr[sl].mean())
    q_min = float(Q_arr[sl].min())
    q_max = float(Q_arr[sl].max())
    worst_idx = int(np.argmax(T_arr[sl])) + warmup_h
    return {"T_max": t_max, "T_mean": t_mean, "Q_mean": q_mean,
            "Q_min": q_min, "Q_max": q_max,
            "Q_fluct_pct": (q_max - q_min) / q_mean * 100.0 if q_mean > 0 else 0.0,
            "Q_cap_mean": float(q_cap_arr[sl].mean()),
            "Q_cap_min": float(q_cap_arr[sl].min()),
            "Q_cap_max": float(q_cap_arr[sl].max()),
            "T_arr": T_arr, "Q_arr": Q_arr, "Q_cap_arr": q_cap_arr,
            "Q_eff_arr": Q_eff_arr,
            "h_mix_arr": h_mix_arr, "h_nat_arr": h_nat_arr,
            "h_forced_arr": h_forced_arr, "regime_cnt": regime_cnt,
            "gr_re2_arr": gr_re2_arr, "t_wo_arr": t_wo_arr,
            "worst_idx": worst_idx, "N": N, "ok": True}


def bisect_max_N(cfg: dict, env: dict, props_at,
                 max_iter: int = 10) -> tuple[dict, int]:
    """二分搜索最大 N：N 递增时全年最高壳温单调上升。
    返回 (最优模拟结果, N_max)。"""
    n_space = math.floor(cfg["geom"]["V_inner"] / V_SERVER)
    lo = 0
    # 用稳态公式缩小二分上界：平均海温/平均流速下的稳态 N 上浮 2 倍
    t_mean = float(np.mean(env["T_inf"][168:]))
    u_mean = float(np.mean(env["U"][168:]))
    n_est = steady_capacity(cfg, t_mean, u_mean, props_at)["N"]
    hi = min(n_space, max(8, int(math.ceil(n_est * 2.0)) + 4))
    # 乐观稳态上限：最高海温 + 最大流速下的 Q_max，超过则必然不可行
    t_worst = float(np.max(env["T_inf"][168:]))
    u_max = float(np.max(env["U"][168:]))
    q_opt = steady_capacity(cfg, t_worst, u_max, props_at)["Q"]
    cache: dict[int, dict] = {}

    def feasible(N: int) -> bool:
        if N * Q0 > q_opt * 1.001:
            return False
        if N not in cache:
            cache[N] = simulate_year(cfg, N, env, props_at)
        return cache[N]["T_max"] <= T_MAX + 1e-9

    if not feasible(0):
        return cache[0], 0
    if feasible(hi):
        # 估计值可能过低（如高导热材料），向上扩展直到不可行
        while hi < n_space and feasible(min(n_space, hi * 2)):
            hi = min(n_space, hi * 2)
        if feasible(n_space):
            return cache[n_space], n_space
    for _ in range(max_iter):
        mid = (lo + hi) // 2
        if mid == lo:
            break
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    if not feasible(lo):
        lo = 0
    return cache[lo], lo


# ==================================================================
# 7. 潮汐调和分析、大小潮识别、季节月度表
# ==================================================================
TIDAL_SPEEDS_DEG_PER_H = {
    "Q1": 13.3986609, "O1": 13.9430356, "P1": 14.9589314,
    "K1": 15.0410686, "N2": 28.4397295, "M2": 28.9841042,
    "S2": 30.0000000, "K2": 30.0821373, "M4": 57.9682084,
    "MS4": 58.9841042, "M6": 86.9523127, "Mf": 1.0980331,
    "Mm": 0.5443747,
}


def tidal_harmonic_analysis(tide_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """13 个标准分潮最小二乘调和分析 + FFT 峰值核对。"""
    t = tide_df["datetime"]
    t0 = pd.Timestamp("2026-01-01 01:00:00")
    hours = (t - t0).dt.total_seconds().to_numpy(float) / 3600.0
    h = tide_df["tide_height_m"].to_numpy(float)

    cols = [np.ones_like(h)]
    for name, speed in TIDAL_SPEEDS_DEG_PER_H.items():
        omega = math.radians(speed) * hours
        cols += [np.cos(omega), np.sin(omega)]
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, h, rcond=None)

    rows = []
    var_total = 0.0
    for i, name in enumerate(TIDAL_SPEEDS_DEG_PER_H):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        amp = math.hypot(a, b)
        phase = math.degrees(math.atan2(b, a)) % 360.0
        var = amp * amp / 2.0
        var_total += var
        rows.append({"分潮": name, "角速度_deg_h": TIDAL_SPEEDS_DEG_PER_H[name],
                     "周期_h": 360.0 / TIDAL_SPEEDS_DEG_PER_H[name],
                     "振幅_m": amp, "相位_deg": phase, "方差贡献": var})
    result = pd.DataFrame(rows).sort_values("振幅_m", ascending=False).reset_index(drop=True)
    result["方差占比_pct"] = result["方差贡献"] / var_total * 100.0

    pred = design @ coef
    resid = h - pred
    ss_tot = float(np.sum((h - h.mean()) ** 2))
    summary = {"Z0_m": float(coef[0]),
               "R2": 1.0 - float(np.sum(resid ** 2)) / ss_tot,
               "RMSE_m": float(np.sqrt(np.sum(resid ** 2) / len(h))),
               "max_abs_resid_m": float(np.abs(resid).max()),
               "n_hours": len(h)}

    # FFT 单边频谱
    ft = np.fft.rfft(h - h.mean())
    freq = np.fft.rfftfreq(len(h), d=1.0)   # 1/h
    amp_fft = np.abs(ft) * 2.0 / len(h)
    peaks = {}
    for name, speed in TIDAL_SPEEDS_DEG_PER_H.items():
        f_t = speed / 360.0
        idx = int(np.argmin(np.abs(freq - f_t)))
        peaks[name] = (float(freq[idx]), float(amp_fft[idx]))
    summary["FFT_peaks"] = peaks

    print("=" * 76)
    print("第 7 步  2026 天文潮调和分析（13 标准分潮，最小二乘）")
    print("=" * 76)
    print("Z0=%.4f m | R2=%.6f | RMSE=%.4f m | max|resid|=%.4f m"
          % (summary["Z0_m"], summary["R2"], summary["RMSE_m"],
             summary["max_abs_resid_m"]))
    print(result.head(10).to_string(index=False))
    print("FFT 峰值核对（M2/S2/K1/O1）：")
    for name in ["M2", "S2", "K1", "O1"]:
        fft_f, fft_a = peaks[name]
        print("  %s: 谱峰频率 %.6f 1/h（理论 %.6f 1/h），FFT振幅 %.4f m"
              % (name, fft_f, TIDAL_SPEEDS_DEG_PER_H[name] / 360.0, fft_a))
    print()
    return result, summary


def spring_neap_days(tide_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """逐日潮差：大潮日=潮差最大，小潮日=潮差最小。"""
    df = tide_df.copy()
    df["date"] = df["datetime"].dt.date
    df = df[df["date"] != pd.Timestamp("2027-01-01").date()]
    daily = df.groupby("date")["tide_height_m"].agg(["min", "max"])
    daily["range_m"] = daily["max"] - daily["min"]
    daily = daily.reset_index()
    spring_i = int(daily["range_m"].idxmax())
    neap_i = int(daily["range_m"].idxmin())
    spring = daily.iloc[spring_i]
    neap = daily.iloc[neap_i]
    # 与春季大潮相邻（±16 天）的小潮，用于同季节大小潮对比
    spring_date = pd.Timestamp(spring["date"])
    near_mask = (pd.to_datetime(daily["date"]) - spring_date).abs().dt.days <= 16
    near = daily.loc[near_mask]
    neap_near = daily.loc[near["range_m"].idxmin()]
    print("第 7 步  大小潮识别（逐日潮差法）")
    print("  大潮日:", spring["date"], "| 日潮差 %.3f m" % spring["range_m"])
    print("  小潮日:", neap["date"], "| 日潮差 %.3f m" % neap["range_m"])
    print("  与春季大潮相邻的小潮日:", neap_near["date"],
          "| 日潮差 %.3f m" % neap_near["range_m"])
    beat = abs(1.0 / (1.0 / 12.4206 - 1.0 / 12.0))
    print("  M2/S2 拍频周期约 %.1f h = %.2f d（12.42 h 与 12.00 h 之差）"
          % (beat, beat / 24.0))
    print()
    return daily, {"spring": spring, "neap": neap,
                   "neap_near_spring": neap_near}


def monthly_season_table(cfg: dict, woa18: pd.DataFrame,
                         ctx: dict, props_at) -> pd.DataFrame:
    """季节影响：按月均海温计算 dT、物性、稳态散热量与可放服务器数。
    另附 20 degC 基准工况行对比。"""
    rows = []
    for _, r in woa18.iterrows():
        m = int(r["月"])
        t_inf = float(r["%s_表层温度_degC" % SITE])
        p = props_at(t_inf)
        rho, cp, k, nu, pr, beta = p
        dT = T_MAX - t_inf
        u_mean = float(np.mean(ctx["U_base_by_month_depth"](cfg["depth"])
                                [ctx["months"] == m])) if m in ctx["months"] else 0.0
        steady = steady_capacity(cfg, t_inf, u_mean, props_at)
        rows.append({
            "月份": m, "海温_Tinf_degC": t_inf, "温差_dT_degC": dT,
            "密度_kg_m3": rho, "比热容_J_kgK": cp,
            "导热系数_W_mK": k, "运动粘度_m2_s": nu, "Pr": pr,
            "热膨胀系数_1_K": beta, "月均背景流速_m_s": u_mean,
            "h_sea_W_m2K": steady["h_sea"],
            "h_total_W_m2K": steady["h_total"],
            "散热能力_Q_W": steady["Q"],
            "可放服务器数_N": steady["N"],
            "空间上限_N_space": steady["N_space"],
        })
    base = steady_capacity(cfg, 20.0, 0.0, props_at)
    rows.append({
        "月份": "20C基准", "海温_Tinf_degC": 20.0, "温差_dT_degC": 60.0,
        "密度_kg_m3": props_at(20.0)[0], "比热容_J_kgK": props_at(20.0)[1],
        "导热系数_W_mK": props_at(20.0)[2], "运动粘度_m2_s": props_at(20.0)[3],
        "Pr": props_at(20.0)[4], "热膨胀系数_1_K": props_at(20.0)[5],
        "月均背景流速_m_s": 0.0, "h_sea_W_m2K": base["h_sea"],
        "h_total_W_m2K": base["h_total"], "散热能力_Q_W": base["Q"],
        "可放服务器数_N": base["N"], "空间上限_N_space": base["N_space"],
    })
    out = pd.DataFrame(rows)
    print("=" * 76)
    print("第 8 步  季节影响月度表（含 20 degC 基准对比）")
    print("=" * 76)
    print(out.round(4).to_string(index=False))
    print()
    return out


# ==================================================================
# 8. NSGA-II（嵌套 RK4 二分）：算法核心
# ==================================================================
class Individual:
    """NSGA-II 个体。"""
    __slots__ = ("chrom", "obj", "feasible", "viol", "rank", "crowd", "res")

    def __init__(self, chrom, obj, feasible, viol, res=None):
        self.chrom = chrom
        self.obj = obj
        self.feasible = feasible
        self.viol = viol
        self.rank = 0
        self.crowd = 0.0
        self.res = res


def dominates(p: Individual, q: Individual, tol: float = 1e-12) -> bool:
    if p.feasible != q.feasible:
        return p.feasible and not q.feasible
    if not p.feasible:
        return p.viol < q.viol - tol
    return bool(np.all(p.obj <= q.obj + tol) and np.any(p.obj < q.obj - tol))


def fast_non_dominated_sort(pop: list) -> list[list[int]]:
    n = len(pop)
    dominated = [set() for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(pop[i], pop[j]):
                dominated[i].add(j)
            elif dominates(pop[j], pop[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            pop[i].rank = 0
            fronts[0].append(i)
    k = 0
    while k < len(fronts) and fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    pop[j].rank = k + 1
                    nxt.append(j)
        k += 1
        if nxt:
            fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(pop: list, front_idx: list[int]) -> None:
    m = len(pop[0].obj)
    front = [pop[i] for i in front_idx]
    for ind in front:
        ind.crowd = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowd = float("inf")
        return
    for mj in range(m):
        order = sorted(range(len(front)), key=lambda i: front[i].obj[mj])
        front[order[0]].crowd = float("inf")
        front[order[-1]].crowd = float("inf")
        rng = front[order[-1]].obj[mj] - front[order[0]].obj[mj]
        if rng < 1e-12:
            continue
        for i in range(1, len(order) - 1):
            if front[order[i]].crowd != float("inf"):
                front[order[i]].crowd += (
                    front[order[i + 1]].obj[mj]
                    - front[order[i - 1]].obj[mj]) / rng


def tournament(pop: list, rng) -> Individual:
    idx = rng.integers(0, len(pop), size=2)
    best = pop[idx[0]]
    other = pop[idx[1]]
    if other.rank < best.rank or (other.rank == best.rank
                                  and other.crowd > best.crowd):
        best = other
    return best


def sbx(p1: float, p2: float, lo: float, hi: float, rng,
        pc: float = 0.90, eta_c: float = 15.0) -> tuple[float, float]:
    if rng.random() > pc:
        return float(p1), float(p2)
    u = rng.random()
    beta = ((2.0 * u) ** (1.0 / (eta_c + 1.0)) if u <= 0.5
            else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return float(np.clip(c1, lo, hi)), float(np.clip(c2, lo, hi))


def poly_mut(x: float, lo: float, hi: float, rng,
             pm: float = 0.10, eta_m: float = 20.0) -> float:
    if rng.random() > pm:
        return float(x)
    u = rng.random()
    delta = ((2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0 if u < 0.5
             else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0)))
    return float(np.clip(x + delta * (hi - lo), lo, hi))


def _evaluate_chrom(chrom: list, ctx: dict) -> dict:
    """NSGA-II 个体评价：NSGA 外层 + RK4 二分内层（嵌套）。
    变量：[材料索引, 深度 m, 壁厚 m]。"""
    mat_idx = int(round(chrom[0]))
    depth = float(chrom[1])
    wall = float(chrom[2])
    if not (0 <= mat_idx < len(ctx["material_df"])):
        return {"feasible": False, "viol": 1e12, "obj": np.array([1e12, 1e12, 1e12]),
                "res": None}
    cfg = make_cfg(mat_idx, depth, wall, ctx["material_df"], ctx["cp_map"])
    geom = cfg["geom"]
    tide_max_dev = float(ctx["tide"].max() - ctx["tide_mean"])
    p_hydro_max = ctx["rho_sw"] * G * (depth + tide_max_dev)
    life, t_req, corr_allow_mm = calc_life(cfg, p_hydro_max)
    g1 = wall - t_req
    g2 = life - MIN_LIFE

    # 环境缓存（同一 worker 内按深度复用）
    env_cache = ctx.setdefault("_env_cache", {})
    dkey = round(depth, 2)
    if dkey not in env_cache:
        env_cache[dkey] = build_environment(ctx, depth)
    env = env_cache[dkey]

    res_sim, n_max = bisect_max_N(cfg, env, ctx["props_at"])
    g3 = n_max - 1
    g4 = T_MAX - res_sim["T_max"] + 1e-9
    g5 = geom["D_in"] - 0.05
    viol = (max(0.0, -g1) + max(0.0, -g2) + max(0.0, -g3)
            + max(0.0, -g4) + max(0.0, -g5))
    feasible = viol <= 1e-9
    q_mean = res_sim["Q_mean"]
    res = {"mat_idx": mat_idx, "depth": depth, "wall": wall,
           "材料": cfg["材料"], "N": n_max, "Q_mean": q_mean,
           "Q_cap_mean": res_sim["Q_cap_mean"],
           "Q_min": res_sim["Q_min"], "Q_max": res_sim["Q_max"],
           "T_max": res_sim["T_max"], "T_mean": res_sim["T_mean"],
           "cost": cfg["cost"], "life": life, "t_req": t_req,
           "corr_allow_mm": corr_allow_mm,
           "regime": res_sim["regime_cnt"],
           "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5,
           "feasible": feasible, "viol": viol}
    res["obj"] = np.array([-q_mean, cfg["cost"], -life])
    return res


WORKER_CTX: dict | None = None


def _init_worker(ctx: dict) -> None:
    global WORKER_CTX
    WORKER_CTX = ctx


def _worker_eval(chrom: list) -> dict:
    return _evaluate_chrom(chrom, WORKER_CTX)


def nsga2(ctx: dict, pop_size: int = NSGA_POP, generations: int = NSGA_GEN,
          seed: int = NSGA_SEED, workers: int = 1) -> tuple[list, list, list]:
    """NSGA-II 主循环，评价函数内嵌 RK4+二分（问题 4 联合优化）。"""
    rng = np.random.default_rng(seed)
    n_mat = len(ctx["material_df"])
    bounds = [(0.0, float(n_mat - 1)), (DEPTH_MIN, DEPTH_MAX),
              (WALL_MIN, WALL_MAX)]

    def evaluate(chrom: list) -> Individual:
        r = _evaluate_chrom(chrom, ctx)
        if r["res"] is None:
            return Individual(np.array(chrom), r["obj"], False, r["viol"])
        return Individual(np.array(chrom), r["obj"], r["feasible"],
                          r["viol"], r)

    def eval_many(chroms: list) -> list[Individual]:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(ctx,)) as ex:
                results = list(ex.map(_worker_eval, chroms))
            return [evaluate(c) if False else _ind_from_res(c, r)
                    for c, r in zip(chroms, results)]
        return [evaluate(c) for c in chroms]

    def _ind_from_res(chrom, r):
        return Individual(np.array(chrom), r["obj"], r["feasible"],
                          r["viol"], r)

    init_chroms = []
    for _ in range(pop_size):
        init_chroms.append(
            [float(rng.integers(0, n_mat)), rng.uniform(DEPTH_MIN, DEPTH_MAX),
             rng.uniform(WALL_MIN, WALL_MAX)])
    pop = eval_many(init_chroms)
    history = []

    for gen in range(1, generations + 1):
        fronts = fast_non_dominated_sort(pop)
        for f_idx in fronts:
            crowding_distance(pop, f_idx)
        parents = [tournament(pop, rng) for _ in range(pop_size)]
        offspring_chroms = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % pop_size]
            c1, c2 = [0.0] * 3, [0.0] * 3
            for j in range(3):
                if j == 0:
                    c1[j] = p1.chrom[j] if rng.random() < 0.5 else p2.chrom[j]
                    c2[j] = p2.chrom[j] if rng.random() < 0.5 else p1.chrom[j]
                    if rng.random() < NSGA_PM:
                        c1[j] = float(rng.integers(0, n_mat))
                    if rng.random() < NSGA_PM:
                        c2[j] = float(rng.integers(0, n_mat))
                else:
                    lo, hi = bounds[j]
                    a, b = sbx(p1.chrom[j], p2.chrom[j], lo, hi, rng,
                               NSGA_PC, NSGA_ETA_C)
                    c1[j] = poly_mut(a, lo, hi, rng, NSGA_PM, NSGA_ETA_M)
                    c2[j] = poly_mut(b, lo, hi, rng, NSGA_PM, NSGA_ETA_M)
            offspring_chroms.append(c1)
            offspring_chroms.append(c2)
        offspring = eval_many(offspring_chroms)
        combined = pop + offspring
        fronts = fast_non_dominated_sort(combined)
        for f_idx in fronts:
            crowding_distance(combined, f_idx)
        new_pop = []
        for f_idx in fronts:
            if len(new_pop) + len(f_idx) <= pop_size:
                new_pop.extend(combined[i] for i in f_idx)
            else:
                need = pop_size - len(new_pop)
                if need <= 0:
                    break
                order = sorted(f_idx, key=lambda i: -combined[i].crowd)
                new_pop.extend(combined[i] for i in order[:need])
                break
        pop = new_pop
        fea = [i for i in pop if i.feasible]
        if fea:
            qs = [i.res["Q_mean"] for i in fea if i.res]
            best_q = max(qs) if qs else 0.0
            mean_cost = float(np.mean([i.res["cost"] for i in fea if i.res]))
            front_sz = len([i for i in pop if i.feasible and i.rank == 0])
        else:
            best_q, mean_cost, front_sz = 0.0, 0.0, 0
        history.append((gen, front_sz, best_q, mean_cost))
        print("    第 %3d 代：可行前沿个体 %3d，最大 Q_mean=%.0f W，"
              "平均成本=%.0f 元" % (gen, front_sz, best_q, mean_cost))

    fronts = fast_non_dominated_sort(pop)
    front0 = [pop[i] for i in fronts[0] if pop[i].feasible]
    pareto = [i.res for i in front0 if i.res is not None]
    return pop, pareto, history


# ==================================================================
# 9. 灵敏度分析
# ==================================================================
def sensitivity_analysis(base_cfg: dict, ctx: dict,
                         props_at) -> pd.DataFrame:
    """对季节振幅、潮差、潮汐流速幅值、壁厚、导热系数做灵敏度扫描。"""
    rows = []

    def run_case(cfg: dict, u_amp: float, tide_scale: float,
                 season_scale: float) -> dict:
        ctx2 = dict(ctx)
        ctx2["U_TIDE_AMP"] = u_amp
        ctx2["tide"] = ctx["tide_mean"] + (ctx["tide"] - ctx["tide_mean"]) * tide_scale
        ctx2["season_anom"] = ctx["season_anom"] * season_scale
        ctx2.pop("_env_cache", None)
        env = build_environment(ctx2, cfg["depth"])
        sim, n = bisect_max_N(cfg, env, props_at)
        return {"N": n, "T_max": sim["T_max"], "Q_mean": sim["Q_mean"],
                "Q_cap_mean": sim["Q_cap_mean"],
                "Q_fluct_pct": sim["Q_fluct_pct"]}

    base = run_case(base_cfg, ctx["U_TIDE_AMP"], 1.0, 1.0)
    cases = [
        ("季节振幅 +10%", {"season_scale": 1.10}),
        ("季节振幅 -10%", {"season_scale": 0.90}),
        ("潮差 +10%", {"tide_scale": 1.10}),
        ("潮差 -10%", {"tide_scale": 0.90}),
        ("潮汐流速 0 m/s", {"u_amp": 0.0}),
        ("潮汐流速 0.10 m/s", {"u_amp": 0.10}),
        ("潮汐流速 0.20 m/s", {"u_amp": 0.20}),
        ("潮汐流速 0.30 m/s", {"u_amp": 0.30}),
    ]
    for name, kw in cases:
        r = run_case(base_cfg, kw.get("u_amp", ctx["U_TIDE_AMP"]),
                     kw.get("tide_scale", 1.0), kw.get("season_scale", 1.0))
        rows.append({"扰动项": name, "N": r["N"],
                     "N变化_pct": (r["N"] - base["N"]) / base["N"] * 100 if base["N"] else 0,
                     "T_max_degC": r["T_max"],
                     "Q_mean_W": r["Q_mean"],
                     "Q_mean变化_pct": (r["Q_mean"] - base["Q_mean"]) / base["Q_mean"] * 100,
                     "Q_cap_mean_W": r["Q_cap_mean"],
                     "Q_cap变化_pct": (r["Q_cap_mean"] - base["Q_cap_mean"])
                     / base["Q_cap_mean"] * 100,
                     "Q波动_pct": r["Q_fluct_pct"]})

    # 壁厚 ±10%（重新构造 cfg）
    for s, name in ((1.10, "壁厚 +10%"), (0.90, "壁厚 -10%")):
        cfg2 = make_cfg(base_cfg["mat_idx"], base_cfg["depth"],
                        base_cfg["wall"] * s, ctx["material_df"], ctx["cp_map"])
        r = run_case(cfg2, ctx["U_TIDE_AMP"], 1.0, 1.0)
        rows.append({"扰动项": name, "N": r["N"],
                     "N变化_pct": (r["N"] - base["N"]) / base["N"] * 100 if base["N"] else 0,
                     "T_max_degC": r["T_max"], "Q_mean_W": r["Q_mean"],
                     "Q_mean变化_pct": (r["Q_mean"] - base["Q_mean"]) / base["Q_mean"] * 100,
                     "Q_cap_mean_W": r["Q_cap_mean"],
                     "Q_cap变化_pct": (r["Q_cap_mean"] - base["Q_cap_mean"])
                     / base["Q_cap_mean"] * 100,
                     "Q波动_pct": r["Q_fluct_pct"]})

    out = pd.DataFrame(rows)
    print("=" * 76)
    print("第 9 步  灵敏度分析（基准：N=%d, T_max=%.2f degC, Q_mean=%.0f W, "
          "Q_cap=%.0f W）"
          % (base["N"], base["T_max"], base["Q_mean"], base["Q_cap_mean"]))
    print("=" * 76)
    print(out.round(3).to_string(index=False))
    print()
    return out


def validate_time_varying_model(base_cfg: dict, env: dict, sim: dict,
                                n_max: int, ctx: dict,
                                props_at) -> dict[str, pd.DataFrame]:
    """时变传热模型六项检验，返回汇总表及各项明细表。"""
    summary: list[dict[str, Any]] = []

    def add(item: str, metric: str, value: Any,
            expected: str, conclusion: str) -> None:
        summary.append({"检验项": item, "指标": metric, "数值": value,
                        "阈值/期望": expected, "结论": conclusion})

    def flag(ok: bool, detail: str = "通过") -> str:
        return detail if ok else "需复核"

    def constant_env(t_inf: float, u: float) -> dict:
        n_h = len(ctx["t_hours"])
        hf = h_forced_cylinder(D_EXT, u, props_at(t_inf)) if u > 1e-9 else 0.0
        return {"t_hours": ctx["t_hours"],
                "T_inf": np.full(n_h, float(t_inf)),
                "U": np.full(n_h, float(u)),
                "h_forced": np.full(n_h, float(hf)),
                "d_eff": np.full(n_h, float(base_cfg["depth"])),
                "hn_grid": ctx["hn_grid"]}

    # ---------- 1. 热平衡残差 ----------
    n_hours = len(env["T_inf"])
    dt = 3600.0
    q_gen = n_max * Q0
    mcp = base_cfg["mcp"]
    dT_step = np.diff(sim["T_arr"])
    q_eff = sim["Q_eff_arr"][1:]
    q_end = sim["Q_arr"][1:]
    res_eff_J = mcp * dT_step - (q_gen - q_eff) * dt
    res_end_J = mcp * dT_step - (q_gen - q_end) * dt
    res_mid_J = mcp * dT_step - (q_gen - 0.5 * (sim["Q_arr"][1:] + sim["Q_arr"][:-1])) * dt
    scale = max(q_gen * dt, 1.0)
    sl_res = slice(168, n_hours - 1)
    max_rel_eff = float(np.abs(res_eff_J[sl_res]).max() / scale)
    max_rel_end = float(np.abs(res_end_J[sl_res]).max() / scale)
    max_rel_mid = float(np.abs(res_mid_J[sl_res]).max() / scale)
    max_abs_eff_W = float(np.abs(res_eff_J[sl_res]).max() / dt)
    annual_rel_end = float(np.abs(res_end_J[sl_res].sum())
                           / max(q_gen * (n_hours - 168) * dt, 1.0))
    max_abs_end_W = float(np.abs(res_end_J[sl_res]).max() / dt)
    ok_energy = (max_rel_end < 0.02 and max_rel_mid < 0.01
                 and annual_rel_end < 0.02)
    add("1. 热平衡残差", "端点Q最大相对残差（独立离散检验）",
        f"{max_rel_end:.3e}", "< 2e-2", flag(ok_energy))
    add("1. 热平衡残差", "中点Q最大相对残差",
        f"{max_rel_mid:.3e}", "< 1e-2", flag(ok_energy))
    add("1. 热平衡残差", "年均累计残差相对值（端点Q）",
        f"{annual_rel_end:.3e}", "< 2e-2", flag(ok_energy))
    add("1. 热平衡残差", "每步最大残差绝对量（端点Q）",
        f"{max_abs_end_W:.4e} W", "接近 0", flag(ok_energy))
    add("1. 热平衡残差", "RK4等效Q内部自洽残差",
        f"{max_rel_eff:.3e}", "非独立判据，仅自检", "通过（自洽）")

    # ---------- 2. 边界条件合理性 ----------
    t_inf = np.asarray(env["T_inf"], dtype=float)
    u_h = np.asarray(env["U"], dtype=float)
    d_eff = np.asarray(env["d_eff"], dtype=float)
    finite_ok = all(np.isfinite(v).all() for v in (t_inf, u_h, d_eff))
    t_ok = 0.0 <= t_inf.min() <= t_inf.max() <= 40.0
    u_ok = 0.0 <= u_h.min() <= u_h.max() <= 1.0
    d_ok = 0.0 < d_eff.min() <= d_eff.max() <= 200.0
    j_t = float(np.abs(np.diff(t_inf)).max())
    j_u = float(np.abs(np.diff(u_h)).max())
    j_d = float(np.abs(np.diff(d_eff)).max())
    jump_ok = j_t <= 0.5 and j_u <= 0.10 and j_d <= 1.0
    ok_boundary = finite_ok and t_ok and u_ok and d_ok and jump_ok
    add("2. 边界条件", "海温范围",
        f"{t_inf.min():.3f} ~ {t_inf.max():.3f} degC",
        "0 ~ 40 degC", flag(t_ok))
    add("2. 边界条件", "流速范围",
        f"{u_h.min():.4f} ~ {u_h.max():.4f} m/s",
        "0 ~ 1 m/s", flag(u_ok))
    add("2. 边界条件", "浸没深度范围",
        f"{d_eff.min():.3f} ~ {d_eff.max():.3f} m",
        "0 ~ 200 m", flag(d_ok))
    add("2. 边界条件", "海温最大小时跳变",
        f"{j_t:.4f} degC/h", "<= 0.5 degC/h", flag(jump_ok))
    add("2. 边界条件", "流速最大小时跳变",
        f"{j_u:.4f} m/s/h", "<= 0.10 m/s/h", flag(jump_ok))
    add("2. 边界条件", "水深最大小时跳变",
        f"{j_d:.4f} m/h", "<= 1.0 m/h", flag(jump_ok))
    add("2. 边界条件", "数值完整性", "无 NaN/Inf",
        "全部有限", flag(finite_ok))

    # ---------- 3. 换热关联式适用域 ----------
    t_rep = float(np.mean(env["T_inf"][168:]))
    t_film_rep = t_rep + 5.0
    p_rep = props_at(t_film_rep)
    dT_rep = 10.0
    T_rep = t_rep + dT_rep
    corr_rows = []
    for u_val in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02,
                  0.05, 0.10, 0.20, 0.30, 0.50):
        env_u = constant_env(t_rep, u_val)
        c = _conductance(base_cfg, T_rep, 0.0, env_u, props_at)
        re = u_val * D_EXT / p_rep[3] if u_val > 1e-9 else 0.0
        gr = G * p_rep[5] * dT_rep * D_EXT ** 3 / p_rep[3] ** 2
        gr_re2 = c["Gr_Re2"]
        hm_hn = (c["h_mixed"] / c["h_nat"] if c["h_nat"] > 1e-12 else 0.0)
        hm_hf = (c["h_mixed"] / c["h_forced"]
                 if c["h_forced"] > 1e-12 else 0.0)
        corr_rows.append({
            "流速_m_s": u_val, "Re": re, "Gr": gr,
            "Gr_Re2": "inf" if u_val <= 1e-9 else f"{gr_re2:.4g}",
            "流态": c["regime"], "h_nat_W_m2K": c["h_nat"],
            "h_forced_W_m2K": c["h_forced"], "h_mixed_W_m2K": c["h_mixed"],
            "h_mixed_h_nat": hm_hn, "h_mixed_h_forced": hm_hf,
        })
    corr_df = pd.DataFrame(corr_rows)

    u_actual = env["U"][168:]
    t_wo_actual = sim["t_wo_arr"][168:]
    t_inf_actual = env["T_inf"][168:]
    t_film_actual = (t_wo_actual + t_inf_actual) / 2.0
    pa = ctx["props_at"].arrays(t_film_actual)
    dT_actual = np.maximum(t_wo_actual - t_inf_actual, 1e-6)
    gr_actual = G * pa["beta"] * dT_actual * D_EXT ** 3 / pa["nu"] ** 2
    ra_actual = gr_actual * pa["Pr"]
    re_actual = u_actual * D_EXT / pa["nu"]
    repr_actual = re_actual * pa["Pr"]
    natural_ok = (ra_actual <= 1e12) & (dT_actual > 0)
    forced_ok = ((u_actual <= 1e-9)
                 | ((repr_actual >= 0.2) & (repr_actual <= 1e7)))
    pct_in_domain = float((natural_ok & forced_ok).mean() * 100.0)
    c_low = _conductance(base_cfg, T_rep, 0.0,
                         constant_env(t_rep, 0.001), props_at)
    c_high = _conductance(base_cfg, T_rep, 0.0,
                          constant_env(t_rep, 0.30), props_at)
    low_ratio = c_low["h_mixed"] / c_low["h_nat"]
    high_ratio = c_high["h_mixed"] / c_high["h_forced"]
    ok_corr = (pct_in_domain >= 99.0
               and abs(low_ratio - 1.0) <= 0.02
               and abs(high_ratio - 1.0) <= 0.05)
    add("3. 关联式适用域", "低速 0.001 m/s 时 h_mixed/h_nat",
        f"{low_ratio:.4f}", "≈ 1（退化为自然对流）",
        flag(abs(low_ratio - 1.0) <= 0.02))
    add("3. 关联式适用域", "高速 0.30 m/s 时 h_mixed/h_forced",
        f"{high_ratio:.4f}", "≈ 1（强制对流主导）",
        flag(abs(high_ratio - 1.0) <= 0.05))
    add("3. 关联式适用域", "RePr 全年范围",
        f"{repr_actual.min():.3g} ~ {repr_actual.max():.3g}",
        "0.2 ~ 1e7", flag(forced_ok.mean() >= 0.99))
    add("3. 关联式适用域", "关联式域内小时占比",
        f"{pct_in_domain:.3f}%", ">= 99%", flag(ok_corr))
    add("3. 关联式适用域", "全年流态统计",
        f"自然 {sim['regime_cnt']['自然对流主导']} h / "
        f"强制 {sim['regime_cnt']['强制对流主导']} h / "
        f"混合 {sim['regime_cnt']['混合对流']} h",
        "自然/强制/混合均有覆盖", flag(ok_corr))

    # ---------- 4. 极限情景 ----------
    t_hot = float(np.max(env["T_inf"]))
    t_cold = float(np.min(env["T_inf"]))
    u_high = 0.50
    sim_hot = simulate_year(base_cfg, n_max, constant_env(t_hot, 0.0),
                            props_at)
    sim_cold = simulate_year(base_cfg, n_max,
                             constant_env(t_cold, u_high), props_at)
    sim_cold_none = simulate_year(base_cfg, 0,
                                  constant_env(t_cold, u_high), props_at,
                                  T0=t_hot)

    def t_min(s: dict) -> float:
        return float(np.min(s["T_arr"][168:]))

    hot_ok = (sim_hot["ok"] and 0.0 < sim_hot["T_max"] < 250.0
              and sim_hot["T_max"] > t_hot)
    cold_ok = (sim_cold["ok"] and t_min(sim_cold) >= t_cold - 0.05
               and t_min(sim_cold) > 0.0)
    none_ok = (sim_cold_none["ok"]
               and abs(t_min(sim_cold_none) - t_cold) <= 0.05
               and t_min(sim_cold_none) > 0.0)
    extreme_rows = [
        {"情景": "最热季节+零流速+Nmax", "N": n_max,
         "T_inf_degC": t_hot, "U_m_s": 0.0,
         "T_min_degC": t_min(sim_hot), "T_max_degC": sim_hot["T_max"],
         "T_mean_degC": sim_hot["T_mean"], "合理性": flag(hot_ok)},
        {"情景": "最冷季节+0.5 m/s+Nmax", "N": n_max,
         "T_inf_degC": t_cold, "U_m_s": u_high,
         "T_min_degC": t_min(sim_cold), "T_max_degC": sim_cold["T_max"],
         "T_mean_degC": sim_cold["T_mean"], "合理性": flag(cold_ok)},
        {"情景": "最冷季节+0.5 m/s+无产热", "N": 0,
         "T_inf_degC": t_cold, "U_m_s": u_high,
         "T_min_degC": t_min(sim_cold_none),
         "T_max_degC": sim_cold_none["T_max"],
         "T_mean_degC": sim_cold_none["T_mean"], "合理性": flag(none_ok)},
    ]
    extreme_df = pd.DataFrame(extreme_rows)
    add("4. 极限情景", "高温极值 T_max",
        f"{sim_hot['T_max']:.3f} degC",
        "有限且大于海温；超过 80 degC 属预期降载", flag(hot_ok))
    add("4. 极限情景", "低温运行 T_min",
        f"{t_min(sim_cold):.3f} degC", ">= 海温且 > 0", flag(cold_ok))
    add("4. 极限情景", "无产热下限 T_min",
        f"{t_min(sim_cold_none):.3f} degC", "≈ 海温且 > 0", flag(none_ok))

    # ---------- 5. 参数敏感性（固定 N=n_max） ----------
    def case_env(season_scale: float = 1.0,
                 u_amp: float | None = None) -> dict:
        ctx2 = dict(ctx)
        if u_amp is not None:
            ctx2["U_TIDE_AMP"] = float(u_amp)
        if season_scale != 1.0:
            ctx2["season_anom"] = ctx["season_anom"] * season_scale
        ctx2.pop("_env_cache", None)
        return build_environment(ctx2, base_cfg["depth"])

    sens_rows = [{"扰动项": "基准", "参数": "-",
                  "T_max_degC": sim["T_max"], "T_max变化_K": 0.0,
                  "方向": "基准"}]
    cases = [
        ("季节振幅 +10%", "season", 1.10),
        ("季节振幅 -10%", "season", 0.90),
        ("潮汐流速幅值 0 m/s", "u_amp", 0.0),
        ("潮汐流速幅值 0.30 m/s", "u_amp", 0.30),
        ("壳体质量 +50%", "mass", 1.50),
        ("壳体质量 -50%", "mass", 0.50),
        ("比热容 +50%", "cp", 1.50),
        ("比热容 -50%", "cp", 0.50),
        ("有效散热面积 +10%", "area", 1.10),
        ("有效散热面积 -10%", "area", 0.90),
    ]
    expected_dir = {
        "季节振幅 +10%": 1, "季节振幅 -10%": -1,
        "潮汐流速幅值 0 m/s": 1, "潮汐流速幅值 0.30 m/s": -1,
        "壳体质量 +50%": -1, "壳体质量 -50%": 1,
        "比热容 +50%": -1, "比热容 -50%": 1,
        "有效散热面积 +10%": -1, "有效散热面积 -10%": 1,
    }
    for name, kind, factor in cases:
        if kind == "season":
            cfg_case = base_cfg
            env_case = case_env(season_scale=factor)
        elif kind == "u_amp":
            cfg_case = base_cfg
            env_case = case_env(u_amp=factor)
        elif kind == "mass":
            cfg_case = deepcopy(base_cfg)
            cfg_case["mcp"] *= factor
            cfg_case["m_shell"] *= factor
            env_case = env
        elif kind == "cp":
            cfg_case = deepcopy(base_cfg)
            cfg_case["cp_shell"] *= factor
            cfg_case["mcp"] *= factor
            env_case = env
        else:
            cfg_case = deepcopy(base_cfg)
            cfg_case["geom"] = dict(base_cfg["geom"])
            cfg_case["geom"]["A_out"] *= factor
            env_case = env
        s_case = simulate_year(cfg_case, n_max, env_case, props_at)
        delta = s_case["T_max"] - sim["T_max"]
        if abs(delta) < 0.02:
            direction = "基本不变（准稳态主导）"
        elif ((delta > 0 and expected_dir[name] > 0)
              or (delta < 0 and expected_dir[name] < 0)):
            direction = "符合物理直觉"
        else:
            direction = "需复核"
        sens_rows.append({"扰动项": name, "参数": kind,
                          "T_max_degC": s_case["T_max"],
                          "T_max变化_K": delta, "方向": direction})
    sens_df = pd.DataFrame(sens_rows)
    n_bad = int((sens_df["方向"] == "需复核").sum())
    add("5. 参数敏感性", "季节振幅 +10% T_max 变化",
        f"{sens_df.loc[1, 'T_max变化_K']:+.4f} K", "> 0",
        sens_df.loc[1, "方向"])
    add("5. 参数敏感性", "潮汐流速 0.30 m/s T_max 变化",
        f"{sens_df.loc[4, 'T_max变化_K']:+.4f} K", "< 0",
        sens_df.loc[4, "方向"])
    add("5. 参数敏感性", "壳体质量 +50% T_max 变化",
        f"{sens_df.loc[5, 'T_max变化_K']:+.4f} K",
        "≤ 0 或基本不变", sens_df.loc[5, "方向"])
    add("5. 参数敏感性", "比热容 +50% T_max 变化",
        f"{sens_df.loc[7, 'T_max变化_K']:+.4f} K",
        "≤ 0 或基本不变", sens_df.loc[7, "方向"])
    add("5. 参数敏感性", "有效散热面积 +10% T_max 变化",
        f"{sens_df.loc[9, 'T_max变化_K']:+.4f} K", "< 0",
        sens_df.loc[9, "方向"])
    add("5. 参数敏感性", "方向一致性",
        f"{len(sens_df) - 1 - n_bad}/10 项合理或基本不变",
        "全部合理或基本不变", flag(n_bad == 0))

    # ---------- 6. 周期稳定性 ----------
    n_year = len(env["T_inf"])
    env2 = {key: np.concatenate([env[key], env[key]])
            for key in ("T_inf", "U", "h_forced", "d_eff")}
    env2["t_hours"] = np.arange(2 * n_year)
    env2["hn_grid"] = env["hn_grid"]
    sim2 = simulate_year(base_cfg, n_max, env2, props_at, warmup_h=0)
    y1 = sim2["T_arr"][:n_year]
    y2 = sim2["T_arr"][n_year:]
    diff2 = np.abs(y2 - y1)
    diff_spin = diff2[168:]
    max_full = float(diff2.max())
    max_spin = float(diff_spin.max())
    mean_spin = float(diff_spin.mean())
    p99_spin = float(np.quantile(diff_spin, 0.99))
    ok_period = sim2["ok"] and max_spin <= 0.1
    period_df = pd.DataFrame([
        {"指标": "两年曲线最大绝对温差", "数值_K": max_full,
         "阈值_K": "越小越好", "结论": "含初值瞬态"},
        {"指标": "去初始瞬态后最大绝对温差", "数值_K": max_spin,
         "阈值_K": "<= 0.1", "结论": flag(ok_period)},
        {"指标": "去初始瞬态后平均绝对温差", "数值_K": mean_spin,
         "阈值_K": "<= 0.1", "结论": flag(ok_period)},
        {"指标": "去初始瞬态后 P99 绝对温差", "数值_K": p99_spin,
         "阈值_K": "<= 0.1", "结论": flag(ok_period)},
    ])
    add("6. 周期稳定性", "去初始瞬态后最大绝对温差",
        f"{max_spin:.6f} K", "<= 0.1 K", flag(ok_period))
    add("6. 周期稳定性", "去初始瞬态后平均绝对温差",
        f"{mean_spin:.6f} K", "<= 0.1 K", flag(ok_period))
    add("6. 周期稳定性", "初值影响结论",
        "第二年与第一年基本重合",
        "差异小时方可直接采用年度结果", flag(ok_period))

    summary_df = pd.DataFrame(summary)
    print("=" * 76)
    print("第 9b 步  时变传热模型六项检验")
    print("=" * 76)
    print(summary_df.to_string(index=False))
    print()
    print("[说明] 1-4 项为模型方程与边界合理性；5 项固定 N=%d 观察 T_max；"
          "6 项连续仿真两个相同年度并比较。" % n_max)
    print()
    return {"汇总": summary_df, "参数敏感性": sens_df,
            "关联式适用性": corr_df, "极限情景": extreme_df,
            "周期稳定性": period_df}


# ==================================================================
# 10. 出图与主流程
# ==================================================================
def plot_season(monthly: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    df = monthly[monthly["月份"] != "20C基准"].copy()
    df["月份"] = df["月份"].astype(int)
    axes[0].plot(df["月份"], df["海温_Tinf_degC"], "-o", label="海温")
    axes[0].plot(df["月份"], df["温差_dT_degC"], "-s", label="温差 dT")
    axes[0].axhline(20.0, ls="--", color="gray", label="20 degC 基准")
    axes[0].set_xlabel("月份"); axes[0].set_ylabel("degC")
    axes[0].set_title("季节海温与温差"); axes[0].legend()
    axes[1].plot(df["月份"], df["散热能力_Q_W"] / 1e3, "-o", label="Q (kW)")
    axes[1].plot(df["月份"], df["可放服务器数_N"], "-s", label="月度 N")
    axes[1].set_xlabel("月份"); axes[1].set_ylabel("kW / 台")
    axes[1].set_title("月度散热能力与服务器数"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_tide(tide_df: pd.DataFrame, tidal: pd.DataFrame, daily: pd.DataFrame,
              summary: dict, out: Path) -> None:
    h = tide_df["tide_height_m"].to_numpy(float)
    hours = np.arange(len(h))
    t0 = pd.Timestamp("2026-01-01 01:00:00")
    ts = t0 + pd.to_timedelta(hours, unit="h")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(ts[:14 * 24], h[:14 * 24], lw=0.8, label="天文潮预报")
    axes[0].set_title("2026 前 14 天潮位（天文潮预报，不含风暴潮/余水位）")
    axes[0].set_ylabel("潮位 m"); axes[0].legend()
    axes[1].plot(pd.to_datetime(daily["date"]), daily["range_m"], lw=0.8)
    spring = daily.iloc[int(daily["range_m"].idxmax())]
    neap = daily.iloc[int(daily["range_m"].idxmin())]
    axes[1].axvline(pd.Timestamp(spring["date"]), color="red",
                    label="大潮 %.3f m" % spring["range_m"])
    axes[1].axvline(pd.Timestamp(neap["date"]), color="blue",
                    label="小潮 %.3f m" % neap["range_m"])
    axes[1].set_title("逐日潮差与大小潮周期"); axes[1].set_ylabel("日潮差 m")
    axes[1].legend()
    freqs = np.fft.rfftfreq(len(h), 1.0)
    amp = np.abs(np.fft.rfft(h - h.mean())) * 2.0 / len(h)
    axes[2].semilogx(freqs[1:], amp[1:], lw=0.6)
    for name in ["M2", "S2", "K1", "O1"]:
        f_t = TIDAL_SPEEDS_DEG_PER_H[name] / 360.0
        idx = int(np.argmin(np.abs(freqs - f_t)))
        axes[2].axvline(f_t, color="gray", ls=":", lw=1)
        axes[2].annotate("%s\n%.3f m" % (name, amp[idx]),
                         xy=(f_t, amp[idx]), fontsize=8)
    axes[2].set_xlabel("频率 (1/h)"); axes[2].set_ylabel("FFT 振幅 m")
    axes[2].set_title("潮汐 FFT 频谱（M2/S2/K1/O1 已标注）")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_spring_neap(env: dict, sim: dict, spring_h: int, neap_h: int,
                     out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sls = slice(max(spring_h - 24, 0), spring_h + 24 * 6)
    sln = slice(max(neap_h - 24, 0), neap_h + 24 * 6)
    axes[0].plot(sim["Q_cap_arr"][sls] / 1e3, label="大潮 Q_cap (kW)")
    axes[0].plot(sim["Q_cap_arr"][sln] / 1e3, label="小潮 Q_cap (kW)")
    axes[0].set_ylabel("Q_cap kW")
    axes[0].set_title("大小潮逐时极限散热能力（壳温 80 degC）")
    axes[1].plot(sim["T_arr"][sls], label="大潮 T_shell")
    axes[1].plot(sim["T_arr"][sln], label="小潮 T_shell")
    axes[1].set_ylabel("T degC"); axes[1].set_xlabel("小时")
    axes[1].set_title("大小潮逐时壳温")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_worst(env: dict, sim: dict, ctx: dict, out: Path) -> None:
    idx = sim["worst_idx"]
    sl = slice(idx - 48, idx + 48)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(sim["T_arr"][sl], label="全年最高壳温窗口")
    axes[0].axhline(T_MAX, color="red", ls="--", label="T_max=80 degC")
    axes[0].set_ylabel("壳温 degC"); axes[0].legend()
    axes[1].plot(sim["Q_cap_arr"][sl] / 1e3, label="极限散热能力 Q_cap")
    axes[1].plot(sim["Q_arr"][sl] / 1e3, label="实际散热量 Q")
    axes[1].plot(env["T_inf"][sl], label="海温 T_inf")
    axes[1].set_ylabel("kW / degC"); axes[1].set_xlabel("小时")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_pareto(pareto_df: pd.DataFrame, history: list, out_dir: Path) -> None:
    if len(pareto_df):
        fig = plt.figure(figsize=(10, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pareto_df["Q_mean_W"] / 1e3, pareto_df["cost_元"] / 1e4,
                   pareto_df["life_年"], c=pareto_df["N"], cmap="viridis")
        ax.set_xlabel("Q_mean kW"); ax.set_ylabel("成本 万元")
        ax.set_zlabel("寿命 年"); ax.set_title("NSGA-II 帕累托前沿（嵌套 RK4）")
        fig.tight_layout(); fig.savefig(out_dir / "图5_帕累托前沿.png", dpi=150)
        plt.close(fig)
    if history:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        h = np.array(history)
        ax.plot(h[:, 0], h[:, 2] / 1e3, "-o", label="最大 Q_mean (kW)")
        ax.set_xlabel("代数"); ax.set_ylabel("kW")
        ax.set_title("NSGA-II 收敛过程")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "图6_NSGA2收敛.png", dpi=150)
        plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(sens))
    col = "Q_cap变化_pct" if "Q_cap变化_pct" in sens.columns else "Q_mean变化_pct"
    ax.barh(y, sens[col], color="#55A868")
    ax.set_yticks(y); ax.set_yticklabels(sens["扰动项"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(col + " %")
    ax.set_title("灵敏度：环境与结构参数对极限散热能力的影响")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_model_validation(base_cfg: dict, env: dict, sim: dict,
                          model_check: dict[str, pd.DataFrame],
                          out: Path) -> None:
    """六项检验汇总图：残差、边界、关联式、极限情景、敏感性、周期稳定性。"""
    fig = plt.figure(figsize=(16.5, 8.8))
    gs = fig.add_gridspec(3, 3)
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0:2, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[1, 1], sharex=axes[0, 1])
    axes[0, 2] = fig.add_subplot(gs[0:2, 2])
    axes[1, 0] = fig.add_subplot(gs[2, 0])
    axes[1, 1] = fig.add_subplot(gs[2, 1])
    axes[1, 2] = fig.add_subplot(gs[2, 2])
    dt = 3600.0
    q_gen = sim["N"] * Q0
    mcp = base_cfg["mcp"]
    res_end_W = (mcp * np.diff(sim["T_arr"]) / dt
                 - (q_gen - sim["Q_arr"][1:]))

    ax = axes[0, 0]
    ax.plot(res_end_W, lw=0.35, color="#C44E52")
    ax.set_title("1 热平衡残差（端点Q，W）")
    ax.set_xlabel("小时"); ax.set_ylabel("残差 (W)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(env["T_inf"], color="#4C72B0", lw=0.7, label="海温")
    ax.set_ylabel("海温 (degC)")
    ax.set_title("2 边界条件：海温/流速")
    ax.grid(alpha=0.3)

    ax_u.plot(env["U"], color="#DD8452", lw=0.7, label="流速")
    ax_u.set_ylabel("流速 (m/s)")
    ax_u.set_xlabel("小时")
    ax_u.grid(alpha=0.3)

    ax = axes[0, 2]
    corr = model_check["关联式适用性"]
    ax.plot(corr["流速_m_s"], corr["h_nat_W_m2K"], "-o",
            label="h_nat", color="#4C72B0")
    ax.plot(corr["流速_m_s"], corr["h_forced_W_m2K"], "-s",
            label="h_forced", color="#DD8452")
    ax.plot(corr["流速_m_s"], corr["h_mixed_W_m2K"], "-^",
            label="h_mixed", color="#55A868")
    ax.set_title("3 换热关联式：h 随流速变化")
    ax.set_xlabel("U (m/s)"); ax.set_ylabel("h (W/(m^2.K))")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ext = model_check["极限情景"]
    xpos = np.arange(len(ext))
    ax.bar(xpos - 0.16, ext["T_max_degC"], width=0.32,
           color="#C44E52", label="T_max")
    ax.bar(xpos + 0.16, ext["T_min_degC"], width=0.32,
           color="#4C72B0", label="T_min")
    ax.axhline(T_MAX, ls="--", color="black", lw=1, label="T_MAX")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["高温极值", "低温运行", "低温无产热"], fontsize=8)
    ax.set_title("4 极限情景温度")
    ax.set_ylabel("degC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sens = model_check["参数敏感性"]
    names = sens["扰动项"].iloc[1:].tolist()
    deltas = sens["T_max变化_K"].iloc[1:].to_numpy(float)
    yy = np.arange(len(names))
    colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in deltas]
    ax.barh(yy, deltas, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8)
    ax.set_title("5 参数敏感性：T_max 变化 (K)")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    per = model_check["周期稳定性"]
    plabels = per["指标"].tolist()
    pvals = per["数值_K"].to_numpy(float)
    ax.bar(plabels, np.maximum(pvals, 1e-9), color="#55A868")
    ax.set_yscale("log")
    for xi, v in zip(plabels, pvals):
        ax.text(xi, max(v, 1e-9) * 1.8, f"{v:.4f}", ha="center",
                fontsize=8)
    ax.set_title("6 周期稳定性：年际温差 (K)")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("时变传热模型检验（问题4）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    setup_chinese_font()
    print("工作区根目录（自动定位）:", WORKSPACE)
    print("数据源目录（仅从此读取）:", CLEAN_DIR)
    print()

    # ---------- 数据读取 ----------
    woa18 = load_woa18_monthly_sst()
    ersst = load_ersst_monthly_sst()
    tide_df = load_tide_2026()
    seawater = load_seawater_props_table()
    cp_map = load_metal_cp()
    currents = load_godas_currents()
    material_df = load_material_table()

    props_at = sea_props_interpolator(seawater)
    ctx = load_context(woa18, tide_df, currents, seawater, SITE,
                       U_TIDE_AMP_BASE)
    ctx["material_df"] = material_df
    ctx["cp_map"] = cp_map
    ctx["rho_sw"] = float(props_at(20.0)[0])
    ctx["props_at"] = props_at

    ersst_check = validate_season_with_ersst(ersst, ctx["sst_model"], SITE)
    ersst_check.to_csv(OUT_DIR / "结果_季节校核_ERSST.csv",
                       index=False, encoding="utf-8-sig")

    # ---------- 基准算例：20 degC 无潮汐 + 季节潮汐动态 ----------
    base_cfg = make_cfg(material_df[material_df["材料"] == "304 不锈钢"].index[0],
                        depth=30.0, wall=0.010, material_df=material_df,
                        cp_map=cp_map)
    print("=" * 76)
    print("第 10 步  基准算例一：20 degC 恒温、无海流（与问题 1 对比）")
    print("=" * 76)
    print("几何：D=%.2f m, L=%.2f m, 壁厚=%.4f m, 材料=%s, k=%.1f W/(m.K)"
          % (D_EXT, L_EXT, base_cfg["wall"], base_cfg["材料"],
             base_cfg["k_wall"]))
    print("壳体质量=%.1f kg, m*cp=%.3e J/K, 允许应力=%.3e Pa"
          % (base_cfg["m_shell"], base_cfg["mcp"], base_cfg["sigma_allow"]))
    env20 = {"t_hours": ctx["t_hours"],
             "T_inf": np.full(len(ctx["t_hours"]), 20.0),
             "U": np.zeros(len(ctx["t_hours"])),
             "h_forced": np.zeros(len(ctx["t_hours"])),
             "d_eff": np.full(len(ctx["t_hours"]), base_cfg["depth"]),
             "hn_grid": ctx["hn_grid"]}
    sim20, n20 = bisect_max_N(base_cfg, env20, props_at)
    print("20 degC 基准：二分最大 N = %d，全年 T_max=%.2f degC，"
          "Q_mean=%.0f W，Q波动=%.2f %%"
          % (n20, sim20["T_max"], sim20["Q_mean"], sim20["Q_fluct_pct"]))

    print()
    print("=" * 76)
    print("第 11 步  季节+潮汐动态基准（%s，深度 %.0f m，潮汐流速幅值 %.2f m/s）"
          % (SITE, base_cfg["depth"], ctx["U_TIDE_AMP"]))
    print("=" * 76)
    env = build_environment(ctx, base_cfg["depth"])
    sim, n_max = bisect_max_N(base_cfg, env, props_at)
    print("动态全年二分最大 N = %d（20 degC 基准 N=%d，变化 %+.1f %%）"
          % (n_max, n20, (n_max - n20) / n20 * 100 if n20 else 0))
    print("全年统计（丢弃前 168 h 初值瞬态）：")
    print("  T_max=%.2f degC, T_mean=%.2f degC" % (sim["T_max"], sim["T_mean"]))
    print("  Q_mean=%.0f W, Q_min=%.0f W, Q_max=%.0f W, 波动=%.1f %%"
          % (sim["Q_mean"], sim["Q_min"], sim["Q_max"], sim["Q_fluct_pct"]))
    print("  流态统计:", sim["regime_cnt"])
    print("  最热小时索引=%d（2026-01-01 01:00 起算）" % sim["worst_idx"])

    pd.DataFrame([
        {"工况": "20C基准", "N": n20, "T_max_degC": sim20["T_max"],
         "Q_mean_W": sim20["Q_mean"], "Q_fluct_pct": sim20["Q_fluct_pct"]},
        {"工况": "季节+潮汐", "N": n_max, "T_max_degC": sim["T_max"],
         "Q_mean_W": sim["Q_mean"], "Q_fluct_pct": sim["Q_fluct_pct"]},
    ]).to_csv(OUT_DIR / "结果_基准算例.csv", index=False, encoding="utf-8-sig")

    # ---------- 季节月度表 ----------
    monthly = monthly_season_table(base_cfg, woa18, ctx, props_at)
    monthly.to_csv(OUT_DIR / "结果_季节月度.csv", index=False, encoding="utf-8-sig")
    plot_season(monthly, OUT_DIR / "图1_季节曲线.png")

    # ---------- 潮汐调和与大小潮 ----------
    tidal, tide_sum = tidal_harmonic_analysis(tide_df)
    tidal.to_csv(OUT_DIR / "结果_潮汐调和分析.csv", index=False,
                 encoding="utf-8-sig")
    daily, sn = spring_neap_days(tide_df)
    daily.to_csv(OUT_DIR / "结果_逐日潮差.csv", index=False, encoding="utf-8-sig")
    plot_tide(tide_df, tidal, daily, tide_sum, OUT_DIR / "图2_潮汐调和分析.png")

    # ---------- 大小潮逐时序列 ----------
    tide_t0 = pd.Timestamp("2026-01-01 01:00:00")
    spring_h = int((pd.Timestamp(sn["spring"]["date"])
                    + pd.Timedelta(hours=12) - tide_t0
                    ).total_seconds() // 3600)
    neap_h = int((pd.Timestamp(sn["neap_near_spring"]["date"])
                  + pd.Timedelta(hours=12) - tide_t0
                  ).total_seconds() // 3600)
    hour_df = pd.DataFrame({
        "hour": np.arange(len(env["T_inf"])),
        "datetime": ctx["tide_dt"],
        "T_inf_degC": env["T_inf"], "U_m_s": env["U"],
        "d_eff_m": env["d_eff"],
        "T_shell_degC": sim["T_arr"], "Q_W": sim["Q_arr"],
        "Q_eff_W": sim["Q_eff_arr"], "T_wo_degC": sim["t_wo_arr"],
        "Gr_Re2": sim["gr_re2_arr"],
        "Q_cap_W": sim["Q_cap_arr"],
        "h_nat_W_m2K": sim["h_nat_arr"], "h_forced_W_m2K": sim["h_forced_arr"],
        "h_mixed_W_m2K": sim["h_mix_arr"],
    })
    hour_df.to_csv(OUT_DIR / "结果_全年逐时.csv", index=False, encoding="utf-8-sig")
    sl_spring = slice(max(spring_h - 24, 0), spring_h + 24 * 6)
    sl_neap = slice(max(neap_h - 24, 0), neap_h + 24 * 6)
    spring_df = hour_df.iloc[sl_spring].copy()
    spring_df.insert(0, "阶段", "大潮")
    neap_df = hour_df.iloc[sl_neap].copy()
    neap_df.insert(0, "阶段", "小潮")
    pd.concat([spring_df, neap_df], ignore_index=True).to_csv(
        OUT_DIR / "结果_大小潮逐时.csv", index=False, encoding="utf-8-sig")
    print("第 12 步  大潮/小潮逐时序列：")
    print("  大潮周 Q_mean=%.0f W，小潮周 Q_mean=%.0f W，"
          "潮汐强迫对流增益=%.1f %%"
          % (spring_df["Q_W"].mean(), neap_df["Q_W"].mean(),
             (spring_df["Q_W"].mean() / neap_df["Q_W"].mean() - 1.0) * 100.0))
    print("  极限散热能力 Q_cap（壳温=80 degC）：大潮=%.0f W，"
          "小潮=%.0f W，大小潮差=%.1f %%"
          % (spring_df["Q_cap_W"].mean(), neap_df["Q_cap_W"].mean(),
             (spring_df["Q_cap_W"].mean() / neap_df["Q_cap_W"].mean() - 1.0)
             * 100.0))
    print("  [说明] 大小潮对比取同一季节相邻潮周：大潮 %s vs 小潮 %s，"
          "避免季节温差掩盖潮汐效应。"
          % (sn["spring"]["date"], sn["neap_near_spring"]["date"]))

    # ---------- 最不利工况 ----------
    worst_t = hour_df.loc[sim["worst_idx"]]
    qmin_idx = int(np.argmin(sim["Q_arr"][168:])) + 168
    qmin_t = hour_df.loc[qmin_idx]
    winter_neap = hour_df[(hour_df["datetime"].dt.month.isin([1, 2, 12]))
                          & (hour_df["datetime"].dt.date
                             == pd.Timestamp(sn["neap"]["date"]).date())]
    winter_neap_mean_q = winter_neap["Q_W"].mean() if len(winter_neap) else np.nan
    winter_neap_mean_qcap = (winter_neap["Q_cap_W"].mean()
                             if len(winter_neap) else np.nan)
    worst_df = pd.DataFrame([
        {"工况": "全年最热（数据识别）",
         "时刻": worst_t["datetime"], "T_shell_degC": worst_t["T_shell_degC"],
         "Q_W": worst_t["Q_W"], "Q_cap_W": worst_t["Q_cap_W"],
         "T_inf_degC": worst_t["T_inf_degC"],
         "U_m_s": worst_t["U_m_s"], "h_mixed_W_m2K": worst_t["h_mixed_W_m2K"]},
        {"工况": "全年散热最低",
         "时刻": qmin_t["datetime"], "T_shell_degC": qmin_t["T_shell_degC"],
         "Q_W": qmin_t["Q_W"], "Q_cap_W": qmin_t["Q_cap_W"],
         "T_inf_degC": qmin_t["T_inf_degC"],
         "U_m_s": qmin_t["U_m_s"], "h_mixed_W_m2K": qmin_t["h_mixed_W_m2K"]},
        {"工况": "冬季小潮（用户示例，最浅浸没）",
         "时刻": winter_neap["datetime"].min() if len(winter_neap) else np.nan,
         "T_shell_degC": winter_neap["T_shell_degC"].mean() if len(winter_neap) else np.nan,
         "Q_W": winter_neap_mean_q if not np.isnan(winter_neap_mean_q) else np.nan,
         "Q_cap_W": (winter_neap_mean_qcap
                     if not np.isnan(winter_neap_mean_qcap) else np.nan),
         "T_inf_degC": winter_neap["T_inf_degC"].mean() if len(winter_neap) else np.nan,
         "U_m_s": winter_neap["U_m_s"].mean() if len(winter_neap) else np.nan,
         "h_mixed_W_m2K": winter_neap["h_mixed_W_m2K"].mean() if len(winter_neap) else np.nan},
    ])
    worst_df.to_csv(OUT_DIR / "结果_最不利工况.csv", index=False,
                    encoding="utf-8-sig")
    print("第 13 步  最不利工况：")
    print(worst_df.to_string(index=False))
    print("  全年 Q 波动 = %.1f %%；季节表 Q 极差 = %.1f %%"
          % (sim["Q_fluct_pct"],
             (monthly[monthly["月份"] != "20C基准"]["散热能力_Q_W"].max()
              - monthly[monthly["月份"] != "20C基准"]["散热能力_Q_W"].min())
             / monthly[monthly["月份"] != "20C基准"]["散热能力_Q_W"].mean() * 100))
    print("  全年极限散热能力 Q_cap：均值=%.0f W，波动=%.1f %%"
          % (sim["Q_cap_mean"],
             (sim["Q_cap_max"] - sim["Q_cap_min"]) / sim["Q_cap_mean"] * 100.0))
    plot_spring_neap(env, sim, spring_h, neap_h,
                     OUT_DIR / "图3_大小潮逐时序列.png")
    plot_worst(env, sim, ctx, OUT_DIR / "图4_最不利工况.png")

    # ---------- 灵敏度分析 ----------
    sens = sensitivity_analysis(base_cfg, ctx, props_at)
    sens.to_csv(OUT_DIR / "结果_灵敏度.csv", index=False, encoding="utf-8-sig")
    plot_sensitivity(sens, OUT_DIR / "图7_灵敏度.png")

    # ---------- 时变传热模型检验 ----------
    model_check = validate_time_varying_model(base_cfg, env, sim, n_max,
                                              ctx, props_at)
    model_check["汇总"].to_csv(OUT_DIR / "结果_时变模型检验.csv",
                               index=False, encoding="utf-8-sig")
    for suffix, df in model_check.items():
        if suffix == "汇总":
            continue
        df.to_csv(OUT_DIR / ("结果_模型检验_%s.csv" % suffix),
                  index=False, encoding="utf-8-sig")
    plot_model_validation(base_cfg, env, sim, model_check,
                          OUT_DIR / "图8_时变传热模型检验.png")

    # ---------- NSGA-II 嵌套 RK4 联合优化 ----------
    print("=" * 76)
    print("第 14 步  NSGA-II 嵌套 RK4 联合优化（变量：材料/深度/壁厚）")
    print("=" * 76)
    import sys
    workers = int(os.environ.get("Q4_WORKERS", "4"))
    if "--skip-opt" in sys.argv:
        print("检测到 --skip-opt，跳过 NSGA-II（保留基准/季节/潮汐结果）。")
        return
    pop, pareto, history = nsga2(ctx, pop_size=NSGA_POP, generations=NSGA_GEN,
                                 seed=NSGA_SEED, workers=workers)
    if not pareto:
        print("[警告] 无可行的帕累托解，请检查约束与 RK4 仿真。")
        return
    pdf = pd.DataFrame([{
        "材料": r["材料"], "depth_m": r["depth"], "wall_m": r["wall"],
        "N": r["N"], "Q_mean_W": r["Q_mean"], "Q_cap_mean_W": r["Q_cap_mean"],
        "cost_元": r["cost"], "life_年": r["life"], "T_max_degC": r["T_max"],
        "t_req_m": r["t_req"], "viol": r["viol"],
    } for r in pareto])
    pdf = pdf.drop_duplicates(subset=["Q_mean_W", "cost_元", "life_年"]).reset_index(drop=True)
    pdf = pdf.sort_values(["Q_mean_W", "cost_元", "life_年"],
                          ascending=[False, True, False]).reset_index(drop=True)
    pdf.to_csv(OUT_DIR / "结果_NSGA2_帕累托.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history, columns=["gen", "front_size", "best_Q_mean",
                                   "mean_cost"]).to_csv(
        OUT_DIR / "结果_NSGA2_历史.csv", index=False, encoding="utf-8-sig")
    print("可行帕累托解 %d 个，前 8 个按 Q_mean 排序：" % len(pdf))
    print(pdf.head(8).to_string(index=False))
    plot_pareto(pdf, history, OUT_DIR)

    print()
    print("输出目录：", OUT_DIR)
    print("[最终提醒] 潮位数据为天文潮预报，不含风暴潮与余水位；"
          "珠海/陵水潮位为香港赤鱲角东代理站。")


if __name__ == "__main__":
    main()
