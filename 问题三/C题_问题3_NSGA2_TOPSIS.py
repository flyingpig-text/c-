# -*- coding: utf-8 -*-
"""
C 题 问题 3：材料 + 海水深度 多目标优化
=================================================================
方法：NSGA-II（手写）求帕累托前沿 + TOPSIS（手写）决策
   目标：在问题 2 最优结构（长方体，每面 160 根翅片，Hf=5mm，df=1mm）
          基础上，优化 材料 / 海水深度 / 壁厚，使
          f1  散热能力最大化（Q，W；题目口径“散热效果/更多服务器”，
              N 由 Q 与空间上限导出并单独输出）
          f2  总成本最小化（元）
          f3  使用寿命最大化（年）
约束：承压强度（t >= 所需壁厚）、使用寿命 >= 10 年、
      散热能力 >= 1 台服务器、内部空间 >= 0.05 m。

统一规范（本版本新增）：
    1) 全部物理量使用国际单位制（Pa、W、m、kg/m^3、W/(m·K)），
       打印时标注单位，并对关键量做数量级校验；
    2) 海水温度剖面、金属导热系数、海洋材料力学性能来自
       C题数据/清洗后数据；价格、腐蚀速率、安全系数允许使用
       公开补充数据或工程默认值（代码中已标注）；
    3) 先运行一个简单基准算例（固定材料/深度/壁厚）验证模型量级，
       再进入 NSGA-II 扩展优化；
    4) 每个模型结束时做灵敏度分析：
       温度回归模型做 ±5%/±10% 扰动灵敏度，优化模型做连续参数与材料离散灵敏度。

模型检验：
    1) 温度-深度回归：R^2、调整 R^2、残差分析；
    2) 5 折交叉验证（手写 KFold）；
    3) 温度模型扰动灵敏度；
    4) NSGA-II 收敛过程记录；
    5) 优化模型灵敏度分析（连续参数 ±10% + 材料离散灵敏度）。

依赖：仅 numpy / pandas / matplotlib（不引入 scipy、pymoo、deap 等）。
运行：python C题_问题3_NSGA2_TOPSIS.py
"""

from __future__ import annotations

import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "输出"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# matplotlib 缓存写入工作区，避免系统用户目录无写权限
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ==================================================================
# 0. 环境准备：中文字体与输出目录
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


def _find_workspace_root() -> Path:
    """从脚本所在目录逐级向上自动寻找工作区根目录（相对路径方式）。

    判定标准：根目录下存在 C题数据/清洗后数据 子目录。
    这样脚本无论放在 问题三/ 还是 数学建模/ 下都能自动定位。
    """
    here = Path(__file__).resolve().parent
    for root in (here, *here.parents):
        if (root / "C题数据" / "清洗后数据").is_dir():
            return root
    raise FileNotFoundError("自动查找失败：未找到 C题数据/清洗后数据 目录。")


WORKSPACE = _find_workspace_root()
DATA_DIR = WORKSPACE / "C题数据" / "清洗后数据"

# 数据源只允许指向用户指定的清洗后目录（绝对路径兜底校验，防走错目录）
_CANONICAL_DATA_DIR = Path(r"D:/46884/Documents/数学建模/C题数据/清洗后数据").resolve()
if os.path.normcase(str(DATA_DIR.resolve())) != os.path.normcase(str(_CANONICAL_DATA_DIR)):
    raise RuntimeError("自动定位到的数据目录与要求目录不一致：%s" % DATA_DIR)
print("工作区根目录（自动定位）：", WORKSPACE)
print("数据源目录（仅从此目录读取）：", DATA_DIR)


# ==================================================================
# 0.5 通用基础参数来源核对：只从交付清单 + 题面取值，禁止编造
# ==================================================================
def _find_delivery_checklist() -> Path:
    """自动向上定位《水下服务器热设计参数与算法交付清单.docx》。"""
    for root in (HERE, *HERE.parents):
        f = root / "附件" / "水下服务器热设计参数与算法交付清单.docx"
        if f.is_file():
            return f
    raise FileNotFoundError("未找到《水下服务器热设计参数与算法交付清单.docx》。")


def extract_docx_paragraphs(path: Path) -> list[str]:
    """用标准库 zipfile/XML 提取 docx 正文与表格文本，不依赖 python-docx。"""
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    texts: list[str] = []
    for node in root.iter(w_ns + "p"):
        txt = "".join(t.text or "" for t in node.iter(w_ns + "t"))
        if txt.strip():
            texts.append(txt.strip())
    for tbl in root.iter(w_ns + "tbl"):
        for row in tbl.iter(w_ns + "tr"):
            cells = ["".join(t.text or "" for t in cell.iter(w_ns + "t"))
                     for cell in row.iter(w_ns + "tc")]
            row_txt = " | ".join(c.strip() for c in cells if c.strip())
            if row_txt:
                texts.append(row_txt)
    return texts


def verify_delivery_checklist() -> None:
    """打印：基础参数与清单条目的对应关系 + 公式实现核对。"""
    checklist_path = _find_delivery_checklist()
    paras = extract_docx_paragraphs(checklist_path)
    full_text = "\n".join(paras)

    print("=" * 76)
    print("第 0.5 步  通用基础参数来源核对（交付清单）")
    print("=" * 76)
    print("清单文件（自动定位）：", checklist_path)

    # (代码参数, 数值+单位, 清单核对关键词, 数值来源说明)
    param_rows = [
        ("ENVELOPE_SIDE", "1.0 m", "外形约束长方体上限", "C题.pdf 题面"),
        ("HULL_LENGTH", "12.0 m", "长度", "C题.pdf 题面"),
        ("Q0", "500.0 W", "单台产热", "C题.pdf 题面"),
        ("T_MAX", "80.0 ℃", "最高允许温度", "C题.pdf 题面"),
        ("SERVER_W/H/L", "0.4826/0.04445/0.525 m",
         "1U 服务器尺寸", "C题.pdf 题面"),
        ("RHO_SW", "1025.0 kg/m^3", "20℃海水密度",
         "数据.md 工程常用值【清单未给数值】"),
        ("G", "9.81 m/s^2", "格拉晓夫数", "物理常数"),
        ("DEPTH_MIN/MAX", "5 / 100 m", "海域安全水深上下限",
         "陵水站数据范围【清单未给数值】"),
        ("WALL_MIN/MAX", "0.004 / 0.050 m", "壁厚下限",
         "问题2默认区间【工程默认】"),
        ("SAFETY_FACTOR", "2.5", "安全余量",
         "工程默认（用户已确认允许）"),
        ("MIN_LIFE / LIFE_CAP", "10 / 50 年", "设计使用寿命基准",
         "工程默认（用户已确认允许）"),
        ("COATING_PRICE", "150.0 元/m^2", "各类海洋金属单价",
         "工程默认（用户已确认允许）"),
        ("GA_POP/GEN/PC/PM", "80/120/0.9/0.1",
         "算法参数",
         "工程默认【清单未给数值】"),
        ("TOPSIS_W", "0.40/0.30/0.30", "权重设定",
         "工程默认【清单未给数值】"),
        ("NF_PER_FACE/HF/DF", "160 / 5mm / 1mm",
         "翅片参数", "问题2 GA+SLSQP 最优结构"),
    ]
    for name, value, key, source in param_rows:
        ok = key in full_text
        flag = "[OK]  " if ok else "[WARN]"
        print("  %s %-18s %-10s 清单项「%s」 来源：%s"
              % (flag, name, value, key, source))
        if not ok:
            print("       清单文本中未检索到关键词「%s」，请人工核对。" % key)

    print()
    print("统一公式实现核对（与清单「统一传热核心公式」对应）：")
    formula_rows = [
        ("Q = h * A_eff * (T_max - T_sea)", "牛顿冷却散热总公式"),
        ("Ra/Pr/Nu -> h（Churchill-Chu）", "自然对流准则关联式"),
        ("eta_f = tanh(m*Hf)/(m*Hf)", "翅片效率计算公式"),
        ("t_req = p*D/(2*sigma_allow)", "耐压壳体壁厚计算公式"),
        ("life = 腐蚀余量 / 腐蚀速率", "腐蚀寿命计算公式"),
    ]
    for formula, key in formula_rows:
        ok = key in full_text
        print("  [%s] %-38s <- %s" % ("OK" if ok else "WARN", formula, key))
    print()


# ==================================================================
# 1. 题目给定参数（与问题 1/2 保持一致）
# ==================================================================
ENVELOPE_SIDE = 1.0        # 外形约束：横截面不超过 1 m
HULL_LENGTH = 12.0         # 机柜长度 L，m
Q0 = 500.0                 # 单台服务器产热，W
T_MAX = 80.0               # 内部允许最高温度，℃
SERVER_W, SERVER_H, SERVER_L = 0.4826, 0.04445, 0.525   # 1U 服务器，m

# ---- 问题 2 最优结构（已由问题 2 GA+SLSQP 求得）----
NF_PER_FACE = 160          # 每个侧面翅片根数
FIN_H = 0.0050             # 翅高，m
FIN_D = 0.0010             # 翅厚，m

# ---- 问题 3 新增变量范围 ----
DEPTH_MIN, DEPTH_MAX = 5.0, 100.0      # 海水深度，m（陵水站数据到 100 m）
WALL_MIN, WALL_MAX = 0.004, 0.050      # 壳体壁厚，m
SAFETY_FACTOR = 2.5                    # 承压安全系数【假设值】
RHO_SW = 1025.0                        # 海水密度，kg/m^3【近似】
G = 9.81                               # 重力加速度，m/s^2
MIN_LIFE = 10.0                        # 最低使用寿命要求，年【假设值】
LIFE_CAP = 50.0                        # 寿命上限（设计寿命），年
COATING_PRICE = 150.0                  # 防腐涂层单价，元/m^2【假设值】

# ---- NSGA-II 超参数（标准设置，可复现）----
GA_POP = 80
GA_GEN = 120
GA_PC = 0.90
GA_PM = 0.10
ETA_C = 15.0             # SBX 交叉分布指数
ETA_M = 20.0             # 多项式变异分布指数
GA_SEED = 20260813

# ---- TOPSIS 权重（散热 / 成本 / 寿命）----
TOPSIS_W = np.array([0.40, 0.30, 0.30])


# ==================================================================
# 2. 数据读取与预处理：南海温度剖面（陵水站）
# ==================================================================
def load_temperature_profile() -> pd.DataFrame:
    """读取清洗后 WOA18 温度剖面，做缺失值 / 重复 / 异常值检查。"""
    f = DATA_DIR / "WOA18_南海温度剖面_clean.csv"
    df = pd.read_csv(f)
    df = df[df["站点"] == "陵水"].reset_index(drop=True)
    df = df[["深度_m", "温度_degC"]].copy()
    df.columns = ["depth_m", "temp_C"]

    print("=" * 76)
    print("第 1 步  数据预处理（温度-深度剖面，站点=陵水）")
    print("=" * 76)
    print("原始 shape：", df.shape)
    print("前 8 行：")
    print(df.head(8).to_string(index=False))

    # 缺失值检查
    missing = df.isna().sum().to_dict()
    print("缺失值统计：", missing)
    if missing["depth_m"] or missing["temp_C"]:
        df["temp_C"] = df["temp_C"].interpolate(method="linear")
        df = df.dropna().reset_index(drop=True)
        print("已用线性插值填补缺失值，处理后 shape：", df.shape)

    # 重复值检查
    dup = df.duplicated(subset=["depth_m"]).sum()
    print("重复深度行数：", int(dup))
    df = df.drop_duplicates(subset=["depth_m"], keep="last").reset_index(drop=True)

    # 异常值检查（IQR 法，按深度分层近似处理）
    q1, q3 = df["temp_C"].quantile(0.25), df["temp_C"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    bad = df[(df["temp_C"] < lo) | (df["temp_C"] > hi)]
    print("IQR 异常值检查：Q1=%.3f Q3=%.3f IQR=%.3f，异常行数=%d"
          % (q1, q3, iqr, len(bad)))
    if len(bad):
        print(bad.to_string(index=False))
        df = df[(df["temp_C"] >= lo) & (df["temp_C"] <= hi)].reset_index(drop=True)
        print("已剔除异常值，处理后 shape：", df.shape)

    print("处理后统计：")
    print(df.describe().to_string())
    print()
    return df


# ==================================================================
# 3. 温度-深度回归模型 + 模型检验（R^2 / 残差 / 5 折交叉验证）
# ==================================================================
def fit_temperature_model(df: pd.DataFrame, degree: int = 2):
    """最小二乘多项式拟合 T(d)，返回模型对象与检验结果。

    为避免深度量级差异影响数值稳定性，先用 z-score 标准化深度，
    再拟合二次多项式；预测时同样先标准化。
    """
    d = df["depth_m"].to_numpy(dtype=float)
    t = df["temp_C"].to_numpy(dtype=float)
    d_mean, d_std = d.mean(), d.std()
    x = (d - d_mean) / d_std

    coef = np.polyfit(x, t, deg=degree)          # 从高次到常数
    pred = np.polyval(coef, x)
    resid = t - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    n, p = len(t), degree + 1
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)

    print("=" * 76)
    print("第 2 步  温度-深度回归模型与模型检验")
    print("=" * 76)
    print("模型：T(d) = a2*x^2 + a1*x + a0，x = (d-%.4f)/%.4f"
          % (d_mean, d_std))
    print("回归系数（a2, a1, a0）：", np.round(coef, 6))
    print("R^2 = %.6f，调整 R^2 = %.6f" % (r2, r2_adj))
    print("残差统计：均值=%.4f，标准差=%.4f，最大绝对残差=%.4f ℃"
          % (resid.mean(), resid.std(), np.abs(resid).max()))

    # 残差相关（一阶自相关，Durbin-Watson 简化版）
    dw = float(np.sum(np.diff(resid) ** 2) / max(ss_res, 1e-12))
    print("残差一阶自相关 DW ≈ %.3f（接近 2 表示无明显自相关）" % dw)
    print()

    model = {
        "coef": coef, "d_mean": d_mean, "d_std": d_std,
        "pred": pred, "resid": resid, "r2": r2, "r2_adj": r2_adj,
        "dw": dw, "df": df,
    }
    return model


def cross_validate_temperature(df: pd.DataFrame, k: int = 5,
                               degree: int = 2, seed: int = 1):
    """手写 k 折交叉验证：每折用训练集拟合、测试集算 R^2。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    folds = np.array_split(idx, k)
    r2_list = []
    d = df["depth_m"].to_numpy(dtype=float)
    t = df["temp_C"].to_numpy(dtype=float)
    d_mean, d_std = d.mean(), d.std()

    for fi in range(k):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fi])
        x_tr = (d[train_idx] - d_mean) / d_std
        coef = np.polyfit(x_tr, t[train_idx], deg=degree)
        x_te = (d[test_idx] - d_mean) / d_std
        pred = np.polyval(coef, x_te)
        ss_res = float(np.sum((t[test_idx] - pred) ** 2))
        ss_tot = float(np.sum((t[test_idx] - t[test_idx].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        r2_list.append(r2)

    r2_arr = np.array(r2_list)
    print("5 折交叉验证 R^2：", np.round(r2_arr, 4))
    print("平均 R^2 = %.4f ± %.4f" % (r2_arr.mean(), r2_arr.std()))
    print()
    return r2_arr


def sensitivity_temperature_model(df: pd.DataFrame, model: dict,
                                  seed: int = 3) -> pd.DataFrame:
    """温度-深度回归模型的扰动灵敏度分析（每个模型结束时必须做）。"""
    rng = np.random.default_rng(seed)
    d = df["depth_m"].to_numpy(dtype=float)
    t = df["temp_C"].to_numpy(dtype=float)
    t_base_50 = float(np.polyval(
        model["coef"], (50.0 - model["d_mean"]) / model["d_std"]))

    print("=" * 76)
    print("第 3 步  温度-深度回归模型灵敏度分析")
    print("=" * 76)
    print("基准预测 T(50 m) = %.4f ℃" % t_base_50)
    print("扰动方式：对观测温度/深度做 ±5%、±10% 系统扰动，重新拟合二次多项式。")
    print("物理含义：若回归系数与预测值变化小，说明模型对输入数据扰动稳健。")

    cases = [
        ("温度观测 +5%", t * 1.05, d),
        ("温度观测 -5%", t * 0.95, d),
        ("深度 +10%", t, d * 1.10),
        ("深度 -10%", t, d * 0.90),
        ("随机噪声 ±0.3 ℃", t + rng.normal(0.0, 0.3, size=len(t)), d),
    ]
    rows = []
    for name, t2, d2 in cases:
        dm, ds = float(d2.mean()), float(d2.std())
        x = (d2 - dm) / ds
        coef = np.polyfit(x, t2, 2)
        pred = np.polyval(coef, x)
        ss_res = float(np.sum((t2 - pred) ** 2))
        ss_tot = float(np.sum((t2 - t2.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        t50 = float(np.polyval(coef, (50.0 - dm) / ds))
        rows.append({
            "扰动项": name,
            "R2": r2,
            "a2": coef[0], "a1": coef[1], "a0": coef[2],
            "T50m_pred": t50,
            "T50m变化_pct": (t50 - t_base_50) / t_base_50 * 100.0,
        })
    sens = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: "%.6g" % v)
    print(sens.to_string(index=False))
    max_drift = np.abs(sens["T50m变化_pct"]).max()
    print("T(50 m) 最大相对漂移 = %.2f%%（<5%% 视为稳健）" % max_drift)
    print()
    return sens


def make_temp_predictor(model: dict):
    """返回深度 -> 海水温度 的预测函数（含上下限保护）。"""
    def predict(depth):
        d = np.asarray(depth, dtype=float)
        x = (d - model["d_mean"]) / model["d_std"]
        val = np.polyval(model["coef"], x)
        return float(np.clip(val, 5.0, 30.0))
    return predict


# ==================================================================
# 4. 候选海洋材料表：力学性能只从清洗后数据包读取，禁止编造
# ==================================================================
LB_IN3_TO_KG_M3 = 27679.904710203  # 密度单位换算：lb/in^3 -> kg/m^3
KSI_TO_MPA = 6.894757293168361     # 强度单位换算：ksi -> MPa


def lookup_k_from_csv(material: str, temp_C: float = 20.0) -> float | None:
    """从清洗后的金属导热系数 CSV 中精确匹配材料名取 20 ℃ 附近值。"""
    f = DATA_DIR / "金属导热系数_EngineeringToolbox_clean.csv"
    if not f.exists():
        return None
    dfm = pd.read_csv(f)
    dfm["material"] = dfm["material"].astype(str).str.strip().str.lower()
    key = material.lower()
    hits = dfm[dfm["material"] == key]
    if hits.empty:
        hits = dfm[dfm["material"].str.startswith(key + ",")]
    if hits.empty:
        hits = dfm[dfm["material"].str.contains(key, na=False)]
    hits = hits[hits["temperature_C"].between(temp_C - 25.0, temp_C + 25.0)]
    if hits.empty:
        return None
    # 温度越接近 20 ℃ 优先级越高
    hits = hits.assign(dist=(hits["temperature_C"] - temp_C).abs())
    hits = hits.sort_values("dist")
    return float(hits.iloc[0]["thermal_conductivity_W_per_mK"])


def lookup_mech_from_attachment(material_key: str) -> tuple[float, float]:
    """从《海洋材料性能_C题附件_clean.csv》读取密度与屈服强度。

    返回：(密度 kg/m^3, 屈服强度 MPa)。找不到时抛错，保证不编造数据。
    屈服强度取附件“代表值”列中的第一个数值：
        * 6061 取 T6 态 40 ksi（牌号 6061-T6）；
        * 316 取退火范围下界 30 ksi（保守）；
        * 其余材料取附件列出的状态值。
    """
    f = DATA_DIR / "海洋材料性能_C题附件_clean.csv"
    dfm = pd.read_csv(f)
    dfm["材料"] = dfm["材料"].astype(str).str.strip().str.lower()
    key = material_key.lower()
    hits = dfm[dfm["材料"] == key]
    if hits.empty:
        raise ValueError("附件材料表中未找到：%s（请核对材料名）" % material_key)
    row = hits.iloc[0]
    dens_lb = float(row["密度"])           # lb/in^3
    rep_text = str(row["屈服强度_ksi"])     # 例如 "40 (T6)"、"30 - 42 (Annealed)"
    m = re.search(r"[-+]?\d*\.?\d+", rep_text)
    rep_ksi = float(m.group(0)) if m else float(row["屈服强度_min_ksi"])
    return dens_lb * LB_IN3_TO_KG_M3, rep_ksi * KSI_TO_MPA


def build_material_table() -> pd.DataFrame:
    """构造候选材料表。

    数据来源（物理数据严格限定清洗后目录）：
        * 密度、屈服强度：海洋材料性能_C题附件_clean.csv
        * 导热系数：金属导热系数_EngineeringToolbox_clean.csv
        * 单价、腐蚀速率、安全系数：允许使用公开补充数据或工程默认值，
          论文中需注明来源。
    """
    # (材料名, 牌号, 附件表材料名, 导热系数csv关键词)
    raw = [
        ("6061 铝合金", "6061-T6", "Aluminum alloy 6061", None),
        ("304 不锈钢", "304", "304 Stainless Steel",
         "steel - stainless, type 304"),
        ("316L 不锈钢", "316L", "316 Stainless Steel", None),
        ("TC4 钛合金", "TC4", "Titanium 6Al-4V", None),
        ("AISI 1040 碳钢", "1040", "AISI 1040 Steel",
         "steel - carbon, 0.5% c"),
        ("T2 紫铜", "T2", "Copper", "copper"),
    ]
    rows = []
    for name, brand, attach_key, k_csv_key in raw:
        rho, sigma_y = lookup_mech_from_attachment(attach_key)
        k = lookup_k_from_csv(k_csv_key) if k_csv_key else None
        rows.append((name, brand, rho, sigma_y, k, k_csv_key))

    dfm = pd.DataFrame(rows, columns=[
        "材料", "牌号", "密度_kg_m3", "屈服强度_MPa",
        "导热系数_W_mK", "csv_key"])

    # 导热系数缺失项：316L 沿用 304（同一系列），6061/TC4 无精确条目时
    # 使用数据.md 引用的公开手册值（Al 6061≈167，Ti≈7.5），打印警示
    default_k = {"6061 铝合金": 167.0, "304 不锈钢": 14.4,
                 "316L 不锈钢": 14.4, "TC4 钛合金": 7.5,
                 "AISI 1040 碳钢": 54.0, "T2 紫铜": 401.0}
    for i, row in dfm.iterrows():
        if pd.isna(row["导热系数_W_mK"]):
            dfm.loc[i, "导热系数_W_mK"] = default_k[row["材料"]]

    # 单价 / 腐蚀速率：数据包无此数值，以下为工程默认（用户已确认允许）
    price_assume = {"6061 铝合金": 22000.0, "304 不锈钢": 15000.0,
                    "316L 不锈钢": 22000.0, "TC4 钛合金": 320000.0,
                    "AISI 1040 碳钢": 5200.0, "T2 紫铜": 68000.0}
    corr_assume = {"6061 铝合金": 0.150, "304 不锈钢": 0.020,
                   "316L 不锈钢": 0.010, "TC4 钛合金": 0.001,
                   "AISI 1040 碳钢": 0.300, "T2 紫铜": 0.080}
    dfm["价格_元_吨"] = dfm["材料"].map(price_assume)
    dfm["腐蚀速率_mm_年"] = dfm["材料"].map(corr_assume)

    print("=" * 76)
    print("第 4 步  候选海洋材料表（密度/屈服强度来自附件清洗表）")
    print("=" * 76)
    print("缺失值检查：", dfm.isna().sum().to_dict())
    print("shape：", dfm.shape)
    print(dfm.to_string(index=False))
    print("说明：")
    print("  * 密度、屈服强度：海洋材料性能_C题附件_clean.csv（lb/in^3、ksi 已换算为 SI）；")
    print("  * 导热系数：金属导热系数_EngineeringToolbox_clean.csv；缺失项用公开手册值并已标注；")
    print("  * Q235 未收录于数据包，本表用附件中的 AISI 1040 碳钢代替；")
    print("  * 价格/腐蚀速率/安全系数采用工程默认值（用户已确认允许），论文中需注明来源。")
    print()
    return dfm.reset_index(drop=True)


# ==================================================================
# 5. 热物性与换热模型（与问题 1/2 同一套关联式）
# ==================================================================
def sea_props(T: float) -> dict:
    rho = 1027.0 - 0.24 * (T - 20.0)
    cp = 3985.0 + 0.35 * T
    k = 0.575 + 0.0016 * T
    mu = 0.00108 * np.exp(-0.019 * (T - 20.0))
    return {"rho": rho, "cp": cp, "k": k, "mu": mu, "beta": 2.5e-4}


def air_props(T: float) -> dict:
    tk = T + 273.15
    rho = 101325.0 / (287.06 * tk)
    return {"rho": rho, "cp": 1006.0, "k": 0.02439 + 0.0000792 * T,
            "mu": 1.72e-5 + 5.0e-8 * T, "beta": 1.0 / tk}


def h_horizontal_cylinder(D: float, dT: float, T_film: float, props_fn) -> float:
    """水平圆柱/方柱自然对流 Churchill-Chu 关联式，W/(m^2·K)。"""
    p = props_fn(T_film)
    nu = p["mu"] / p["rho"]
    alpha = p["k"] / (p["rho"] * p["cp"])
    pr = nu / alpha
    ra = G * p["beta"] * dT * D ** 3 / (nu * alpha)
    denom = (1.0 + (0.559 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    nu_corr = (0.60 + 0.387 * ra ** (1.0 / 6.0) / denom) ** 2
    return nu_corr * p["k"] / D


# ==================================================================
# 6. 问题 3 目标函数 / 约束求值（一个设计 -> 全部指标）
# ==================================================================
def evaluate_design(mat_idx: int, depth: float, wall: float,
                    material_df: pd.DataFrame, temp_pred) -> dict:
    """计算一个设计方案的全部物理量。

    决策变量：
        mat_idx : 材料索引（离散）
        depth   : 海水深度，m
        wall    : 壳体壁厚，m
    返回：dict，含 Q、成本、寿命、可行性与约束违和量。
    """
    mat = material_df.iloc[int(mat_idx)]
    k_mat = float(mat["导热系数_W_mK"])
    rho_mat = float(mat["密度_kg_m3"])
    sigma_y = float(mat["屈服强度_MPa"]) * 1e6
    price = float(mat["价格_元_吨"])
    corr = float(mat["腐蚀速率_mm_年"])

    depth = float(depth)
    wall = float(wall)
    t_sea = temp_pred(depth)

    # ---- 承压强度：薄壁压力容器公式（近似，D 取 1 m）----
    p_hydro = RHO_SW * G * depth
    sigma_allow = sigma_y / SAFETY_FACTOR
    t_req = p_hydro * ENVELOPE_SIDE / (2.0 * sigma_allow)

    # ---- 几何（问题 2 最优结构：长方体 + 每面 160 根翅）----
    a = ENVELOPE_SIDE - 2.0 * FIN_H          # 基体外边长
    a_in = a - 2.0 * wall
    l_in = HULL_LENGTH - 2.0 * wall
    if a_in <= 0.0 or l_in <= 0.0:
        return _bad_result()

    a_in_side = 4.0 * a_in * l_in + 2.0 * a_in ** 2
    a_out_base = 4.0 * a * HULL_LENGTH + 2.0 * a ** 2
    total_fins = NF_PER_FACE * 4
    a_fin_one = (2.0 * FIN_H + FIN_D) * HULL_LENGTH
    a_base = a_out_base - total_fins * FIN_D * HULL_LENGTH

    # ---- 换热模型（与问题 2 同一口径：Q = h_sea * A_eff * (Tmax - Tsea)）----
    # 材料导热系数影响翅片效率 eta_f；壁厚主要影响承压、成本、寿命与内部空间。
    dT = T_MAX - t_sea
    t_film = (T_MAX + t_sea) / 2.0
    h_sea = h_horizontal_cylinder(a, dT, t_film, sea_props)
    m = math.sqrt(2.0 * h_sea / (k_mat * FIN_D))
    mh = m * FIN_H
    eta_f = 1.0 if mh < 1e-12 else math.tanh(mh) / mh
    a_eff = a_base + total_fins * eta_f * a_fin_one
    q_total = h_sea * a_eff * dT
    h_air = 0.0

    # ---- 散热与装机容量 ----
    v_inner = a_in ** 2 * l_in
    v_server = SERVER_W * SERVER_H * SERVER_L
    n_space = v_inner / v_server
    n_theory = q_total / Q0
    n_final = int(math.floor(min(n_theory, n_space))) if n_theory >= 1.0 else 0

    # ---- 成本：材料费 + 防腐涂层费 ----
    v_wall = a ** 2 * HULL_LENGTH - a_in ** 2 * l_in
    v_fin = total_fins * FIN_H * FIN_D * HULL_LENGTH
    mass_kg = (v_wall + v_fin) * rho_mat
    material_cost = mass_kg * price / 1000.0
    coating_area = a_out_base + total_fins * (2.0 * FIN_H + FIN_D) * HULL_LENGTH
    coating_cost = coating_area * COATING_PRICE
    cost = material_cost + coating_cost

    # ---- 使用寿命：腐蚀裕量 / 腐蚀速率 ----
    corr_allow_mm = (wall - t_req) * 1000.0
    life = min(LIFE_CAP, corr_allow_mm / corr) if corr_allow_mm > 0.0 else 0.0

    # ---- 约束（g >= 0 形式）----
    g1 = wall - t_req                 # 承压
    g2 = life - MIN_LIFE              # 寿命
    g3 = n_theory - 1.0               # 散热至少支持 1 台
    g4 = a_in - 0.05                  # 内部空间
    viol = (max(0.0, -g1) + max(0.0, -g2) + max(0.0, -g3)
            + max(0.0, -g4))
    feasible = viol <= 1e-9

    return {
        "mat_idx": int(mat_idx), "depth": depth, "wall": wall,
        "Q": q_total, "cost": cost, "life": life,
        "N": n_final, "n_theory": n_theory, "n_space": n_space,
        "h_sea": h_sea, "h_air": h_air, "eta_f": eta_f,
        "A_eff": a_eff, "T_sea": t_sea, "p_hydro": p_hydro,
        "t_req": t_req, "corr_allow_mm": corr_allow_mm,
        "mass_kg": mass_kg, "material_cost": material_cost,
        "coating_cost": coating_cost, "feasible": feasible, "viol": viol,
        "g1": g1, "g2": g2, "g3": g3, "g4": g4,
        # f1 按题目口径使用散热能力 Q；N 由 Q 与空间上限导出并单独输出
        "obj": np.array([-q_total, cost, -life]),
    }


def _bad_result() -> dict:
    """几何失效时的兜底结果（超大目标 + 超大违和）。"""
    return {
        "mat_idx": -1, "depth": np.nan, "wall": np.nan,
        "Q": 0.0, "cost": 1e12, "life": 0.0,
        "N": 0, "n_theory": 0.0, "n_space": 0.0,
        "h_sea": 0.0, "h_air": 0.0, "eta_f": 0.0, "A_eff": 0.0,
        "T_sea": np.nan, "p_hydro": np.nan, "t_req": np.nan,
        "corr_allow_mm": -1e12, "mass_kg": 0.0,
        "material_cost": 0.0, "coating_cost": 0.0,
        "feasible": False, "viol": 1e12,
        "g1": -1e12, "g2": -1e12, "g3": -1e12, "g4": -1e12,
        "obj": np.array([1e12, 1e12, 1e12]),
    }


def magnitude_checks(res: dict, label: str) -> None:
    """对关键物理量做数量级校验（国际单位制 + 结果单位）。"""
    checks = [
        ("静水压力 p", res["p_hydro"], "Pa", 5e3, 3e6),
        ("所需壁厚 t_req", res["t_req"], "m", 1e-4, 0.05),
        ("换热系数 h", res["h_sea"], "W/(m^2·K)", 50.0, 5e4),
        ("有效散热面积 A_eff", res["A_eff"], "m^2", 10.0, 500.0),
        ("散热能力 Q", res["Q"], "W", 1e5, 1e8),
        ("理论装机 N_theory", res["n_theory"], "台", 100.0, 1e5),
        ("实际装机 N", res["N"], "台", 0.0, 1e4),
        ("总成本 cost", res["cost"], "元", 1e3, 1e8),
        ("寿命 life", res["life"], "年", 0.0, 60.0),
    ]
    print("数量级校验（%s）" % label)
    all_ok = True
    for name, val, unit, lo, hi in checks:
        ok = lo <= val <= hi
        all_ok = all_ok and ok
        print("  [%s] %-14s = %12.4g %-10s 预期 [%g, %g]"
              % ("OK" if ok else "WARN", name, val, unit, lo, hi))
    print("  结论：%s" % ("全部通过，数量级合理。" if all_ok
                        else "存在越界项，请检查输入参数。"))
    print()


def run_baseline(material_df: pd.DataFrame, temp_pred) -> pd.DataFrame:
    """先跑一个简单基准算例，核对公式、量纲与数量级，再进入 NSGA-II。"""
    print("=" * 76)
    print("第 5 步  基准算例（先跑通，再扩展优化）")
    print("=" * 76)
    mat_idx = 0                     # 6061 铝合金（材料表第 1 行）
    depth = 50.0                    # 海水深度，m
    wall = 0.020                    # 壳体壁厚，m（20 mm 工程基准）
    mat = material_df.iloc[mat_idx]
    sigma_y_pa = float(mat["屈服强度_MPa"]) * 1e6

    print("基准设计：%s，深度 = %.1f m，壁厚 = %.4f m（沿用问题 2 结构）"
          % (mat["材料"], depth, wall))
    print("-" * 76)
    print("分步公式与中间量（全部国际单位制）：")
    p_hydro = RHO_SW * G * depth
    print("  1) 静水压力 p = rho_w*g*d = %.1f*%.2f*%.1f = %.4g Pa = %.3f MPa"
          % (RHO_SW, G, depth, p_hydro, p_hydro / 1e6))
    print("  2) 许用应力 sigma_a = sigma_y / n_s = %.4g / %.1f = %.4g Pa"
          % (sigma_y_pa, SAFETY_FACTOR, sigma_y_pa / SAFETY_FACTOR))
    t_req = p_hydro * ENVELOPE_SIDE / (2.0 * sigma_y_pa / SAFETY_FACTOR)
    print("  3) 所需壁厚 t_req = p*D/(2*sigma_a) = %.4g m = %.3f mm"
          % (t_req, t_req * 1000.0))

    res = evaluate_design(mat_idx, depth, wall, material_df, temp_pred)
    print("  4) 海水温度 T_sea(d=%.0f m) = %.4f ℃（温度-深度回归模型）"
          % (depth, res["T_sea"]))
    print("  5) 换热系数 h = %.2f W/(m^2·K)（Churchill-Chu 自然对流）"
          % res["h_sea"])
    print("  6) 翅片效率 eta_f = %.4f，有效散热面积 A_eff = %.3f m^2"
          % (res["eta_f"], res["A_eff"]))
    print("  7) 总散热量 Q = h*A_eff*(T_max-T_sea) = %.4g W = %.3f MW"
          % (res["Q"], res["Q"] / 1e6))
    print("  8) 理论装机 N_theory = Q/q0 = %.1f 台；空间上限 N_space = %.1f 台；"
          % (res["n_theory"], res["n_space"]))
    print("     实际装机 N = floor(min(N_theory, N_space)) = %d 台" % res["N"])
    print("  9) 材料成本 = %.2f 元，涂层成本 = %.2f 元，总成本 = %.2f 元"
          % (res["material_cost"], res["coating_cost"], res["cost"]))
    print("  10) 寿命 = (wall - t_req)*1000 / 腐蚀速率 = %.2f 年（上限 50 年）"
          % res["life"])
    print("  约束检查（g>=0 为满足）：承压 g1=%.4g，寿命 g2=%.4g，"
          "散热 g3=%.4g，空间 g4=%.4g，可行=%s"
          % (res["g1"], res["g2"], res["g3"], res["g4"], res["feasible"]))
    print("-" * 76)
    magnitude_checks(res, "基准算例")

    baseline_df = pd.DataFrame([{
        "材料": mat["材料"], "depth_m": depth, "wall_m": wall,
        "Q_W": res["Q"], "cost_元": res["cost"], "life_年": res["life"],
        "N_台": res["N"], "n_theory_台": res["n_theory"],
        "n_space_台": res["n_space"], "T_sea_C": res["T_sea"],
        "t_req_m": t_req, "p_hydro_Pa": p_hydro,
        "h_W_m2K": res["h_sea"], "A_eff_m2": res["A_eff"],
        "feasible": res["feasible"],
    }])
    print("基准算例结果表：")
    print(baseline_df.to_string(index=False))
    print()
    return baseline_df


# ==================================================================
# 7. 手写 NSGA-II
# ==================================================================
class Individual:
    __slots__ = ("chrom", "obj", "feasible", "viol", "rank", "crowd", "res")

    def __init__(self, chrom, obj, feasible, viol, res):
        self.chrom = chrom
        self.obj = obj
        self.feasible = feasible
        self.viol = viol
        self.rank = 0
        self.crowd = 0.0
        self.res = res


def _dominates(p: Individual, q: Individual) -> bool:
    """约束支配（最小化目标）。"""
    if p.feasible != q.feasible:
        return p.feasible and not q.feasible
    if not p.feasible:
        return p.viol < q.viol - 1e-12
    return (np.all(p.obj <= q.obj + 1e-12)
            and np.any(p.obj < q.obj - 1e-12))


def fast_non_dominated_sort(pop):
    """NSGA-II 快速非支配排序，返回分层索引列表。"""
    n = len(pop)
    dominated = [set() for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(pop[i], pop[j]):
                dominated[i].add(j)
            elif _dominates(pop[j], pop[i]):
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


def crowding_distance(pop, front_idx):
    """拥挤距离：目标空间用目标值，不可行个体用违和量。"""
    m = len(pop[0].obj)
    front = [pop[i] for i in front_idx]
    for ind in front:
        ind.crowd = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowd = float("inf")
        return

    # 可行个体按目标逐维计算；不可行个体按违和量计算
    feasible_idx = [i for i, ind in enumerate(front) if ind.feasible]
    infeasible_idx = [i for i, ind in enumerate(front) if not ind.feasible]
    for i in feasible_idx:
        front[i].crowd = 0.0
    if feasible_idx:
        for mj in range(m):
            order = sorted(feasible_idx, key=lambda i: front[i].obj[mj])
            front[order[0]].crowd = float("inf")
            front[order[-1]].crowd = float("inf")
            rng = front[order[-1]].obj[mj] - front[order[0]].obj[mj]
            if rng < 1e-12:
                continue
            for i in range(1, len(order) - 1):
                if front[order[i]].crowd != float("inf"):
                    front[order[i]].crowd += (front[order[i + 1]].obj[mj]
                                              - front[order[i - 1]].obj[mj]) / rng
    if infeasible_idx:
        order = sorted(infeasible_idx, key=lambda i: front[i].viol)
        front[order[0]].crowd = float("inf")
        front[order[-1]].crowd = float("inf")
        rng = front[order[-1]].viol - front[order[0]].viol
        if rng > 1e-12:
            for i in range(1, len(order) - 1):
                if front[order[i]].crowd != float("inf"):
                    front[order[i]].crowd += (front[order[i + 1]].viol
                                              - front[order[i - 1]].viol) / rng


def _tournament(pop, rng):
    a, b = rng.integers(0, len(pop), size=2)
    ia, ib = pop[a], pop[b]
    if ia.rank < ib.rank:
        return ia
    if ib.rank < ia.rank:
        return ib
    return ia if ia.crowd > ib.crowd else ib


def _sbx(p1, p2, lo, hi, rng):
    """模拟二进制交叉（实数基因）。"""
    if rng.random() > GA_PC:
        return p1, p2
    u = rng.random()
    beta = (2.0 * u) ** (1.0 / (ETA_C + 1.0)) if u <= 0.5 \
        else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (ETA_C + 1.0))
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return float(np.clip(c1, lo, hi)), float(np.clip(c2, lo, hi))


def _poly_mut(x, lo, hi, rng):
    """多项式变异（实数基因）。"""
    if rng.random() > GA_PM:
        return float(x)
    u = rng.random()
    if u < 0.5:
        delta = (2.0 * u) ** (1.0 / (ETA_M + 1.0)) - 1.0
    else:
        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (ETA_M + 1.0))
    return float(np.clip(x + delta * (hi - lo), lo, hi))


def nsga2(material_df, temp_pred, pop_size=GA_POP, generations=GA_GEN,
          seed=GA_SEED, verbose_every=20) -> tuple[list, list, list]:
    """主优化流程，返回（最终种群, 帕累托结果列表, 收敛历史）。"""
    rng = np.random.default_rng(seed)
    n_mat = len(material_df)
    bounds = [(0.0, float(n_mat - 1)),
              (DEPTH_MIN, DEPTH_MAX),
              (WALL_MIN, WALL_MAX)]

    def evaluate(chrom):
        mi = int(round(chrom[0]))
        mi = int(np.clip(mi, 0, n_mat - 1))
        res = evaluate_design(mi, chrom[1], chrom[2], material_df, temp_pred)
        return Individual(chrom.copy(), res["obj"].copy(),
                          res["feasible"], res["viol"], res)

    # 初始种群（含部分启发式深度/壁厚初值，加速收敛）
    pop = []
    for i in range(pop_size):
        mi = int(rng.integers(0, n_mat))
        if i < pop_size // 5:
            depth = float(np.linspace(DEPTH_MIN + 5, DEPTH_MAX - 5, pop_size // 5)[i])
            wall = float(np.linspace(WALL_MIN + 0.003, WALL_MAX - 0.005, pop_size // 5)[i])
        else:
            depth = rng.uniform(DEPTH_MIN, DEPTH_MAX)
            wall = rng.uniform(WALL_MIN, WALL_MAX)
        pop.append(evaluate([mi, depth, wall]))

    history = []
    for gen in range(1, generations + 1):
        # 非支配排序 + 拥挤距离
        fronts = fast_non_dominated_sort(pop)
        for f_idx in fronts:
            crowding_distance(pop, f_idx)

        # 二元锦标赛选择父代
        parents = [_tournament(pop, rng) for _ in range(pop_size)]

        # 交叉 + 变异生成子代
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[i + 1] if i + 1 < pop_size else parents[0]
            # 材料基因：均匀交叉
            m1 = p1.chrom[0] if rng.random() < 0.5 else p2.chrom[0]
            m2 = p2.chrom[0] if rng.random() < 0.5 else p1.chrom[0]
            # 实数基因：SBX
            d1, d2 = _sbx(p1.chrom[1], p2.chrom[1], DEPTH_MIN, DEPTH_MAX, rng)
            w1, w2 = _sbx(p1.chrom[2], p2.chrom[2], WALL_MIN, WALL_MAX, rng)
            c1 = [m1, _poly_mut(d1, DEPTH_MIN, DEPTH_MAX, rng),
                  _poly_mut(w1, WALL_MIN, WALL_MAX, rng)]
            c2 = [m2, _poly_mut(d2, DEPTH_MIN, DEPTH_MAX, rng),
                  _poly_mut(w2, WALL_MIN, WALL_MAX, rng)]
            # 材料小概率变异
            if rng.random() < GA_PM:
                c1[0] = int(rng.integers(0, n_mat))
            if rng.random() < GA_PM:
                c2[0] = int(rng.integers(0, n_mat))
            offspring.append(evaluate(c1))
            offspring.append(evaluate(c2))
        offspring = offspring[:pop_size]

        # (mu+lambda) 精英保留
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

        # 收敛记录：前沿规模、最好 Q、平均成本、平均拥挤距离
        fea = [ind for ind in pop if ind.feasible]
        if fea:
            best_q = max(ind.res["Q"] for ind in fea)
            mean_cost = float(np.mean([ind.res["cost"] for ind in fea]))
            front_sz = len([ind for ind in pop
                            if ind.feasible and ind.rank == 0])
            crowd = float(np.mean([ind.crowd for ind in fea
                                   if np.isfinite(ind.crowd)]) or 0.0)
        else:
            best_q, mean_cost, front_sz, crowd = 0.0, 0.0, 0, 0.0
        history.append((gen, front_sz, best_q, mean_cost, crowd))
        if gen % verbose_every == 0 or gen == generations:
            print("    第 %3d 代：前沿可行个体 %3d，最好 Q=%10.1f W，"
                  "前沿平均成本=%10.0f 元"
                  % (gen, front_sz, best_q, mean_cost))

    fronts = fast_non_dominated_sort(pop)
    front0 = [pop[i] for i in fronts[0] if pop[i].feasible]
    pareto = [ind.res for ind in front0]
    print("    优化结束：帕累托前沿可行解数量 = %d" % len(pareto))
    print()
    return pop, pareto, history


# ==================================================================
# 8. TOPSIS 决策（含矩阵预处理与方向处理）
# ==================================================================
def preprocess_pareto_matrix(pareto: list) -> pd.DataFrame:
    """帕累托解转 DataFrame，做缺失/重复/异常值（IQR 缩尾）处理。"""
    df = pd.DataFrame([{k: r[k] for k in
                        ("mat_idx", "depth", "wall", "Q", "cost", "life",
                         "N", "n_theory", "n_space", "T_sea", "t_req",
                         "corr_allow_mm", "mass_kg")} for r in pareto])
    df["材料"] = df["mat_idx"].map(
        lambda i: MAT_DF.iloc[int(i)]["材料"] if int(i) < len(MAT_DF) else "?")
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=["Q", "cost", "life"]).reset_index(drop=True)
    print("    去重：%d -> %d 行" % (before, len(df)))

    # 异常值缩尾（只针对三个目标列，防止极端值影响 TOPSIS）
    n_out = 0
    for col in ("Q", "cost", "life"):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        n_out += bad
        df[col] = df[col].clip(lo, hi)
    print("    异常值缩尾（IQR 1.5 倍）共处理 %d 个单元格" % n_out)
    return df


def zscore_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    return z


def minmax_directed(z: pd.DataFrame) -> pd.DataFrame:
    """方向处理：散热 Q、寿命为效益型，成本为成本型（取反）。"""
    out = pd.DataFrame(index=z.index)
    for col in z.columns:
        mn, mx = z[col].min(), z[col].max()
        if mx - mn < 1e-12:
            out[col] = 1.0
            continue
        if col == "cost":
            out[col] = (mx - z[col]) / (mx - mn)     # 成本越小越好
        else:
            out[col] = (z[col] - mn) / (mx - mn)     # 效益型
    return out


def topsis(pareto_df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """TOPSIS 完整流程，返回带评分与排序的表。"""
    cols = ["Q", "cost", "life"]
    print("=" * 76)
    print("第 8 步  TOPSIS 决策（权重：散热 %.2f / 成本 %.2f / 寿命 %.2f）"
          % tuple(weights))
    print("=" * 76)
    print("帕累托矩阵 shape：", pareto_df.shape)
    print("前 8 行（含全部具体数字）：")
    show_cols = ["材料", "depth", "wall", "Q", "cost", "life", "N",
                 "n_theory", "n_space", "T_sea", "t_req"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: "%.6g" % v)
    print(pareto_df[show_cols].head(8).to_string(index=False))

    # 1) z-score 标准化（均值 0，方差 1）
    z = zscore_matrix(pareto_df, cols)
    print("\n标准化矩阵（z-score，shape=%s）前 8 行：" % (z.shape,))
    print(z.head(8).to_string())

    # 2) 极差变换：统一为效益型（成本列已取反）
    mm = minmax_directed(z)
    print("\n方向处理后的正向化矩阵（Q/寿命越大越好，成本已反向）前 8 行：")
    print(mm.head(8).to_string())

    # 3) 向量归一化 + 加权
    norm = mm / np.sqrt((mm ** 2).sum(axis=0))
    w = weights / weights.sum()
    v = norm.mul(w, axis=1)
    print("\n向量归一化 + 加权矩阵（shape=%s）前 8 行：" % (v.shape,))
    print(v.head(8).to_string())

    # 4) 正负理想解与距离
    v_pos = v.max(axis=0)
    v_neg = v.min(axis=0)
    d_pos = np.sqrt(((v - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - v_neg) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg)
    score = pd.DataFrame({
        "D_plus": d_pos, "D_minus": d_neg, "TOPSIS贴近度": closeness,
    })
    score = score.sort_values("TOPSIS贴近度", ascending=False).reset_index()
    print("\n正理想解（各列最大值）：")
    print(v_pos.to_string())
    print("负理想解（各列最小值）：")
    print(v_neg.to_string())

    result = pareto_df.copy()
    result["D_plus"] = d_pos.to_numpy()
    result["D_minus"] = d_neg.to_numpy()
    result["TOPSIS贴近度"] = closeness.to_numpy()
    result = result.sort_values("TOPSIS贴近度", ascending=False).reset_index(drop=True)
    result.insert(0, "排名", np.arange(1, len(result) + 1))
    print("\nTOPSIS 排名前 10（全部具体数字）：")
    print(result.head(10)[["排名", "材料", "depth", "wall", "Q", "cost",
                           "life", "N", "n_theory", "n_space",
                           "T_sea", "t_req", "TOPSIS贴近度"]].to_string(index=False))
    print()
    return result


# ==================================================================
# 9. 灵敏度分析
# ==================================================================
def sensitivity_analysis(best: pd.Series, material_df: pd.DataFrame,
                         temp_pred) -> pd.DataFrame:
    """连续参数 ±10% 灵敏度 + 材料离散灵敏度。"""
    print("=" * 76)
    print("第 9 步  灵敏度分析（基准 = TOPSIS 最优方案）")
    print("=" * 76)
    base = evaluate_design(int(best["mat_idx"]), float(best["depth"]),
                           float(best["wall"]), material_df, temp_pred)
    base_q, base_cost, base_life = base["Q"], base["cost"], base["life"]

    rows = []

    def rel(v, b):
        return (v - b) / b * 100.0 if abs(b) > 1e-12 else np.nan

    # ---- 连续参数灵敏度 ----
    mat = material_df.iloc[int(best["mat_idx"])]

    def wrap(mat_idx, depth, wall, k_scale=1.0, price_scale=1.0,
             corr_scale=1.0, yield_scale=1.0, tsea_scale=1.0):
        # 临时改材料表参数（不修改全局）
        tmp = material_df.copy()
        tmp.loc[mat_idx, "导热系数_W_mK"] *= k_scale
        tmp.loc[mat_idx, "价格_元_吨"] *= price_scale
        tmp.loc[mat_idx, "腐蚀速率_mm_年"] *= corr_scale
        tmp.loc[mat_idx, "屈服强度_MPa"] *= yield_scale
        pred0 = temp_pred
        if abs(tsea_scale - 1.0) > 1e-9:
            pred0 = lambda d: temp_pred(d) * tsea_scale   # noqa: E731
        return evaluate_design(mat_idx, depth, wall, tmp, pred0)

    factors = [
        ("海水深度 +10%", "depth", 1.10),
        ("海水深度 -10%", "depth", 0.90),
        ("壁厚 +10%", "wall", 1.10),
        ("壁厚 -10%", "wall", 0.90),
        ("材料导热系数 +10%", "k", 1.10),
        ("材料导热系数 -10%", "k", 0.90),
        ("材料价格 +10%", "price", 1.10),
        ("材料价格 -10%", "price", 0.90),
        ("腐蚀速率 +10%", "corr", 1.10),
        ("腐蚀速率 -10%", "corr", 0.90),
        ("屈服强度 +10%", "yield", 1.10),
        ("屈服强度 -10%", "yield", 0.90),
        ("海水温度 +10%", "tsea", 1.10),
        ("海水温度 -10%", "tsea", 0.90),
    ]
    for name, kind, scale in factors:
        if kind == "depth":
            d = float(np.clip(best["depth"] * scale, DEPTH_MIN, DEPTH_MAX))
            r = evaluate_design(int(best["mat_idx"]), d, float(best["wall"]),
                                material_df, temp_pred)
        elif kind == "wall":
            w = float(np.clip(best["wall"] * scale, WALL_MIN, WALL_MAX))
            r = evaluate_design(int(best["mat_idx"]), float(best["depth"]), w,
                                material_df, temp_pred)
        else:
            scales = {"k": 1.0, "price": 1.0, "corr": 1.0, "yield": 1.0,
                      "tsea": 1.0}
            scales[kind] = scale
            r = wrap(int(best["mat_idx"]), float(best["depth"]),
                     float(best["wall"]),
                     k_scale=scales["k"], price_scale=scales["price"],
                     corr_scale=scales["corr"], yield_scale=scales["yield"],
                     tsea_scale=scales["tsea"])
        if r["feasible"]:
            rows.append({
                "扰动项": name, "Q变化%": rel(r["Q"], base_q),
                "成本变化%": rel(r["cost"], base_cost),
                "寿命变化%": rel(r["life"], base_life),
            })

    sens = pd.DataFrame(rows)
    print("连续参数 ±10% 灵敏度（相对基准的变化百分比）：")
    pd.set_option("display.float_format", lambda v: "%.3f" % v)
    print(sens.to_string(index=False))

    # ---- 材料离散灵敏度（保持最优 depth/wall）----
    mat_rows = []
    for i in range(len(material_df)):
        r = evaluate_design(i, float(best["depth"]), float(best["wall"]),
                            material_df, temp_pred)
        if r["feasible"]:
            mat_rows.append({
                "材料": material_df.iloc[i]["材料"],
                "Q_W": r["Q"], "成本_元": r["cost"], "寿命_年": r["life"],
                "N_台": r["N"], "Q相对基准%": rel(r["Q"], base_q),
                "成本相对基准%": rel(r["cost"], base_cost),
                "寿命相对基准%": rel(r["life"], base_life),
            })
    mat_sens = pd.DataFrame(mat_rows)
    print("\n材料离散灵敏度（固定 depth=%.2f m、wall=%.4f m）："
          % (best["depth"], best["wall"]))
    pd.set_option("display.float_format", lambda v: "%.6g" % v)
    print(mat_sens.to_string(index=False))
    print()
    return sens, mat_sens


# ==================================================================
# 10. 绘图
# ==================================================================
def plot_temperature_fit(model: dict, out: Path):
    df = model["df"]
    d_grid = np.linspace(df["depth_m"].min(), df["depth_m"].max(), 200)
    x_grid = (d_grid - model["d_mean"]) / model["d_std"]
    t_fit = np.polyval(model["coef"], x_grid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(df["depth_m"], df["temp_C"], "o", ms=4, label="WOA18 陵水实测")
    ax.plot(d_grid, t_fit, "-", lw=2, label="二次多项式拟合")
    ax.set_xlabel("海水深度 (m)")
    ax.set_ylabel("海水温度 (℃)")
    ax.set_title("南海陵水站温度-深度剖面与回归拟合\nR$^2$=%.4f" % model["r2"])
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.stem(df["depth_m"], model["resid"], linefmt="C0-", markerfmt="C0o",
            basefmt="k-")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("海水深度 (m)")
    ax.set_ylabel("残差 (℃)")
    ax.set_title("残差图（最大绝对残差 %.3f ℃）" % np.abs(model["resid"]).max())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "图1_温度剖面拟合与残差.png", dpi=160)
    plt.close(fig)


def plot_pareto(pareto_df: pd.DataFrame, out: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    q = pareto_df["Q"].to_numpy() / 1e6
    c = pareto_df["cost"].to_numpy() / 1e4
    lf = pareto_df["life"].to_numpy()
    fig = plt.figure(figsize=(16, 4.4))

    ax = fig.add_subplot(1, 4, 1, projection="3d")
    sc = ax.scatter(q, c, lf, c=lf, cmap="viridis", s=12)
    ax.set_xlabel("Q (MW)")
    ax.set_ylabel("成本 (万元)")
    ax.set_zlabel("寿命 (年)")
    ax.set_title("帕累托前沿（3D）")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="寿命 (年)")

    ax = fig.add_subplot(1, 4, 2)
    sc = ax.scatter(q, c, c=lf, cmap="viridis", s=18)
    ax.set_xlabel("散热能力 Q (MW)")
    ax.set_ylabel("成本 (万元)")
    ax.set_title("Q vs 成本")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="寿命 (年)")

    ax = fig.add_subplot(1, 4, 3)
    sc = ax.scatter(q, lf, c=c, cmap="plasma", s=18)
    ax.set_xlabel("散热能力 Q (MW)")
    ax.set_ylabel("寿命 (年)")
    ax.set_title("Q vs 寿命")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="成本 (万元)")

    ax = fig.add_subplot(1, 4, 4)
    sc = ax.scatter(c, lf, c=q, cmap="coolwarm", s=18)
    ax.set_xlabel("成本 (万元)")
    ax.set_ylabel("寿命 (年)")
    ax.set_title("成本 vs 寿命")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Q (MW)")
    fig.tight_layout()
    fig.savefig(out / "图2_帕累托前沿.png", dpi=160)
    plt.close(fig)


def plot_convergence(history: list, out: Path):
    hist = np.asarray(history)
    gens = hist[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(gens, hist[:, 1], "-o", ms=2.5)
    axes[0].set_xlabel("代数")
    axes[0].set_ylabel("第 0 层可行解数量")
    axes[0].set_title("NSGA-II 前沿规模收敛")
    axes[0].grid(alpha=0.3)
    axes[1].plot(gens, hist[:, 2] / 1e6, "-o", ms=2.5)
    axes[1].set_xlabel("代数")
    axes[1].set_ylabel("最好 Q (MW)")
    axes[1].set_title("最好散热能力收敛")
    axes[1].grid(alpha=0.3)
    axes[2].plot(gens, hist[:, 4], "-o", ms=2.5)
    axes[2].set_xlabel("代数")
    axes[2].set_ylabel("平均拥挤距离")
    axes[2].set_title("种群多样性收敛")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "图3_NSGA2收敛过程.png", dpi=160)
    plt.close(fig)


def plot_topsis(result: pd.DataFrame, out: Path):
    top = result.head(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].barh(np.arange(len(top))[::-1], top["TOPSIS贴近度"].to_numpy(),
                 color="#4C72B0")
    axes[0].set_yticks(np.arange(len(top))[::-1])
    axes[0].set_yticklabels([f"#{i+1} {m[:8]}" for i, m in
                             enumerate(top["材料"].to_numpy())])
    axes[0].set_xlabel("TOPSIS 贴近度")
    axes[0].set_title("TOPSIS 排序 Top 10")
    axes[0].grid(alpha=0.3)
    axes[1].bar(np.arange(3), [top.iloc[0]["Q"] / 1e6,
                               top.iloc[0]["cost"] / 1e4,
                               top.iloc[0]["life"]], color=["#55A868", "#C44E52",
                                                            "#4C72B0"])
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(["Q (MW)", "成本 (万元)", "寿命 (年)"])
    axes[1].set_title("最优方案三目标取值")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "图4_TOPSIS决策.png", dpi=160)
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame, out: Path):
    if sens.empty:
        return
    df = sens.copy()
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, df["Q变化%"], width=w, label="散热 Q 变化%", color="#4C72B0")
    ax.bar(x, df["成本变化%"], width=w, label="成本变化%", color="#C44E52")
    ax.bar(x + w, df["寿命变化%"], width=w, label="寿命变化%", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(df["扰动项"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("相对变化 (%)")
    ax.set_title("灵敏度分析：参数 ±10% 对三目标的影响")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "图5_灵敏度分析.png", dpi=160)
    plt.close(fig)


# ==================================================================
# 11. 主流程
# ==================================================================
def main():
    font = setup_chinese_font()
    print("使用中文字体：", font)
    print("输出目录：", OUT_DIR)
    print()

    # 第 0.5 步：交付清单基础参数核对（禁止编造）
    verify_delivery_checklist()

    # 第 1-4 步：数据、模型、材料表
    temp_df = load_temperature_profile()
    model = fit_temperature_model(temp_df, degree=2)
    r2_cv = cross_validate_temperature(temp_df, k=5, degree=2)
    temp_sens = sensitivity_temperature_model(temp_df, model)
    temp_pred = make_temp_predictor(model)
    global MAT_DF
    MAT_DF = build_material_table()

    # 温度模型检验图
    plot_temperature_fit(model, OUT_DIR)

    # 第 5 步：基准算例（先跑通，再扩展优化）
    baseline_df = run_baseline(MAT_DF, temp_pred)

    # 第 6 步：NSGA-II
    print("=" * 76)
    print("第 6 步  NSGA-II 多目标优化（种群 %d，代数 %d，目标："
          "max Q / min 成本 / max 寿命）" % (GA_POP, GA_GEN))
    print("=" * 76)
    pop, pareto, history = nsga2(MAT_DF, temp_pred)
    plot_convergence(history, OUT_DIR)

    # 第 7 步：帕累托矩阵预处理
    print("=" * 76)
    print("第 7 步  帕累托前沿结果提取与预处理")
    print("=" * 76)
    pareto_df = preprocess_pareto_matrix(pareto)
    print("预处理后帕累托矩阵 shape：", pareto_df.shape)
    print()

    # 第 8 步：TOPSIS
    topsis_result = topsis(pareto_df, TOPSIS_W)
    plot_pareto(topsis_result, OUT_DIR)
    plot_topsis(topsis_result, OUT_DIR)

    # 第 9 步：优化模型灵敏度（每个模型结束都要做）
    best = topsis_result.iloc[0]
    sens, mat_sens = sensitivity_analysis(best, MAT_DF, temp_pred)
    plot_sensitivity(sens, OUT_DIR)

    # 第 10 步：保存结果表
    pareto_df.to_csv(OUT_DIR / "结果_帕累托前沿.csv",
                     index=False, encoding="utf-8-sig")
    topsis_result.to_csv(OUT_DIR / "结果_TOPSIS排名.csv",
                         index=False, encoding="utf-8-sig")
    baseline_df.to_csv(OUT_DIR / "结果_基准算例.csv",
                       index=False, encoding="utf-8-sig")
    temp_sens.to_csv(OUT_DIR / "结果_温度模型灵敏度.csv",
                     index=False, encoding="utf-8-sig")
    pd.DataFrame(model["resid"], columns=["残差_degC"]).to_csv(
        OUT_DIR / "结果_温度模型残差.csv", index=False, encoding="utf-8-sig")
    cv = pd.DataFrame({"折号": np.arange(1, len(r2_cv) + 1), "R2": r2_cv})
    cv.to_csv(OUT_DIR / "结果_交叉验证.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(OUT_DIR / "结果_连续灵敏度.csv",
                index=False, encoding="utf-8-sig")
    mat_sens.to_csv(OUT_DIR / "结果_材料离散灵敏度.csv",
                    index=False, encoding="utf-8-sig")

    # 最终结果解读
    print("=" * 76)
    print("第 10 步  结果汇总与解读")
    print("=" * 76)
    print("TOPSIS 最优方案（全部具体数字）：")
    cols = ["材料", "depth", "wall", "Q", "cost", "life", "N",
            "n_theory", "n_space", "T_sea", "t_req", "corr_allow_mm",
            "mass_kg", "TOPSIS贴近度"]
    print(topsis_result.iloc[0][cols].to_string())
    print()
    best_res = evaluate_design(int(best["mat_idx"]), float(best["depth"]),
                               float(best["wall"]), MAT_DF, temp_pred)
    magnitude_checks(best_res, "TOPSIS 最优方案")
    print("模型检验汇总：")
    print("  温度-深度回归 R^2 = %.6f，调整 R^2 = %.6f"
          % (model["r2"], model["r2_adj"]))
    print("  5 折交叉验证平均 R^2 = %.4f ± %.4f"
          % (r2_cv.mean(), r2_cv.std()))
    print("  残差最大绝对值 = %.4f ℃，DW = %.3f"
          % (np.abs(model["resid"]).max(), model["dw"]))
    print("  温度模型灵敏度：T(50 m) 最大相对漂移 = %.2f%%"
          % np.abs(temp_sens["T50m变化_pct"]).max())
    print("  NSGA-II 帕累托前沿可行解 = %d 个" % len(pareto_df))
    print()
    print("结果文件已保存到：", OUT_DIR)


if __name__ == "__main__":
    main()
