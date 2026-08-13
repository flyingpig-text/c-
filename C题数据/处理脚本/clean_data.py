"""C题数据清洗：把原始数据整理成可直接建模的干净数据，并记录每一步操作。

输入：
    C题数据/海洋环境数据/*.csv
    C题数据/材料数据/EngineeringToolbox_ThermalConductivity_Metals.html
输出：
    C题数据/清洗后数据/*_clean.csv
    C题数据/数据清洗日志.json
    C题数据/数据清洗报告.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber


BASE = Path(r"D:\46884\Documents\数学建模\C题数据")
RAW_OCEAN = BASE / "海洋环境数据"
RAW_MATERIAL = BASE / "材料数据"
OUT = BASE / "清洗后数据"
LOG_FILE = BASE / "数据清洗日志.json"
REPORT_FILE = BASE / "数据清洗报告.md"

STEPS: list[dict] = []


def log(
    dataset: str,
    step: str,
    action: str,
    before: int,
    after: int,
    reason: str,
    detail: str = "",
) -> None:
    """记录一次清洗操作，用于生成清洗日志与报告。"""
    STEPS.append(
        {
            "数据集": dataset,
            "步骤": step,
            "操作": action,
            "清洗前行数": before,
            "清洗后行数": after,
            "原因": reason,
            "说明": detail,
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def save_csv(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def clean_ersst() -> None:
    """ERSST v5 月均海温：校验连续性、重复、范围后原样保留。"""
    path = RAW_OCEAN / "ERSST_v5_2020-2021_南海站点SST.csv"
    df = load_csv(path)
    n0 = len(df)
    log("ERSST海温", "读取", "读取原始 CSV", n0, n0, "", f"列：{list(df.columns)}")

    miss = int(df.isna().sum().sum())
    log(
        "ERSST海温",
        "缺失值检查",
        "统计缺失值",
        n0,
        n0,
        "月度海温应完整覆盖观测期，缺失会破坏时间序列",
        f"缺失值总数={miss}",
    )

    dup = int(df.duplicated().sum())
    if dup:
        df = df.drop_duplicates().reset_index(drop=True)
    log(
        "ERSST海温",
        "重复检查",
        "删除完全重复行" if dup else "检查重复记录",
        n0,
        len(df),
        "重复记录会导致同一月份被重复计数",
        f"删除重复行数={dup}",
    )

    months = df["month"].astype(str).tolist()
    expected = [str(p) for p in pd.period_range("2020-01", "2021-12", freq="M")]
    missing_months = sorted(set(expected) - set(months))
    log(
        "ERSST海温",
        "连续性检查",
        "校验月份连续并排序",
        len(df),
        len(df),
        "建模需要连续月度时间轴，避免静默缺口",
        f"缺失月份={missing_months if missing_months else '无'}",
    )
    df = df.sort_values("month").reset_index(drop=True)

    sst_cols = [c for c in df.columns if c != "month"]
    lo, hi = float(df[sst_cols].min().min()), float(df[sst_cols].max().max())
    ok = 10 <= lo and hi <= 35
    log(
        "ERSST海温",
        "范围检查",
        "海温物理范围检查",
        len(df),
        len(df),
        "南海表层海温合理范围约为10–35°C，超出说明数据异常",
        f"min={lo:.2f}°C, max={hi:.2f}°C, 通过={ok}",
    )

    save_csv(df, "ERSST_v5_2020-2021_南海站点SST_clean.csv")


def clean_woa_monthly() -> None:
    """WOA18 月均表层温度：校验 1–12 月完整并排序。"""
    path = RAW_OCEAN / "WOA18_1981-2010_月均表层温度_南海站点.csv"
    df = load_csv(path)
    n0 = len(df)
    log("WOA18月均表层温度", "读取", "读取原始 CSV", n0, n0, "", f"列：{list(df.columns)}")

    miss = int(df.isna().sum().sum())
    log("WOA18月均表层温度", "缺失值检查", "统计缺失值", n0, n0, "气候态月均值应完整", f"缺失值总数={miss}")

    dup = int(df.duplicated().sum())
    if dup:
        df = df.drop_duplicates().reset_index(drop=True)
    log(
        "WOA18月均表层温度",
        "重复检查",
        "删除完全重复行" if dup else "检查重复记录",
        n0,
        len(df),
        "同一月份出现两次会导致月均序列失真",
        f"删除重复行数={dup}",
    )

    months = sorted(df["月"].astype(int).tolist())
    missing_months = sorted(set(range(1, 13)) - set(months))
    log(
        "WOA18月均表层温度",
        "完整性检查",
        "校验 1–12 月齐全并按月份排序",
        len(df),
        len(df),
        "季节变化建模需要完整月循环",
        f"缺失月份={missing_months if missing_months else '无'}",
    )
    df = df.sort_values("月").reset_index(drop=True)

    temp_cols = [c for c in df.columns if c != "月"]
    lo, hi = float(df[temp_cols].min().min()), float(df[temp_cols].max().max())
    ok = 10 <= lo and hi <= 35
    log(
        "WOA18月均表层温度",
        "范围检查",
        "温度物理范围检查",
        len(df),
        len(df),
        "南海表层海温合理范围约为10–35°C",
        f"min={lo:.2f}°C, max={hi:.2f}°C, 通过={ok}",
    )

    save_csv(df, "WOA18_1981-2010_月均表层温度_南海站点_clean.csv")


def clean_woa_profile() -> None:
    """WOA18 温度剖面：删除海底以下无数据行，保留完整有效剖面。"""
    path = RAW_OCEAN / "WOA18_南海温度剖面.csv"
    df = load_csv(path)
    n0 = len(df)
    log("WOA18温度剖面", "读取", "读取原始 CSV", n0, n0, "", f"列：{list(df.columns)}")

    miss = int(df["温度_degC"].isna().sum())
    max_valid = (
        df.loc[df["温度_degC"].notna(), ["站点", "深度_m"]]
        .groupby("站点")["深度_m"]
        .max()
        .to_dict()
    )
    reason = (
        "缺失值全部位于站点最大有效深度以下（珠海>50m、陵水>100m），"
        "即海底/陆架掩膜之外，并非观测缺测，插值无依据，直接删除"
    )
    df2 = df.dropna(subset=["温度_degC"]).reset_index(drop=True)
    log(
        "WOA18温度剖面",
        "缺失值处理",
        "删除海底以下无数据行",
        n0,
        len(df2),
        reason,
        f"珠海最大有效深度={max_valid['珠海']}m，陵水最大有效深度={max_valid['陵水']}m；"
        f"删除行数={n0 - len(df2)}",
    )

    dup = int(df2.duplicated(subset=["站点", "深度_m"]).sum())
    log(
        "WOA18温度剖面",
        "重复检查",
        "检查站点+深度是否唯一",
        len(df2),
        len(df2),
        "同一站点同一深度只能有一条温度记录",
        f"重复行数={dup}",
    )

    depths = sorted(df["深度_m"].unique())
    lo, hi = float(df2["温度_degC"].min()), float(df2["温度_degC"].max())
    ok = -2 <= lo and hi <= 35
    log(
        "WOA18温度剖面",
        "范围检查",
        "温度物理范围检查",
        len(df2),
        len(df2),
        "海水温度合理范围约为-2–35°C",
        f"深度网格 0–{max(depths):.0f}m 共 {len(depths)} 层；min={lo:.2f}°C, max={hi:.2f}°C, 通过={ok}",
    )

    save_csv(df2, "WOA18_南海温度剖面_clean.csv")


def clean_tide() -> None:
    """HKO 逐时潮高：修正 24 时编码、校验逐时连续，保留全年完整序列。"""
    path = RAW_OCEAN / "潮汐" / "HKO_ChekLapKokE_2026_hourly_tide.csv"
    df = load_csv(path)
    n0 = len(df)
    log("HKO潮汐", "读取", "读取原始 CSV", n0, n0, "", f"列：{list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    log(
        "HKO潮汐",
        "时间列修正",
        "将 hour=24 转换为次日 00:00 并生成 datetime 列",
        n0,
        n0,
        "香港天文台逐时表小时编码为01–24，24:00 即当日结束、次日零点，直接按 24:00 处理会破坏时间轴",
        f"时间范围：{df['datetime'].min()} 至 {df['datetime'].max()}",
    )

    df = df.sort_values("datetime").reset_index(drop=True)
    dup = int(df["datetime"].duplicated().sum())
    gaps = df["datetime"].diff().dropna()
    bad = int((gaps != pd.Timedelta(hours=1)).sum())
    log(
        "HKO潮汐",
        "时间连续性检查",
        "校验逐时序列无重复、无缺口",
        len(df),
        len(df),
        "潮汐时间序列必须逐时连续，缺口会导致频谱分析/季节分析失真",
        f"重复时间点={dup}，非 1 小时间隔数={bad}",
    )

    lo, hi = float(df["tide_height_m"].min()), float(df["tide_height_m"].max())
    ok = -1 <= lo and hi <= 5
    log(
        "HKO潮汐",
        "范围检查",
        "潮高物理范围检查",
        len(df),
        len(df),
        "赤鱲角天文潮预报合理范围约为-1–4m（潮汐基准面以上）",
        f"min={lo:.2f}m, max={hi:.2f}m, 通过={ok}",
    )

    df = df[["datetime", "tide_height_m"]].copy()
    df.insert(0, "站点", "赤鱲角东(ChekLapKokE)_2026天文潮预报")
    save_csv(df, "HKO_ChekLapKokE_2026_hourly_tide_clean.csv")


def clean_metals_html() -> None:
    """Engineering Toolbox 金属导热系数表：HTML 转 CSV，处理承前材料名。"""
    path = RAW_MATERIAL / "EngineeringToolbox_ThermalConductivity_Metals.html"
    df = pd.read_html(path)[0]
    n0 = len(df)
    df.columns = ["material", "temperature_C", "thermal_conductivity_W_per_mK"]

    mat = df["material"].replace('"', np.nan).replace("", np.nan)
    raw_missing = mat.isna()
    df["material"] = mat.ffill().astype("string").str.strip()

    def parse_number_or_range(value: object) -> tuple[float, float, float] | None:
        """把 '237'、'0 - 25'、'61 – 90' 解析成 (min, max, 中点)。"""
        if pd.isna(value) or not str(value).strip():
            return None
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
        if not nums:
            return None
        lo, hi = min(nums), max(nums)
        return lo, hi, (lo + hi) / 2.0

    temp_parsed = df["temperature_C"].apply(parse_number_or_range)
    k_parsed = df["thermal_conductivity_W_per_mK"].apply(parse_number_or_range)

    df["temperature_min_C"] = temp_parsed.apply(lambda v: v[0] if v else np.nan)
    df["temperature_max_C"] = temp_parsed.apply(lambda v: v[1] if v else np.nan)
    df["temperature_C"] = temp_parsed.apply(lambda v: v[2] if v else np.nan)
    df["thermal_conductivity_min_W_per_mK"] = k_parsed.apply(lambda v: v[0] if v else np.nan)
    df["thermal_conductivity_max_W_per_mK"] = k_parsed.apply(lambda v: v[1] if v else np.nan)
    df["thermal_conductivity_W_per_mK"] = k_parsed.apply(lambda v: v[2] if v else np.nan)

    def quality_note(row: pd.Series) -> str:
        notes = []
        if (
            not pd.isna(row["temperature_min_C"])
            and row["temperature_min_C"] != row["temperature_max_C"]
        ):
            notes.append("温度取范围中点")
        if (
            not pd.isna(row["thermal_conductivity_min_W_per_mK"])
            and row["thermal_conductivity_min_W_per_mK"]
            != row["thermal_conductivity_max_W_per_mK"]
        ):
            notes.append("导热系数取范围中点")
        if pd.isna(row["temperature_C"]):
            notes.append("原表温度缺失，未填补")
        return "; ".join(notes) if notes else "完整点值"

    df["数据质量"] = df.apply(quality_note, axis=1)

    invalid = int((raw_missing & df["temperature_C"].isna() & df["thermal_conductivity_W_per_mK"].isna()).sum())
    df = df[
        ~(raw_missing & df["temperature_C"].isna() & df["thermal_conductivity_W_per_mK"].isna())
    ].reset_index(drop=True)
    log(
        "金属导热系数",
        "HTML转CSV",
        "解析网页表格：前向填充材料名、范围取中点、删除空分隔行",
        n0,
        len(df),
        '网页中用引号""表示与上一行相同的材料名，已前向填充；"0 - 25"等温度/导热系数范围保留为 min/max 并取中点；空分隔行无任何数据，删除',
        f"删除空分隔行数={invalid}；温度取中点的行数="
        f"{int(((df['temperature_min_C'] != df['temperature_max_C']) & df['temperature_min_C'].notna()).sum())}；"
        f"导热系数取中点的行数="
        f"{int(((df['thermal_conductivity_min_W_per_mK'] != df['thermal_conductivity_max_W_per_mK']) & df['thermal_conductivity_min_W_per_mK'].notna()).sum())}",
    )

    dup = int(df.duplicated(subset=["material", "temperature_C"]).sum())
    log(
        "金属导热系数",
        "重复检查",
        "检查材料+温度组合是否唯一",
        len(df),
        len(df),
        "同一材料同一温度只允许一条导热系数",
        f"重复行数={dup}",
    )

    lo_t, hi_t = float(df["temperature_C"].min()), float(df["temperature_C"].max())
    lo_k, hi_k = float(df["thermal_conductivity_W_per_mK"].min()), float(
        df["thermal_conductivity_W_per_mK"].max()
    )
    log(
        "金属导热系数",
        "范围检查",
        "温度与导热系数范围检查",
        len(df),
        len(df),
        "金属导热系数应为正数且数量级合理（约0.5–500 W/(m·K)）",
        f"温度 {lo_t:.0f}–{hi_t:.0f}°C，导热系数 {lo_k:.3f}–{hi_k:.3f} W/(m·K)；温度缺失未填补行数={int(df['temperature_C'].isna().sum())}",
    )

    df = df[
        [
            "material",
            "temperature_C",
            "temperature_min_C",
            "temperature_max_C",
            "thermal_conductivity_W_per_mK",
            "thermal_conductivity_min_W_per_mK",
            "thermal_conductivity_max_W_per_mK",
            "数据质量",
        ]
    ]
    save_csv(df, "金属导热系数_EngineeringToolbox_clean.csv")


MATERIAL_COLS = [
    "类别",
    "材料",
    "成分/描述",
    "密度",
    "密度单位",
    "弹性模量_psi",
    "屈服强度_ksi",
    "抗拉强度_ksi",
    "压缩强度_ksi",
    "抗弯强度_ksi",
    "弯曲模量_ksi",
    "吸水率",
    "含水率_pct",
    "撕裂强度",
    "耐磨性",
    "泊松比",
    "海水电位_V",
    "腐蚀类型",
    "海洋环境行为",
    "孔隙率_vol%",
    "折射率",
    "用途",
    "特别说明",
    "来源页码",
]


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _parse_eng_numbers(text: str) -> list[float]:
    """解析 '10 x 106'、'3.0 x 105 - 4.1 x 105'、'0.2 - 0.45' 中的数值。"""
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[x×]\s*10\s*(\d+)", r"\1e\2", text)
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text)]


def _material_category(section: str, material: str) -> str:
    """按附件章节归类；章节识别失败时用材料名兜底。"""
    en2zh = {
        "aluminum alloys": "铝及铝合金",
        "copper and copper alloys": "铜及铜合金",
        "nickel alloys": "镍及镍合金",
        "iron and steels": "铁和钢",
        "titanium alloys": "钛及钛合金",
        "titanium and titanium alloys": "钛及钛合金",
        "stainless steels": "不锈钢",
        "other metals": "其他金属",
        "polymers, rubbers, and elastomers": "聚合物/橡胶/弹性体",
        "concrete and glass": "混凝土和玻璃",
        "wood": "木材",
        "fiber reinforced plastics (frp)": "纤维增强塑料(FRP)",
    }
    key = section.strip().lower()
    if key in en2zh:
        return en2zh[key]
    m = material.lower()
    if "titanium" in m:
        return "钛及钛合金"
    if any(k in m for k in ("aluminum bronze", "brass", "bronze", "copper", "copper-nickel")):
        return "铜及铜合金"
    if "aluminum" in m:
        return "铝及铝合金"
    if any(k in m for k in ("monel", "inconel", "incoloy")):
        return "镍及镍合金"
    if "stainless" in m:
        return "不锈钢"
    if any(k in m for k in ("steel", "iron")):
        return "铁和钢"
    if any(k in m for k in ("magnesium", "zinc", "lead", "gold", "platinum", "silver")):
        return "其他金属"
    if any(k in m for k in ("concrete", "glass")):
        return "混凝土和玻璃"
    if any(k in m for k in ("hardwood", "teak", "mahogany", "softwood", "cedar", "cypress", "pine", "spruce", "oak", "maple")):
        return "木材"
    if any(k in m for k in ("epoxy", "kevlar", "carbon fiber", "glass reinforced", "frp")):
        return "纤维增强塑料(FRP)"
    return "未分类"


def _parse_material_block(block: list, page_no: int, category: str) -> dict | None:
    rows = [r for r in block if r and any(c not in (None, "") for c in r)]
    if len(rows) < 4:
        return None

    rec = {col: "" for col in MATERIAL_COLS}
    rec["类别"] = category
    rec["来源页码"] = str(page_no)

    h0, v0 = rows[0], rows[1]
    rec["材料"] = _clean_cell(v0[0])
    rec["成分/描述"] = _clean_cell(v0[1]) if len(v0) > 1 else ""
    rec["密度"] = _clean_cell(v0[2]) if len(v0) > 2 else ""
    h2 = _clean_cell(h0[2]) if len(h0) > 2 else ""
    if "lb/in3" in h2:
        rec["密度单位"] = "lb/in3"
    elif "lb/ft3" in h2:
        rec["密度单位"] = "lb/ft3"
    elif "specific gravity" in h2.lower():
        rec["密度单位"] = "SG(无量纲)"

    def fill_from_header(header_row: list, value_row: list) -> None:
        for j, hcell in enumerate(header_row):
            if hcell is None:
                continue
            h = _clean_cell(hcell).lower()
            val = _clean_cell(value_row[j]) if j < len(value_row) else ""
            if any(k in h for k in ("elastic modulus", "tensile modulus", "modulus in bending")):
                rec["弹性模量_psi"] = val
            elif "yield" in h:
                rec["屈服强度_ksi"] = val
            elif "tensile strength" in h:
                rec["抗拉强度_ksi"] = val
            elif "compressive" in h:
                rec["压缩强度_ksi"] = val
            elif "flexural" in h:
                rec["抗弯强度_ksi"] = val
            elif "modulus of rupture" in h:
                rec["弯曲模量_ksi"] = val
            elif "water absorption" in h:
                rec["吸水率"] = val
            elif "moisture content" in h:
                rec["含水率_pct"] = val
            elif "tear resistance" in h:
                rec["撕裂强度"] = val
            elif "abrasion resistance" in h:
                rec["耐磨性"] = val
            elif "poisson" in h:
                rec["泊松比"] = val
            elif "behavior" in h:
                rec["海洋环境行为"] = val
            elif "potential" in h:
                rec["海水电位_V"] = val
            elif "corrosion" in h:
                rec["腐蚀类型"] = val
            elif "uses" in h:
                rec["用途"] = val
            elif "porosity" in h:
                rec["孔隙率_vol%"] = val
            elif "marine attack" in h:
                rec["海洋环境行为"] = val
            elif "refractive" in h:
                rec["折射率"] = val

    fill_from_header(rows[2], rows[3])
    if len(rows) >= 6:
        fill_from_header(rows[4], rows[5])
    if len(rows) == 7:
        notes = " ".join(_clean_cell(c) for c in rows[6] if c)
        if "Special Notes:" in notes:
            notes = notes.replace("Special Notes:", "").strip()
        rec["特别说明"] = notes
    return rec


def clean_materials_pdf() -> None:
    """把官方附件《海洋材料性能》PDF 结构化成宽表 CSV。"""
    pdf_path = BASE.parent / "题目" / "C题附件.pdf"
    section_pattern = re.compile(
        r"^(Aluminum Alloys|Copper and Copper alloys|Nickel Alloys|Iron and Steels|"
        r"Titanium and Titanium Alloys|Titanium Alloys|Stainless Steels|Other Metals|Polymers, Rubbers, and Elastomers|"
        r"Concrete and Glass|Wood|Fiber Reinforced Plastics \(FRP\))$",
        re.IGNORECASE,
    )

    records: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        current_section = ""
        for page_no, page in enumerate(pdf.pages, 1):
            words = page.extract_words()
            lines: list[tuple[float, str, bool]] = []
            for top in sorted({round(w["top"], 1) for w in words}):
                line_words = sorted(
                    [w for w in words if round(w["top"], 1) == top], key=lambda w: w["x0"]
                )
                line = " ".join(w["text"] for w in line_words).strip()
                lines.append((top, line, bool(section_pattern.match(line))))

            events: list[tuple[float, int, object]] = [
                (top, 0, text) for top, text, is_section in lines if is_section
            ]
            events += [
                (table.bbox[1], 1, table)
                for table in page.find_tables()
                if table.extract() and table.bbox[1] >= 0
            ]
            events.sort(key=lambda e: (e[0], e[1]))

            for _, kind, payload in events:
                if kind == 0:
                    current_section = str(payload)
                    continue
                table = payload
                block = table.extract()
                rec = _parse_material_block(
                    block,
                    page_no,
                    _material_category(current_section, ""),
                )
                if rec:
                    records.append(rec)

    n0 = len(records)
    # 材料名兜底分类（章节标题识别失败时）
    for rec in records:
        if rec["类别"] == "未分类":
            rec["类别"] = _material_category("", rec["材料"])

    log(
        "海洋材料性能",
        "PDF结构化",
        "解析官方附件并识别章节",
        n0,
        len(records),
        "附件为半结构化表格，逐块按字段头映射，未删除任何材料",
        f"识别材料块数={n0}，来源页码1–19",
    )

    dup = int(pd.Series([r["材料"] for r in records]).duplicated().sum())
    log(
        "海洋材料性能",
        "重复检查",
        "检查材料名唯一性",
        len(records),
        len(records),
        "同一材料不应出现两次",
        f"重复材料名={dup}",
    )

    # 对关键力学与物性字段补充 min/max 数值列
    numeric_fields = [
        ("密度", "密度_min", "密度_max"),
        ("弹性模量_psi", "弹性模量_min_psi", "弹性模量_max_psi"),
        ("屈服强度_ksi", "屈服强度_min_ksi", "屈服强度_max_ksi"),
        ("抗拉强度_ksi", "抗拉强度_min_ksi", "抗拉强度_max_ksi"),
        ("海水电位_V", "电位_min_V", "电位_max_V"),
    ]
    for src, lo_col, hi_col in numeric_fields:
        for rec in records:
            nums = _parse_eng_numbers(rec[src])
            rec[lo_col] = min(nums) if nums else ""
            rec[hi_col] = max(nums) if nums else ""

    columns = MATERIAL_COLS[:]
    for _, lo_col, hi_col in numeric_fields:
        columns.append(lo_col)
        columns.append(hi_col)
    df = pd.DataFrame(records)[columns]
    save_csv(df, "海洋材料性能_C题附件_clean.csv")

    log(
        "海洋材料性能",
        "数值解析",
        "为密度/弹性模量/强度/电位提取 min/max",
        len(df),
        len(df),
        "原文多为范围（如 10 x 106 – 14 x 106 psi），保留原文同时补充数值区间便于建模",
        f"共 {len(df)} 行，{len(numeric_fields)} 组字段补充 min/max 列",
    )


def write_log_and_report() -> None:
    """写出清洗日志（JSON）与供论文使用的清洗报告（Markdown）。"""
    LOG_FILE.write_text(
        json.dumps(
            {"生成时间": datetime.now().isoformat(timespec="seconds"), "清洗步骤": STEPS},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    datasets = [s["数据集"] for s in STEPS if s["数据集"]]
    ordered = []
    for name in datasets:
        if name not in ordered:
            ordered.append(name)

    lines = [
        "# C题数据清洗报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 清洗总览",
        "",
        "| 数据集 | 原始行数 | 清洗后行数 | 删除行数 | 填补缺失数 | 说明 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    out_names = {
        "ERSST海温": "ERSST_v5_2020-2021_南海站点SST_clean.csv",
        "WOA18月均表层温度": "WOA18_1981-2010_月均表层温度_南海站点_clean.csv",
        "WOA18温度剖面": "WOA18_南海温度剖面_clean.csv",
        "HKO潮汐": "HKO_ChekLapKokE_2026_hourly_tide_clean.csv",
        "金属导热系数": "金属导热系数_EngineeringToolbox_clean.csv",
        "海洋材料性能": "海洋材料性能_C题附件_clean.csv",
    }
    for name in ordered:
        steps = [s for s in STEPS if s["数据集"] == name]
        before = steps[0]["清洗前行数"]
        after = steps[-1]["清洗后行数"]
        deleted = sum(max(0, s["清洗前行数"] - s["清洗后行数"]) for s in steps)
        filled = 0
        lines.append(
            f"| {out_names.get(name, name)} | {before} | {after} | {deleted} | {filled} | 见下方分步记录 |"
        )
    lines += [
        "",
        "## 分步清洗记录",
        "",
    ]
    for s in STEPS:
        lines.append(
            f"- **{s['数据集']} · {s['步骤']}**：{s['操作']}（{s['清洗前行数']} → {s['清洗后行数']} 行）。"
            f"原因：{s['原因']}。{s['说明']}"
        )
    lines += [
        "",
        "## 数据字典",
        "",
        "| 文件 | 字段 | 含义与单位 |",
        "| --- | --- | --- |",
        "| ERSST 海温 | month | 月份，YYYY-MM |",
        "| ERSST 海温 | 珠海_SST_degC / 陵水_SST_degC / 南海区域平均_SST_degC | 月均海表温度，°C（ERSST v5，2°网格最近点/区域平均） |",
        "| WOA18 月均表层温度 | 月 | 1–12 月 |",
        "| WOA18 月均表层温度 | 珠海_表层温度_degC / 陵水_表层温度_degC | 月均表层温度，°C（1981–2010 气候态） |",
        "| WOA18 温度剖面 | 站点 / 深度_m / 温度_degC | 站点、深度、温度，°C |",
        "| HKO 潮汐 | 站点 / datetime / tide_height_m | 站点、逐时时间、潮高，m（香港天文台预报，2026） |",
        "| 金属导热系数 | material / temperature_C / temperature_min_C / temperature_max_C / thermal_conductivity_W_per_mK / thermal_conductivity_min_W_per_mK / thermal_conductivity_max_W_per_mK / 数据质量 | 材料、温度(°C)、导热系数(W/(m·K))；范围值同时保留 min/max，数据质量列标注处理方式 |",
        "| 海洋材料性能 | 类别 / 材料 / 成分描述 / 密度及单位 / 弹性模量 / 屈服强度 / 抗拉强度 / 压缩强度 / 抗弯强度 / 弯曲模量 / 吸水率 / 含水率 / 撕裂强度 / 耐磨性 / 泊松比 / 海水电位 / 腐蚀类型 / 海洋环境行为 / 孔隙率 / 折射率 / 用途 / 特别说明 / 来源页码 / 各关键数值 min-max | 官方附件材料性能表；强度为 ksi，弹性模量为 psi，密度单位见“密度单位”列；非金属材料不适用字段留空 |",
        "",
        "## 使用注意事项",
        "",
        "- ERSST 网格为 2°，站点值为最近网格点，适用于季节与年际趋势，不宜当作站点实测。",
        "- WOA18 月均值为 1981–2010 气候态；温度剖面仅保留站点有效水深处数据，珠海最大有效深度 50 m、陵水 100 m。",
        "- HKO 潮汐为天文潮预报，不含风暴潮与余水位；小时编码 01–24 已统一转为标准 datetime，24:00 表示次日 00:00。",
        "- 金属导热系数表为 Engineering Toolbox 参考数据，材料名中引号承前行已前向填充。",
        "- 金属导热系数表有 16 行温度为范围、1 行导热系数为范围，均保留 min/max 并取中点；Palladium 行原表未给出温度，未做填补，仅保留导热系数并标注。",
        "- 海洋材料性能表来自题目官方附件（C题附件.pdf），PDF 中为范围或条件性数值（如 H34、Heat Treated），原文与数值区间均保留；聚合物、木材、FRP 等类别的力学字段口径不同，不适用字段留空。",
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    clean_ersst()
    clean_woa_monthly()
    clean_woa_profile()
    clean_tide()
    clean_metals_html()
    clean_materials_pdf()
    write_log_and_report()
    print(f"清洗完成：{len(STEPS)} 条操作已记录")
    print(f"输出目录：{OUT}")
    print(f"清洗日志：{LOG_FILE}")
    print(f"清洗报告：{REPORT_FILE}")


if __name__ == "__main__":
    main()
