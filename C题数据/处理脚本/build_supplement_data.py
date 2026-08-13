# -*- coding: utf-8 -*-
"""把补充下载的原始数据整理为清洗后 CSV（价格/腐蚀速率/安全系数）。

数据源：
    1) 价格：上海金属网公开行情接口 https://api.shmet.com/api
    2) 腐蚀速率：国家材料环境腐蚀平台 http://data.ecorr.org.cn
    3) 安全系数：GB/T 150.1-2024《压力容器》PDF
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\46884\Documents\数学建模\C题数据\补充材料数据")


def build_price_csv() -> None:
    """把 shmet 期货行情 JSON 整理为 CSV。"""
    raw = ROOT / "价格" / "上海金属网_期货行情_raw.json"
    with raw.open(encoding="utf-8") as f:
        data = json.load(f)["data"]
    rows = [{
        "合约代码": d["contract"], "合约名称": d["name"],
        "最新价_元_吨": d["last"], "涨跌_元_吨": d["updown"],
        "涨跌幅": d["percent"], "成交量": d["volume"],
        "持仓量": d["interest"], "持仓变化": d["chgInterest"],
    } for d in data]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "价格" / "上海金属网_期货行情_clean.csv",
              index=False, encoding="utf-8-sig")
    print("价格 CSV：", ROOT / "价格" / "上海金属网_期货行情_clean.csv")
    print(df.to_string(index=False))


def _extract_fields(html: Path) -> dict:
    text = html.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(
        r"<div align='left'>(.*?)</div></td>\s*"
        r"<td><div class=\"col-md-4\">\s*(.*?)\s*</div></td>",
        re.S)
    out = {}
    for m in pat.finditer(text):
        key = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        val = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if key:
            out[key] = val
    return out


def build_corrosion_csv() -> None:
    """把国家材料环境腐蚀平台明细页解析为 CSV。"""
    files = sorted((ROOT / "腐蚀速率").glob("*.html"))
    rows = []
    for f in files:
        fields = _extract_fields(f)
        rate = fields.get("腐蚀速率(mm/a)", "")
        rows.append({
            "数据表": fields.get("所属数据表", ""),
            "材料名称": fields.get("材料名称", ""),
            "材料牌号": fields.get("材料牌号", ""),
            "试验周期_月": fields.get("试验周期(月)", ""),
            "试验地点": fields.get("查询地点", fields.get("试验地点", "")),
            "试验区域": fields.get("试验区域", ""),
            "腐蚀介质": fields.get("腐蚀介质", ""),
            "腐蚀类型": fields.get("腐蚀类型", ""),
            "腐蚀速率_mm_a": rate,
            "数据来源": fields.get("数据来源", ""),
            "原始文件": f.name,
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "腐蚀速率" / "国家材料环境腐蚀平台_腐蚀速率_clean.csv",
              index=False, encoding="utf-8-sig")
    print("腐蚀速率 CSV：",
          ROOT / "腐蚀速率" / "国家材料环境腐蚀平台_腐蚀速率_clean.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    build_price_csv()
    build_corrosion_csv()
