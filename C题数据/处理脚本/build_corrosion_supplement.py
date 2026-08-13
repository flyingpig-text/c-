# -*- coding: utf-8 -*-
"""下载国家材料环境腐蚀平台 304/316L/1Cr18Ni9Ti 海水腐蚀明细并重建腐蚀速率 CSV。

数据表：不锈钢海水环境腐蚀基础数据
来源：http://data.ecorr.org.cn/edata/01/0101/010103/01010305/
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(r"D:\46884\Documents\数学建模\C题数据\补充材料数据\腐蚀速率")
BASE_URL = "http://data.ecorr.org.cn/edata/01/0101/010103/01010305/2016-06-03/"
IDS = {
    "304": [64, 67, 70, 73],
    "316L": [75, 78, 81, 84],
    "1Cr18Ni9Ti": [29, 30, 31, 32],
}


def _extract_fields(html_text: str) -> dict:
    pat = re.compile(
        r"<div align='left'>(.*?)</div></td>\s*"
        r"<td><div class=\"col-md-4\">\s*(.*?)\s*</div></td>",
        re.S,
    )
    out = {}
    for m in pat.finditer(html_text):
        key = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        val = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if key:
            out[key] = val
    return out


def download_pages() -> None:
    for material, ids in IDS.items():
        for i in ids:
            dst = ROOT / f"不锈钢海水_{material}_{i}.html"
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            with urlopen(BASE_URL + f"{i}.html", timeout=30) as r:
                dst.write_bytes(r.read())
            print("downloaded", dst.name)


def rebuild_csv() -> None:
    rows = []
    for f in sorted(ROOT.glob("*.html")):
        fields = _extract_fields(f.read_text(encoding="utf-8", errors="ignore"))
        rows.append({
            "数据表": fields.get("所属数据表", ""),
            "材料名称": fields.get("材料名称", ""),
            "材料牌号": fields.get("材料牌号", ""),
            "试验周期_月": fields.get("试验周期(月)", ""),
            "试验地点": fields.get("查询地点", fields.get("试验地点", "")),
            "试验区域": fields.get("试验区域", ""),
            "腐蚀介质": fields.get("腐蚀介质", ""),
            "腐蚀类型": fields.get("腐蚀类型", ""),
            "腐蚀速率_mm_a": fields.get("腐蚀速率(mm/a)", ""),
            "数据来源": fields.get("数据来源", ""),
            "原始文件": f.name,
        })
    df = pd.DataFrame(rows)
    out = ROOT / "国家材料环境腐蚀平台_腐蚀速率_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out, "rows:", len(df))


if __name__ == "__main__":
    download_pages()
    rebuild_csv()
