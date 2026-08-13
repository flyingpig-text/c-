# -*- coding: utf-8 -*-
"""从国家材料环境腐蚀平台已下载明细页中补抓“腐蚀失厚率(μm/a)”。

原清洗脚本只读取“腐蚀速率(mm/a)”字段，导致黑色/有色金属水环境
716 条记录数值为空；本脚本按页面 metadata 表格结构补抓失厚率
（页面中“腐蚀失厚率”有 “μm/a” 与 “um/a” 两种写法，统一归一化处理）。
同时回填 `国家材料环境腐蚀平台_腐蚀速率_clean.csv` 的
腐蚀介质、腐蚀失厚率、腐蚀失重率与空缺的腐蚀速率(mm/a)。
"""

from __future__ import annotations

import glob
import csv
import os
import re


ROOT = r"D:\46884\Documents\数学建模\C题数据\补充材料数据\腐蚀速率"
CLEAN_CSV = os.path.join(ROOT, "国家材料环境腐蚀平台_腐蚀速率_clean.csv")
LABEL_RE = re.compile(
    r"<td><div[^>]*>([^<]+)</div></td>\s*"
    r"<td><div[^>]*>(.*?)</div></td>",
    re.S | re.I,
)


def parse_page(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in LABEL_RE.finditer(text):
        label = m.group(1).strip()
        value = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        label = label.replace("μm", "um").replace("µ", "u")
        if label and label not in out:
            out[label] = value
    return out


def main() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "*.html")))
    rows: list[dict[str, str]] = []
    n_with = 0
    for f in files:
        page = parse_page(open(f, encoding="utf-8", errors="ignore").read())
        loss = page.get("腐蚀失厚率(um/a)", "")
        if loss:
            n_with += 1
        rows.append({
            "文件": os.path.basename(f),
            "材料": page.get("材料名称", ""),
            "周期_月": page.get("试验周期(月)", ""),
            "地点": page.get("试验地点", ""),
            "环境": page.get("环境类别", ""),
            "区域": page.get("试验区域", ""),
            "腐蚀类型": page.get("腐蚀类型", ""),
            "失厚率_um_a": loss,
            "失重率_g_cm2_a": page.get("腐蚀失重率(g/cm2·a)", ""),
            "来源": page.get("数据来源", ""),
        })
    print(f"总页面 {len(files)}，有失厚率数值 {n_with}")
    keys = ("Q235", "2205", "2A12", "LY12", "LC4", "TC4", "5052", "6061",
            "7075", "316", "304")
    for r in rows:
        mat = r["材料"]
        if any(k.lower() in mat.lower() for k in keys):
            print(" | ".join(str(r[k]) for k in (
                "材料", "周期_月", "地点", "环境", "区域",
                "失厚率_um_a", "失重率_g_cm2_a", "文件")))

    with open(CLEAN_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        out_rows = list(reader)
    for row in out_rows:
        page = parse_page(open(
            os.path.join(ROOT, row["原始文件"]), encoding="utf-8",
            errors="ignore").read())
        if not row.get("腐蚀介质"):
            row["腐蚀介质"] = page.get("环境类别", "")
        loss = page.get("腐蚀失厚率(um/a)", "").strip()
        mass = page.get("腐蚀失重率(g/cm2·a)", "").strip()
        row["腐蚀失厚率_um_a"] = loss
        row["腐蚀失重率_g_cm2_a"] = mass
        if not row.get("腐蚀速率_mm_a") and loss:
            try:
                row["腐蚀速率_mm_a"] = f"{float(loss) / 1000.0:.6g}"
            except ValueError:
                pass
    if "腐蚀失厚率_um_a" not in fields:
        fields += ["腐蚀失厚率_um_a", "腐蚀失重率_g_cm2_a"]
    with open(CLEAN_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    filled = sum(1 for r in out_rows if r.get("腐蚀失厚率_um_a"))
    print(f"已回填 {CLEAN_CSV}：失厚率非空 {filled} 行")


if __name__ == "__main__":
    main()
