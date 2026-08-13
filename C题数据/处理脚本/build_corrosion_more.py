# -*- coding: utf-8 -*-
"""从国家材料环境腐蚀平台抓取更多黑色/有色金属水环境腐蚀条目。

目录：
  01010301 黑色金属水环境腐蚀数据
  01010302 有色金属水环境腐蚀数据
站内搜索接口已失效，此处改为逐页扫描目录并按材料关键词筛选明细页。
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(r"D:\46884\Documents\数学建模\C题数据\补充材料数据\腐蚀速率")
CATS = {
    "01010301": ("黑色金属水环境", ["Q235", "16Mn", "Q345", "2205", "A3", "20钢", "45钢"]),
    "01010302": ("有色金属水环境", ["铝", "铜", "钛", "TC4", "TA2", "B10", "B30", "T2", "QBe2", "LY12", "LC4", "6061", "5052", "青铜", "黄铜", "白铜"]),
}
INDEX = "http://data.ecorr.org.cn/edata/01/0101/010103/{cat}/index{page}.html"


def get(url: str, timeout: float = 20) -> str:
    req = Request(url, headers={"User-Agent": "math-modeling-c-task/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def page_count(text: str) -> int:
    m = re.search(r"页次：\d+/(\d+)", text)
    return int(m.group(1)) if m else 1


def entries(text: str) -> list[tuple[str, str]]:
    return re.findall(r'<a href="([^"]+)"[^>]*class="sys_url">([^<]+)</a>', text)


def sanitize(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "_", name).strip("_")


def main() -> None:
    picked: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for cat, (label, keywords) in CATS.items():
            first = get(INDEX.format(cat=cat, page=""))
            total = page_count(first)
            pages = [first] + [get(INDEX.format(cat=cat, page=f"_{i}")) for i in range(2, total + 1)]
            for text in pages:
                for href, title in entries(text):
                    if any(k.lower() in title.lower() for k in keywords):
                        picked.append((cat, label, href, title))
        for cat, label, href, title in picked:
            url = urljoin(INDEX.format(cat=cat, page=""), href)
            fid = href.rsplit("/", 1)[-1].removesuffix(".html")
            dst = ROOT / f"{label}_{fid}.html"
            if dst.exists() and dst.stat().st_size > 1000:
                continue
            futures[pool.submit(get, url)] = dst

        for fut in as_completed(futures):
            dst = futures[fut]
            try:
                dst.write_bytes(fut.result().encode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print("failed", dst.name, exc)

    print("picked entries:", len(picked))

    # 重建腐蚀速率 CSV（含已有 304/316L/321/2Cr13 页面）
    pat = re.compile(
        r"<div align='left'>(.*?)</div></td>\s*"
        r"<td><div class=\"col-md-4\">\s*(.*?)\s*</div></td>",
        re.S,
    )
    rows = []
    for f in sorted(ROOT.glob("*.html")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        fields = {}
        for m in pat.finditer(text):
            key = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            val = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if key:
                fields[key] = val
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
    main()
