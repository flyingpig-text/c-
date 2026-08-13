# -*- coding: utf-8 -*-
"""从 Engineering Toolbox 金属比热容表生成清洗 CSV。"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


HTML = Path(r"D:\46884\Documents\数学建模\C题数据\热物性数据\EngineeringToolbox_SpecificHeat_Metals.html")
OUT = Path(r"D:\46884\Documents\数学建模\C题数据\清洗后数据\金属比热容_EngineeringToolbox_clean.csv")


def main() -> None:
    text = HTML.read_text(encoding="utf-8", errors="replace")
    table = re.search(r"<table.*?</table>", text, re.S)
    rows = re.findall(r"<tr.*?</tr>", table.group(0), re.S)
    values: list[tuple[str, float]] = []
    for r in rows[1:]:
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td.*?</td>", r, re.S)]
        if len(cells) >= 2 and cells[0] and cells[1]:
            try:
                values.append((cells[0], float(cells[1])))
            except ValueError:
                continue
    df = pd.DataFrame(values, columns=["材料", "比热容_kJ_kgK"])
    df.insert(2, "比热容_J_kgK", df["比热容_kJ_kgK"] * 1000)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
