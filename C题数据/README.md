# C 题数据包：海底数据中心散热优化（2021 MathorCup C 题）

下载时间：2026-08-11；2026-08-13 补充下载 MIT 海水物性表/函数源码、香港 2021 逐时潮汐、金属比热容表、HYCOM 2021 海流、国家材料环境腐蚀平台 304/316L/1Cr18Ni9Ti 明细。所有文件均来自公开数据源，版权归原机构所有，仅供学习与竞赛使用。

## 目录结构

- `海洋环境数据/`
  - `WOA18/woa18_documentation.pdf`：World Ocean Atlas 2018 使用说明文档（NOAA NCEI）
  - `WOA18_南海温度剖面.csv`：珠海、陵水两点 0-5500 m 年气候态温度剖面（WOA18, 1981-2010, 0.25°，站点附近 0.75° 海域平均）
  - `WOA18_南海盐度剖面.csv`：珠海、陵水两点年气候态盐度剖面（WOA18, 1981-2010, 0.25°，站点附近 0.75° 海域平均）
  - `WOA18_1981-2010_月均表层温度_南海站点.csv`：两站点逐月表层温度气候态
  - `ERSST_v5/`：NOAA ERSST v5 逐月 SST，2020-01 至 2021-12（2° 全球网格）
  - `ERSST_v5_2020-2021_南海站点SST.csv`：由上述文件提取的两站点与南海区域月均 SST
  - `OISST_每日SST/`：NOAA OISST v2.1 日海温目录。本次下载时 NOAA 服务器限速严重，日文件未能完整下完；季节变化请直接用 `ERSST_v5_2020-2021_南海站点SST.csv` 或 WOA18 月均表层温度。原始下载地址：https://www.ncei.noaa.gov/products/optimum-interpolation-sst
  - `潮汐/HKO_ChekLapKokE_2026_hourly_tide.csv`：香港赤鱲角东 2026 年逐小时天文潮预报（米）
  - `潮汐/HKO_ChekLapKokE_2021_hourly_tide.html / .csv`：香港赤鱲角东 2021 年逐小时天文潮预报（米），与 2026 版并存
  - `潮汐/*.html`：香港天文台 2026 年赤鱲角东、长洲、芝麻湾高低潮与逐时潮高原始网页
  - `HYCOM_GOFS3.1_2021_南海站点海流/`：HYCOM GOFS 3.1（1/12°）2021 年珠海/陵水逐日 0Z 海流（含全深度剖面）
  - `GODAS_2021_南海站点海流/`：NCEP GODAS 2021 年月均海流（珠海/陵水，1/3° 网格邻域均值）
- `热物性数据/`
  - `EngineeringToolbox_Air_Properties.html`：空气热物性表
  - `EngineeringToolbox_Water_Properties.html`：水/海水密度参考
  - `MIT_Seawater_Properties.html`：海水热物性公式与计算器（附 Sharqawy 2010 论文）
  - `MIT_Seawater_Property_Tables_r2b_2023c.pdf`：MIT 海水物性数值表（0-120℃、多盐度、P0/7/12 MPa）
  - `SEAWATER_v3.1.5_07Aug24.zip`：MIT 海水物性函数源码（MATLAB/EES/VBA）
  - `EngineeringToolbox_SpecificHeat_Metals.html`：金属比热容表
- `材料数据/`
  - `EngineeringToolbox_ThermalConductivity_Metals.html`：金属导热系数表
- `数据中心资料/`
  - `Microsoft_ProjectNatick.html`：微软海底数据中心官方页面
  - `IEA_EnergyEfficiency2023.pdf`：IEA 2023 能效报告（含数据中心耗电与 PUE 数据）
  - `gov_新型数据中心三年行动计划.html / .pdf`：工信部等四部门政策原文
- `处理脚本/`
  - `process_data.py`：把 HKO 潮汐网页、ERSST、WOA18 本地文件转成 CSV
  - `fetch_woa18_remote.py`：从 APDRC OPeNDAP 远程拉取 WOA18 并生成 CSV
  - `fetch_hycom_ncss_daily.py`：从 HYCOM THREDDS NCSS 拉取 2021 年站点逐日海流
  - `fetch_hycom_2021_currents.py`：从 APDRC OPeNDAP 拉取 HYCOM 2021 海流（备用脚本）
  - `fetch_godas_2021_currents.py`：从 NOAA PSL 拉取 GODAS 2021 年月均海流（稳定兜底）
  - `fetch_woa18_salinity.py`：拉取 WOA18 站点盐度剖面并计算 20℃ 等温线深度
  - `build_seawater_site_properties.py`：按站点盐度重算 20℃ 海水热物性
  - `build_corrosion_more.py`：逐页扫描黑色/有色金属水环境腐蚀目录并下载明细页
  - `build_seawater_properties.py`：按 MIT 公式生成 35 g/kg 海水热物性表
  - `build_hko_2021_tide.py`：解析香港 2021 逐时潮汐
  - `build_metal_specific_heat.py`：解析金属比热容表
  - `build_corrosion_supplement.py`：下载并解析 304/316L/1Cr18Ni9Ti 海水腐蚀明细

## 数据源清单

| 数据 | 原始来源 |
|---|---|
| WOA18 温盐气候态 | https://www.ncei.noaa.gov/products/world-ocean-atlas （经 APDRC OPeNDAP 拉取） |
| OISST v2.1 | https://www.ncei.noaa.gov/products/optimum-interpolation-sst |
| ERSST v5 | https://www.ncei.noaa.gov/data/ersst/ |
| 潮汐预报 | https://www.hko.gov.hk/en/tide/predtide.htm （香港天文台） |
| 空气/水/材料热物性 | https://www.engineeringtoolbox.com/ |
| 海水热物性 | https://web.mit.edu/seawater/ |
| 海水热物性数值表/源码 | https://web.mit.edu/seawater/ （Tables PDF、SEAWATER zip） |
| 2021 海流（GOFS 3.1） | https://www.hycom.org/dataserver ；APDRC OPeNDAP：http://apdrc.soest.hawaii.edu/dods/public_data/Model_output/HYCOM/gofs3.1/ |
| 2021 月均海流（GODAS） | https://psl.noaa.gov/data/gridded/data.godas.html （ucur/vcur.2021.nc） |
| Project Natick | https://natick.research.microsoft.com/ |
| IEA 数据中心报告 | https://www.iea.org/energy-system/buildings/data-centres-and-data-transmission-networks |
| 政策文件 | https://www.gov.cn/zhengce/zhengceku/2021-07/14/content_5624964.htm |

## 需要注册或动态页面的数据源

- Copernicus Marine Service（CMEMS）：免费注册后下载高分辨率海温/海流：https://marine.copernicus.eu/
- 国家海洋科学数据中心（中国近海实测温盐、潮汐）：https://mds.nmdis.org.cn/ （需实名注册）
- MatWeb 材料数据库：https://www.matweb.com/ （需注册，脚本访问返回 403）
- SMM 上海有色网、生意社：金属价格行情页面为动态数据，不适合直接存档：https://www.smm.cn/、https://www.100ppi.com/

## 备注

- WOA18 全球 0.25° 原始文件约 730 MB/卷，本包经 APDRC OPeNDAP 只拉取南海站点附近小区域，节省空间。
- ERSST 网格为 2°，站点值取最近网格点；OISST 网格为 0.25°。站点坐标：珠海 (113.75°E, 22.25°N)、陵水 (110.0°E, 18.5°N)。
- 潮汐数据为天文潮预报，不含风暴潮余水位；单位 m（香港潮汐基准面，即 Chart Datum 附近）。
- 潮汐现同时保留 2021 与 2026 香港赤鱲角东版本；站点仍为香港，作为珠海/陵水代理使用需在论文中说明。
- HYCOM NCSS 返回的 u/v 为 Int16 打包值（scale_factor=0.001），清洗 CSV 已乘 0.001 转为 m/s；数据为逐日 0Z 快照，非逐时。
- HYCOM 2021 逐日抓取因服务器限流仅完成部分日期（珠海 151 天、陵水 162 天，约 1–6 月）；全年覆盖由 GODAS 月均（12 个月）补齐，GODAS 为 1/3° 网格、站点邻域海洋格点均值。
- WOA18 盐度剖面已补（珠海/陵水，0.25° 气候态）；20℃ 等温线深度按剖面线性外推：陵水约 102.6 m、珠海约 108.9 m（珠海超出有效剖面 50 m，仅参考）。
- 黑色/有色金属水环境腐蚀明细已抓取 724 条，多数只有试验元数据（材料/周期/地点/区域/腐蚀类型），数值腐蚀速率仍需文献或平台权限。
