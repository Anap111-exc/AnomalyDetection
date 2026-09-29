# AnomalyDetection

内采订单异常采购行为检测系统：**规则引擎 + 机器学习（KDE / Prophet / DBSCAN）+ 图算法**，多维度识别采购异常。

## 运行形态

| 入口 | 用途 | 输出 |
| --- | --- | --- |
| `run_full_detect.py` | **全品类全量检测**（读全量 CSV，快速模式跳过图算法） | `全品类-全量标注.xlsx`（全部订单 + 异常标注）、`全品类-汇总.xlsx` |
| `detect_categories.py 茶叶,标签机` | **品类专项检测**（指定品类，含图算法提权） | `品类专项/<品类名>-内采异常数据分析.xlsx` |
| `run_schedule.py` | 定时检测（配合 cron，内置默认品类列表） | 同品类专项 |
| `run_qty_full.py` | 仅"数量异常"维度全量检测 | `全量数量异常检测结果_<ts>.xlsx` |
| `main.py` | v2.0 完整流程（4 检测器 + 图算法 + 审计 + 漂移监控） | `异常检测结果_<ts>.xlsx` |

> 全品类全量检测的产物会回写到数仓结果表（回写脚本在本仓库之外，见文末"部署"）。

## 主流程（`run_full_detect.py`）

```
读取 CSV（/ad_dataload/pur_ord_rolling.csv）
  → data_process  数据清洗（8 步）
  → feature_engineer  特征工程（50+ 特征，时间穿越保护）
  → load/apply_whitelist  白名单过滤
  → RuleEngine.execute  规则引擎
  → price_detector / qty_detector / split_detector / concen_detector  四个检测器
  → compute_final_score  融合判定（risk_level / anomaly_type / 各维度 *_type）
  → 输出 全品类-全量标注.xlsx + 全品类-汇总.xlsx
```

## 检测维度

| 维度 | 检测器 | 规则 | ML / 算法 | 结果列 |
| --- | --- | --- | --- | --- |
| 价格异常 | `detectors/price_detector.py` | P1–P4 | per-SKU KDE（CDF 分位） | `price_type` |
| 数量异常 | `detectors/qty_detector.py` | Q1 / Q3 | per-SKU KDE（CDF 分位） | `qty_type` |
| 拆单 | `detectors/split_detector.py` | S1 / S2 / S3 / S6 | 纯规则 | `split_type` |
| 高价聚量 | `detectors/concen_detector.py` | C2–C4 | Prophet 时序价格异常 + DBSCAN 数量聚类 | `concen_type` |

（`detectors/invis_detector.py`、`detectors/timeseries_detector.py` 为可选维度，默认不在 `config.DIMS` 融合范围内。）

## 规则一览

阈值与开关集中在 `config.py`。

- **价格（P）**：P1 同 SKU 多供应商价差（且时间重叠）；P2 小样本 SKU 价格显著偏高；P3 突破历史最高价；P4 突破前向常态基准价。
- **数量（Q）**：Q1 小样本数量超全量中位数倍数；Q3 数量"纵横双超"（纵向=本单位二级部门同 SKU 中位数倍数，横向=全局同 SKU 中位数倍数）。（Q2 已废弃。）
- **拆单（S）**：S1 同日拆单；S2 跨天拆单（窗口天数见 config）；S3 收货人维度拆单；S6 短期高频下单。
- **高价聚量（C）**：C2 高价供应商依赖（95）；C3 高价高量组合（100）；C4 首次涨价 + 数量暴增（100）。
- **时序（T）**：T1 采购人月单量环比突变；T2 SKU 月均价环比突变。
- **隐形（I）**：I1 稀疏数据标记。

拆单/聚量规则命中后会写 `*_rule_reason`（含规则号与中文名），融合层据此生成可读标签。

## 高价聚量 & Prophet 价格异常

- **Prophet 价格异常**（`_prophet_price_anomaly`）：按 `sku_name` 分组，样本 ≥ `PROPHET_MIN_TRAIN` 才做；异常需同时满足
  1. 单价 > Prophet 95% 预测上界 `yhat_upper`；
  2. 单价 > 前向中位价 × `(1 + PROPHET_MIN_MEDIAN_DEV)`（时间穿越基准）；
  3. 段首过滤：同价连续段只保留前 `PROPHET_JUMP_KEEP_N` 笔；
  4. **倒V过滤**：未来 14 天或未来 5 单内须出现回落（≤ 该笔 × `RULE_CONCEN_INVERTED_V_DROP`），否则不算异常（窗口末尾无法确认的行不判）。
- **高价聚量** = 价格信号（Prophet ∪ `price_score≥80`）**AND** 数量信号（`qty_score≥80` ∪ DBSCAN 数量异常），再过同一个倒V过滤；C3/C4 规则聚量同样要求倒V（C2 除外）。
- **新增输出字段**：`confidence_interval`（Prophet `yhat_upper`）与 `deviation = (tax_price - yhat_upper) / yhat_upper`，对 Prophet 覆盖到的行输出，其余为空。

## 数据清洗 & 特征工程（`data_process.py`）

清洗（8 步）：状态过滤 → 数值转换（容错解析"万"/千分位）→ 零值剔除 → `tax_price` 回落 `origin_tax_price` → 时间转换与异常日期过滤 → 关键字段去空 → 去重（明细 ID / 业务键）→ 金额偏差标记。

特征工程：SKU 均价/中位数/方差/最大最小、价格偏离率与 Z-score、价格分位与整数度、供应商排名与溢价、数量偏离率/Z-score、采购人维度（日均/月均单量、供应商与 SKU 多样性）、采购人×供应商金额集中度、订单与拆单分组特征、稀疏度（CV/IQR/Gini/Gap）等；窗口类特征均用 `expanding().shift(1)` 做**时间穿越保护**。

## 白名单（`whitelist.py`）

`whitelist.xlsx`（列：`entity_id`、`reason`、`expiry_date`、`approved_by`）对采购人 / 供应商 / SKU 生效，命中且未过期的行在融合层直接判为"无异常"。

## 融合判定（`fusion.py`）

- 白名单 → 无异常；
- 规则命中（`is_violated`）→ 异常，主类型取分数最高维度（并列时"高价聚量"优先）；
- 否则各维度 `boosted_score`（分数 + 图提权）超阈值（通用 `BINARY_THRESHOLD`，拆单单独 `SPLIT_BINARY_THRESHOLD`）→ 异常；
- 输出 `risk_level`、`anomaly_type` 以及各维度类型列 `price_type/qty_type/split_type/concen_type`。

## 输出字段（`全品类-全量标注.xlsx`）

```
order_id, order_no, ord_item_id, submit_time,
smtr_name, thd_dept_short_name, sec_dept_short_name, sup_short_name,
consigner_name, rec_address, sku_name, proj_name, four_cat_name,
sku_code, stnd_sku_code, confidence_interval, deviation,
tax_price, pur_qty, sub_ttl,
anomaly_type, risk_level,
price_type, qty_type, split_type, concen_type
```

## 环境与运行

```bash
pip install -r requirements.txt

# 可选环境变量
export AD_INPUT_FILE=pur_ord_rolling.csv      # 指定输入文件（默认取数据目录首个 csv/xlsx）
export AD_RESULT_DIR=/path/to/result          # 指定结果目录
export USE_PROPHET=0                           # 关闭 Prophet（默认 1 开启）

python run_full_detect.py                      # 全品类全量
python detect_categories.py "茶叶,标签机"       # 品类专项（含图算法）
python run_schedule.py                          # 定时（默认品类列表）
```

依赖：`pandas`、`numpy`、`scikit-learn`、`scipy`、`networkx`、`openpyxl`、`python-dateutil`、`prophet`（可选，缺失时自动跳过时序检测）。

## 目录结构

```
AnomalyDetection/
├── run_full_detect.py      # 全品类全量检测入口
├── detect_categories.py    # 品类专项检测入口
├── run_schedule.py         # 定时检测入口
├── run_qty_full.py         # 数量维度全量
├── main.py                 # v2.0 完整流程
├── config.py               # 全局配置（路径/列名/阈值/参数）
├── data_process.py         # 清洗 + 特征工程
├── rule_engine.py          # 规则引擎 + 规则注册
├── fusion.py               # 融合判定
├── whitelist.py            # 白名单
├── audit.py / drifts.py / ml_utils.py
├── detectors/              # price / qty / split / concen / timeseries / invis
└── graph/                  # build_graph / graph_algo / boost
```

## 部署（Ti-One / webide）

- 检测侧代码由内部 Git 仓库同步到服务器 `AnomalyDetection/` 目录；更新方式（webide 终端）：
  ```bash
  cd AnomalyDetection
  git config core.autocrlf false
  git fetch origin
  git reset --hard origin/main
  ```
- **取数、回写、调度脚本在本仓库之外**（`ad_dataload/dataload_rolling.py`、`dataresult/resultload_truncate.py`、`ad_dataload/pipeline.py`），手工维护。
- 在 Notebook 中运行前请先重启内核（import 的模块有缓存）；命令行直接 `python xxx.py` 会加载最新代码。
- 回写脚本采用"已存在则只 `TRUNCATE` 清空按列名 `INSERT`、绝不重建表"的方式，保护结果表上的既有设置。
