# -*- coding: utf-8 -*-
"""
全局配置文件
所有阈值、路径、列名、模型参数集中管理
"""

import os
import time

# ==================== 路径（ti-one 平台，可用环境变量覆盖用于本地测试） ====================
BASE_DIR = os.environ.get("ANOMALY_BASE_DIR", "/AnomalyDetection")
DATA_DIR = os.environ.get("AD_DATALOAD_DIR", "/ad_dataload")
RESULT_ROOT = os.environ.get("AD_RESULT_DIR", "/dataresult/品类专项")
INPUT_FILE = ""  # 运行时在 detect_categories.py 中动态定位数据文件
WHITELIST_FILE = os.path.join(BASE_DIR, "whitelist.xlsx")
OUTPUT_FILE = os.path.join(RESULT_ROOT, f"异常检测结果_{int(time.time())}.xlsx")
AUDIT_DIR = RESULT_ROOT

# ==================== 列名映射 ====================
COL_ORDER_ID = 'order_id'
COL_ORDER_NO = 'order_no'
COL_ORD_ITEM_ID = 'ord_item_id'
COL_SUBMIT_TIME = 'submit_time'
COL_ORD_TYPE = 'ord_type_name'
COL_ORD_STATUS = 'ord_status_name'
COL_SMTR_NAME = 'smtr_name'            # 采购人
COL_DEPT = 'thd_dept_short_name'       # 三级部门
COL_SEC_DEPT = 'sec_dept_short_name'   # 二级部门
COL_SEC_DEPT_TYPE = 'sec_dept_type'    # 部门类型
COL_SUP_NAME = 'sup_short_name'        # 供应商
COL_COMPANY = 'company_name'
COL_STORE_NAME = 'store_name'
COL_SKU_ID = 'sku_id'
COL_SKU_CODE = 'sku_code'
COL_SKU_NAME = 'sku_name'
COL_MATERIAL_TYPE = 'material_type_name'
COL_IS_POVT = 'is_povt_alevt'
COL_ONE_CAT = 'one_cat_name'
COL_TWO_CAT = 'two_cat_name'
COL_THREE_CAT = 'three_cat_name'
COL_FOUR_CAT = 'four_cat_name'
COL_MEASURE_UNIT = 'measure_unit_name'
COL_TAX_PRICE = 'tax_price'
COL_ORIGIN_PRICE = 'origin_tax_price'
COL_PUR_QTY = 'pur_qty'
COL_SUB_TTL = 'sub_ttl'
COL_MATERIAL_ZONE = 'material_type'
COL_STORE_PZN_LEVEL = 'store_pzn_level'
COL_APPLY_PEOPLE = 'apply_people_name'   # 申购人
COL_CONSIGNER = 'consigner_name'         # 收货人
COL_PROJ_NAME = 'proj_name'             # 项目名称
COL_REC_ADDRESS = 'rec_address'         # 收货地址

# ==================== 清洗参数 ====================
INVALID_STATUSES = ['交易关闭', '取消', '新建', '待付款']

# ==================== ML 模型参数 ====================
MIN_SKU_SAMPLES = 10
IF_CONTAMINATION = 0.015             # per-SKU IF 预期异常比例 (仅1.5%)
IF_CONTAMINATION_INVIS = 0.02        # 隐形检测 IF 预期异常比例
IF_N_ESTIMATORS = 100
IF_RANDOM_STATE = 42
LOF_N_NEIGHBORS = 20
PRICE_KDE_MIN_CV = 0.03           # 价格CV低于该值跳过KDE（无波动时KDE排名是随机噪声）
PRICE_KDE_MIN_RATIO = 1.05        # 价格须高于中位价该倍数才允许KDE给分（原1.5倍拦死温和涨价）
PROPHET_MIN_MEDIAN_DEV = 0.04     # Prophet价格异常须高于前向基准价4%（放宽微涨边缘，如+4.8%涨价）
PROPHET_JUMP_KEEP_N = 3           # 价格突变段内保留前N笔（段首过滤，后续同价单为新常态不标）

# ==================== 规则引擎参数 ====================
RULE_PRICE_GAP_RATIO = 2.0           # P1: 多供应商价差需2倍
RULE_P1_OVERLAP_DAYS = 7             # P1: 高价笔与低价笔须在±N天内并存（区分涨价与真价差）
RULE_SMALL_SAMPLE_DEVIATION = 0.8    # P2: 偏离需80%
RULE_HISTORY_MAX_BREAK = 1.5         # P3: 需超历史最高价1.5倍
RULE_PREV_MEDIAN_BREAK = 1.15        # P4: 需超前向常态基准价15%
RULE_QTY_EXCEED_RATIO = 4.0          # Q1: 数量超全量中位数倍数(2.5→4:过滤压线误判,保留显著突增)
RULE_QTY_HIST_RATIO = 10.0          # Q3: 纵向突变（超本单位前10笔中位数倍数）
RULE_QTY_DEPT_RATIO = 3.0           # Q3: 横向超配（超同二级部门该SKU中位数倍数）
RULE_QTY_STRONG_LONGI = 20.0        # Q3: 纵向强信号倍数（单条件兜底）
RULE_QTY_STRONG_LATER = 10.0        # Q3: 横向强信号倍数（单条件兜底）
RULE_QTY_KDE_PERCENTILE = 0.95      # 数量KDE门槛分位（默认95%，90%→95%提高精确率）
RULE_SPLIT_MIN_FLOOR = 30            # 动态拆单阈值最低地板（元）：防止品类太便宜时阈值过低
RULE_SPLIT_WINDOW_DAYS = 7           # S2: 跨天拆单窗口
RULE_CONCEN_PRICE_RATIO = 0.8        # C2: 浓度+高价组合
RULE_CONCEN_PRICE_DEVIATION = 0.4    # C2: 价格偏离阈值
RULE_TS_BUYER_RATIO = 4.0            # T1: 采购人月单量环比
RULE_TS_SKU_PRICE_RATIO = 0.6        # T2: SKU均价环比

# ==================== 图算法参数 ====================
GRAPH_ROLLING_DAYS = 30
GRAPH_W1 = 0.7                       # PageRank 权重（个人重要性）
GRAPH_W2 = 0.3                       # 社区权重（圈子风险）

# ==================== 融合参数 ====================
BINARY_THRESHOLD = 90               # 通用阈值（price/qty/concen/invis）
SPLIT_BINARY_THRESHOLD = 80         # 拆单阈值（单独降低，让S6的80分也能命中）

# 图算法 boost + risk_level 阈值统一由此参数控制，不再有复杂的加权融合

# ==================== 审计 ====================
AUDIT_TOP_N = 50

# ==================== 检测器维度 ====================
DIMS = ['price', 'qty', 'split', 'concen']  # 移除 ts（与价格/数量强重合）、invis（无监督不可解释）
