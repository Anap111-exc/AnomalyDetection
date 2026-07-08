# -*- coding: utf-8 -*-
"""
数据清洗 + 特征工程
输入: 原始 DataFrame
输出: 清洗后带全部特征的 DataFrame
"""

import pandas as pd
import numpy as np
from config import *


def data_process(df):
    """数据清洗：8步"""
    initial_rows = len(df)

    # ① 状态过滤
    if COL_ORD_STATUS in df.columns:
        df = df[~df[COL_ORD_STATUS].isin(INVALID_STATUSES)]

    # ② 类型转换（必须先转再过滤）
    df[COL_PUR_QTY] = pd.to_numeric(df[COL_PUR_QTY], errors='coerce').fillna(0).astype(int)
    df[COL_TAX_PRICE] = pd.to_numeric(df[COL_TAX_PRICE], errors='coerce')
    df[COL_SUB_TTL] = pd.to_numeric(df[COL_SUB_TTL], errors='coerce')

    # ③ 零值剔除
    df = df[(df[COL_PUR_QTY] > 0) & (df[COL_SUB_TTL] > 0)]

    # ④ tax_price 空值回退到 origin_tax_price
    if COL_ORIGIN_PRICE in df.columns:
        df[COL_TAX_PRICE] = df[COL_TAX_PRICE].fillna(
            pd.to_numeric(df[COL_ORIGIN_PRICE], errors='coerce')
        )

    # ⑤ 时间转换 + 异常日期过滤
    df[COL_SUBMIT_TIME] = pd.to_datetime(df[COL_SUBMIT_TIME], errors='coerce')
    today = pd.Timestamp.now()
    df = df[(df[COL_SUBMIT_TIME] >= '2020-01-01') & (df[COL_SUBMIT_TIME] <= today)]

    # ⑥ 关键字段去空
    key_cols = [COL_SUBMIT_TIME, COL_SKU_ID, COL_TAX_PRICE, COL_PUR_QTY,
                COL_SUB_TTL, COL_SMTR_NAME, COL_SUP_NAME, COL_DEPT, COL_FOUR_CAT]
    key_cols = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=key_cols)

    # ⑦ 去重
    if COL_ORD_ITEM_ID in df.columns:
        df = df.drop_duplicates(subset=[COL_ORD_ITEM_ID])

    # ⑧ 数值有效性 + 金额偏差标记
    df = df[df[COL_TAX_PRICE] > 0]
    df = df[df[COL_SUB_TTL] > 0]
    expected_amt = df[COL_PUR_QTY] * df[COL_TAX_PRICE]
    df['amount_deviation_flag'] = (abs(expected_amt - df[COL_SUB_TTL]) / expected_amt.replace(0, np.nan) > 0.05).astype(int)

    cleaned_rows = len(df)
    print(f"  数据清洗: {initial_rows} -> {cleaned_rows} 行 "
          f"(过滤 {initial_rows - cleaned_rows} 行)")

    return df.copy()


def feature_engineer(df):
    """特征工程：50+ 特征，带时间穿越保护"""
    df = df.sort_values(COL_SUBMIT_TIME).reset_index(drop=True)

    # ======================== A. 时间扩展 ========================
    dt = df[COL_SUBMIT_TIME]
    df['submit_year'] = dt.dt.year
    df['quarter'] = dt.dt.quarter
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['weekday'] = dt.dt.weekday
    df['is_weekend'] = (dt.dt.weekday >= 5).astype(int)
    df['submit_day'] = dt.dt.strftime('%Y%m%d')
    df['submit_month'] = dt.dt.strftime('%Y%m')
    df['submit_week'] = dt.dt.strftime('%Y%W')
    df['is_month_end'] = (dt.dt.days_in_month - dt.dt.day <= 2).astype(int)
    try:
        df['hour'] = dt.dt.hour
        df['is_off_hour'] = ((df['hour'] < 7) | (df['hour'] >= 22)).astype(int)
    except Exception:
        df['hour'] = 12
        df['is_off_hour'] = 0

    # ======================== 聚合工具函数 ========================
    def exp_mean(grp):
        return grp.expanding(min_periods=1).mean().shift(1)
    def exp_median(grp):
        return grp.expanding(min_periods=1).median().shift(1)
    def exp_std(grp):
        return grp.expanding(min_periods=2).std().shift(1)
    def exp_max(grp):
        return grp.expanding(min_periods=1).max().shift(1)
    def exp_min(grp):
        return grp.expanding(min_periods=1).min().shift(1)

    # ======================== B. SKU 维度聚合 ========================
    sku_mappings = [
        (COL_TAX_PRICE, exp_mean, 'sku_avg_price'),
        (COL_TAX_PRICE, exp_median, 'sku_median_price'),
        (COL_TAX_PRICE, exp_std, 'sku_std_price'),
        (COL_TAX_PRICE, exp_max, 'sku_max_price'),
        (COL_TAX_PRICE, exp_min, 'sku_min_price'),
        (COL_PUR_QTY, exp_mean, 'sku_avg_pur_qty'),
        (COL_PUR_QTY, exp_std, 'sku_std_pur_qty'),
        (COL_PUR_QTY, exp_max, 'sku_max_pur_qty'),
    ]
    for col, func, name in sku_mappings:
        df[name] = df.groupby(COL_SKU_NAME)[col].transform(func)

    # sku_purchase_count（带时间穿越）
    df['sku_purchase_count'] = (
        df.groupby(COL_SKU_NAME).cumcount()
    )
    # sku_supplier_count（带时间穿越）
    df['_tmp_sup'] = df[COL_SUP_NAME].astype(str)
    df['sku_supplier_count'] = (
        df.groupby(COL_SKU_NAME)['_tmp_sup']
        .apply(lambda g: (~g.duplicated()).cumsum().shift(1).fillna(1))
        .reset_index(level=0, drop=True)
    )

    # 品类均值（用品类兜底首次出现的SKU）
    df['cat_avg_price'] = df.groupby(COL_FOUR_CAT)[COL_TAX_PRICE].transform(exp_mean)
    # 动态拆单阈值：SKU中位数优先，样本不足3笔用品类中位数兜底
    df['_sku_median'] = df.groupby(COL_SKU_NAME)[COL_SUB_TTL].transform('median')
    df['_cat_median'] = df.groupby(COL_FOUR_CAT)[COL_SUB_TTL].transform('median')
    df['_sku_count'] = df.groupby(COL_SKU_NAME)[COL_SKU_NAME].transform('count')
    df['split_threshold'] = np.where(df['_sku_count'] >= 3, df['_sku_median'], df['_cat_median'])
    df['split_threshold'] = df['split_threshold'].clip(lower=RULE_SPLIT_MIN_FLOOR)
    for col in ['sku_avg_price', 'sku_median_price', 'sku_max_price', 'sku_min_price']:
        df[col] = df[col].fillna(df['cat_avg_price'])

    df['sku_avg_pur_qty'] = df['sku_avg_pur_qty'].fillna(1)
    df['sku_std_pur_qty'] = df['sku_std_pur_qty'].fillna(1)
    df['sku_max_pur_qty'] = df['sku_max_pur_qty'].fillna(df[COL_PUR_QTY])
    df['sku_std_price'] = df['sku_std_price'].fillna(1)

    # 样本量
    df['sku_sample_count'] = df.groupby(COL_SKU_NAME)[COL_SKU_NAME].transform('count')
    df['is_small_sample'] = (df['sku_sample_count'] < MIN_SKU_SAMPLES).astype(int)

    # ======================== C. 价格维度特征 ========================
    eps = 0.01
    df['price_deviation_rate'] = (
        (df[COL_TAX_PRICE] - df['sku_avg_price']) / df['sku_avg_price'].clip(lower=eps)
    )
    df['price_zscore'] = (
        (df[COL_TAX_PRICE] - df['sku_avg_price']) / df['sku_std_price'].clip(lower=eps)
    )
    df['price_vs_max'] = df[COL_TAX_PRICE] / df['sku_max_price'].clip(lower=eps)
    df['price_vs_median'] = df[COL_TAX_PRICE] / df['sku_median_price'].clip(lower=eps)
    df['price_vs_category'] = df[COL_TAX_PRICE] / df['cat_avg_price'].clip(lower=eps)

    # 价格分位（简化：当前价在 [min, max] 区间的分位）
    price_range = (df['sku_max_price'] - df['sku_min_price']).clip(lower=eps)
    df['price_percentile'] = (
        (df[COL_TAX_PRICE] - df['sku_min_price']) / price_range
    )
    df['price_percentile'] = df['price_percentile'].clip(0, 1)

    # 整数度
    def calc_roundness(p):
        try:
            s = str(int(p))
            return len(s) - len(s.rstrip('0'))
        except Exception:
            return 0
    df['price_roundness'] = df[COL_TAX_PRICE].apply(calc_roundness)

    # 供应商价格排名和溢价
    df['sup_price_rank_in_sku'] = (
        df.groupby([COL_SKU_NAME, COL_SUP_NAME])[COL_TAX_PRICE]
        .transform('mean')
        .groupby(df[COL_SKU_NAME])
        .rank(method='dense', ascending=True)
    )
    df['sup_price_premium'] = (
        df[COL_TAX_PRICE] /
        df.groupby(COL_SKU_NAME)[COL_TAX_PRICE].transform('min').clip(lower=eps)
    )

    # ======================== D. 数量维度特征 ========================
    df['qty_deviation_rate'] = (
        (df[COL_PUR_QTY] - df['sku_avg_pur_qty']) / df['sku_avg_pur_qty'].clip(lower=eps)
    )
    df['qty_zscore'] = (
        (df[COL_PUR_QTY] - df['sku_avg_pur_qty']) / df['sku_std_pur_qty'].clip(lower=eps)
    )

    # ======================== E. 采购人维度特征 ========================
    df['buyer_avg_order_amount'] = (
        df.groupby(COL_SMTR_NAME)[COL_SUB_TTL].transform(exp_mean)
    )
    global_avg = df[COL_SUB_TTL].mean()
    df['buyer_avg_order_amount'] = df['buyer_avg_order_amount'].fillna(global_avg)

    df['buyer_daily_total_qty'] = (
        df.groupby([COL_SMTR_NAME, 'submit_day'])[COL_PUR_QTY].transform('sum')
    )
    df['buyer_daily_order_count'] = (
        df.groupby([COL_SMTR_NAME, 'submit_day'])[COL_ORDER_ID].transform('nunique')
        if COL_ORDER_ID in df.columns else 1
    )
    df['buyer_monthly_order_count'] = (
        df.groupby([COL_SMTR_NAME, 'submit_month'])[COL_ORDER_ID].transform('nunique')
        if COL_ORDER_ID in df.columns else 1
    )

    # 供应商多样性（带时间穿越）
    df['buyer_supplier_diversity'] = (
        df.groupby(COL_SMTR_NAME)[COL_SUP_NAME]
        .apply(lambda g: (~g.astype(str).duplicated()).cumsum().shift(1).fillna(1))
        .reset_index(level=0, drop=True)
    )

    # ======================== F. 采购人x供应商维度 ========================
    # 简化版：按 buyer x supplier 累积金额占比
    df['_buyer_cumsum'] = df.groupby(COL_SMTR_NAME)[COL_SUB_TTL].transform(
        lambda x: x.expanding().sum().shift(1).fillna(0)
    )
    df['_buyer_sup_cumsum'] = (
        df.groupby([COL_SMTR_NAME, COL_SUP_NAME])[COL_SUB_TTL]
        .transform(lambda x: x.expanding().sum().shift(1).fillna(0))
    )
    df['buyer_sup_amount_ratio_30d'] = (
        df['_buyer_sup_cumsum'] / df['_buyer_cumsum'].clip(lower=1)
    )
    df['buyer_sup_amount_ratio_30d'] = df['buyer_sup_amount_ratio_30d'].fillna(0).clip(0, 1)

    # 交易笔数
    df['buyer_sup_order_count_30d'] = (
        df.groupby([COL_SMTR_NAME, COL_SUP_NAME]).cumcount()
    )

    # ======================== G. 订单维度特征 ========================
    if COL_ORDER_ID in df.columns:
        df['order_total'] = df.groupby(COL_ORDER_ID)[COL_SUB_TTL].transform('sum')
        df['order_sku_count'] = df.groupby(COL_ORDER_ID)[COL_SKU_ID].transform('nunique')
        df['order_is_single_sku'] = (df['order_sku_count'] == 1).astype(int)

        # 同日同组（按sku_name分组）
        split_key = [COL_SMTR_NAME, COL_SUP_NAME, COL_SKU_NAME, 'submit_day']
        split_key = [c for c in split_key if c in df.columns]
        if len(split_key) >= 3:
            df['same_day_order_count'] = df.groupby(split_key)[COL_ORDER_ID].transform('nunique')
            df['same_day_merged_amount'] = df.groupby(split_key)[COL_SUB_TTL].transform('sum')

        # 7天滑窗（简化：按 buyer+supplier+sku 滚动计数）
        df['_buyer_sup_sku'] = df[COL_SMTR_NAME].astype(str) + '|' + df[COL_SUP_NAME].astype(str) + '|' + df[COL_SKU_NAME].astype(str)
        df['_rownum'] = df.groupby('_buyer_sup_sku').cumcount() + 1
        df['same_7d_order_count'] = df['_rownum'].clip(upper=10)  # 简化上限
        df['same_7d_merged_amount'] = df.groupby('_buyer_sup_sku')[COL_SUB_TTL].transform('sum')

        # 新增：收货人维度拆单特征
        consigner_key = [c for c in [COL_CONSIGNER, COL_SKU_NAME, 'submit_day'] if c in df.columns]
        if len(consigner_key) >= 2:
            df['consigner_day_order_count'] = df.groupby(consigner_key)[COL_ORDER_ID].transform('nunique')
            df['consigner_day_amount'] = df.groupby(consigner_key)[COL_SUB_TTL].transform('sum')

        # 新增：收货地址维度拆单特征
        address_key = [c for c in [COL_REC_ADDRESS, COL_SKU_NAME, 'submit_day'] if c in df.columns]
        if len(address_key) >= 2:
            df['address_day_order_count'] = df.groupby(address_key)[COL_ORDER_ID].transform('nunique')
            df['address_day_amount'] = df.groupby(address_key)[COL_SUB_TTL].transform('sum')

        # 新增：项目名称维度拆单特征（仅对有项目名的行生效）
        proj_key = [c for c in [COL_PROJ_NAME, COL_SKU_NAME, 'submit_day'] if c in df.columns]
        if len(proj_key) >= 2:
            df['proj_day_order_count'] = df.groupby(proj_key)[COL_ORDER_ID].transform('nunique')
            df['proj_day_amount'] = df.groupby(proj_key)[COL_SUB_TTL].transform('sum')

    # ======================== H. 稀疏度特征 ========================
    def _cv(x):
        return x.std(ddof=0) / x.mean() if x.mean() > 0 and len(x) > 1 else 0
    def _iqr_sparse(x):
        if len(x) < 2 or x.mean() == 0: return 0
        q1, q3 = np.percentile(x, [25, 75])
        return (q3 - q1) / x.mean()
    def _gini(x):
        if len(x) < 2 or x.sum() == 0: return 0
        s = np.sort(x)
        n = len(s)
        cs = np.cumsum(s)
        return (n + 1 - 2 * cs.sum() / cs[-1]) / n
    def _gap(x):
        if len(x) < 2: return 0
        s = np.sort(x)
        rng = s[-1] - s[0]
        return np.diff(s).max() / rng if rng > 0 else 0

    sparsity = df.groupby(COL_SKU_NAME)[COL_TAX_PRICE].agg(
        sku_price_cv=_cv, sku_price_iqr=_iqr_sparse,
        sku_price_gini=_gini, sku_price_gap=_gap
    ).reset_index()

    sparsity = sparsity.set_index(COL_SKU_NAME)
    for c in ['sku_price_cv', 'sku_price_iqr', 'sku_price_gini', 'sku_price_gap']:
        df[c] = df[COL_SKU_NAME].map(sparsity[c]).fillna(0)

    # ======================== I. 清理 ========================
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 删除临时列
    tmp_cols = ['_tmp_sup', '_buyer_cumsum', '_buyer_sup_cumsum',
                '_buyer_sup_sku', '_rownum', '_sku_median', '_cat_median', '_sku_count']
    df.drop(columns=[c for c in tmp_cols if c in df.columns], inplace=True, errors='ignore')

    print(f"  特征工程完成: {len(df.columns)} 列 (含 {len(numeric_cols)} 个数值列)")
    return df
