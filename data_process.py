# -*- coding: utf-8 -*-
"""
数据清洗 + 特征工程
输入: 原始 DataFrame
输出: 清洗后带全部特征的 DataFrame
"""

import pandas as pd
import numpy as np
from config import *


def _normalize_address(addr):
    """提取地址中的核心机构名，用于地址归一化匹配"""
    import re
    addr = str(addr)
    if ' ' in addr:
        parts = addr.split(' ', 1)
        prefix = parts[0]
        detail = parts[1]
    else:
        prefix = addr
        detail = addr
    orgs = ['供电局', '分公司', '支行', '大厦', '大楼', '中心',
            '学校', '医院', '学院', '局', '部', '院', '集团', '公司', '厂', '所']
    for p in orgs:
        idx = detail.find(p)
        if idx >= 0:
            start = max(0, idx - 20)
            org = detail[start:idx + len(p)].lstrip(' ').rstrip(' ')
            return f'{prefix}|{org}' if prefix != detail else org
    return addr


def data_process(df):
    """数据清洗：8步"""
    initial_rows = len(df)

    # ① 状态过滤
    if COL_ORD_STATUS in df.columns:
        df = df[~df[COL_ORD_STATUS].isin(INVALID_STATUSES)]

    # ② 类型转换（必须先转再过滤）
    # 先清洗中文数字格式：去逗号、处理"万"
    for col in [COL_PUR_QTY, COL_TAX_PRICE, COL_SUB_TTL]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            wan_mask = df[col].str.contains('万', na=False)
            if wan_mask.any():
                df.loc[wan_mask, col] = df.loc[wan_mask, col].str.replace('万', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[wan_mask, col] = df.loc[wan_mask, col] * 10000
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

    # ⑤ 时间转换 + 异常日期过滤（统一UTC时区，兼容CSV带时区时间列）
    df[COL_SUBMIT_TIME] = pd.to_datetime(df[COL_SUBMIT_TIME], errors='coerce', utc=True)
    today = pd.Timestamp.now(tz='UTC')
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

    # P4前向常态基准：该行之前(不含自身)的价格中位数（时间穿越），首笔回退全历史中位数
    df['sku_prev_median'] = (
        df.groupby(COL_SKU_NAME)[COL_TAX_PRICE]
        .apply(lambda g: g.expanding().median().shift(1))
        .reset_index(level=0, drop=True)
    )
    df['sku_prev_median'] = df['sku_prev_median'].fillna(
        df.groupby(COL_SKU_NAME)[COL_TAX_PRICE].transform('median')
    )
    # P4基准升级版：前10笔价格中位数（时间穿越，近期常态），不足3笔回退expanding，再回退全历史
    _p10 = df.groupby(COL_SKU_NAME)[COL_TAX_PRICE].transform(
        lambda g: g.shift(1).rolling(10, min_periods=3).median()
    )
    df['sku_prev_price_10'] = _p10.fillna(df['sku_prev_median'])
    # ===== Q3数量纵横基准 =====
    # 纵向：该SKU全部订单采购量中位数（全量中位数，稳健且同量同判；中位数对极端大单不敏感，无需时间穿越）
    df['sku_qty_median'] = df.groupby(COL_SKU_NAME)[COL_PUR_QTY].transform('median')
    # 横向：同二级部门内该SKU采购量中位数（全量），SKU级样本<3回退同部门同品类中位数，再回退品类全局中位数
    dept_sku_cnt = df.groupby([COL_SEC_DEPT, COL_SKU_NAME], dropna=False)[COL_PUR_QTY].transform('count')
    dept_sku_med = df.groupby([COL_SEC_DEPT, COL_SKU_NAME], dropna=False)[COL_PUR_QTY].transform('median')
    dept_cat_med = df.groupby([COL_SEC_DEPT, COL_FOUR_CAT], dropna=False)[COL_PUR_QTY].transform('median')
    cat_all_med = df.groupby(COL_FOUR_CAT)[COL_PUR_QTY].transform('median')
    df['dept_sku_qty_median'] = np.where(dept_sku_cnt >= 3, dept_sku_med, dept_cat_med)
    df['dept_sku_qty_median'] = pd.Series(df['dept_sku_qty_median']).fillna(cat_all_med).values

    # ===== 价格段内序号（段首过滤：同价连续段内第几笔） =====
    def _seg_rank(g):
        prices = g.values
        seg = np.zeros(len(prices), dtype=int)
        rank = np.ones(len(prices), dtype=int)
        for i in range(1, len(prices)):
            if prices[i] != prices[i - 1]:
                seg[i] = seg[i - 1] + 1
                rank[i] = 1
            else:
                seg[i] = seg[i - 1]
                rank[i] = rank[i - 1] + 1
        return pd.Series(rank, index=g.index)

    df['sku_price_seg_rank'] = (
        df.groupby(COL_SKU_NAME)[COL_TAX_PRICE]
        .apply(_seg_rank)
        .reset_index(level=0, drop=True)
    )
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

    # 采购人SKU多样性（带时间穿越）：买过多少种不同的SKU
    df['buyer_sku_diversity'] = (
        df.groupby(COL_SMTR_NAME)[COL_SKU_NAME]
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

        # 地址归一化
        if COL_REC_ADDRESS in df.columns:
            df['_address_normalized'] = df[COL_REC_ADDRESS].apply(_normalize_address)
        else:
            df['_address_normalized'] = df[COL_SMTR_NAME].astype(str) + '_default'

        # 无效占位符处理
        invalid = ['-', '', '无', 'N/A', 'none', 'null', 'nan']
        if COL_CONSIGNER in df.columns:
            df['_consigner_valid'] = df[COL_CONSIGNER].astype(str)
            mask = df['_consigner_valid'].isin(invalid)
            df.loc[mask, '_consigner_valid'] = df.loc[mask, COL_ORDER_ID].astype(str)
        else:
            df['_consigner_valid'] = df[COL_ORDER_ID].astype(str)
        if COL_PROJ_NAME in df.columns:
            df['_proj_valid'] = df[COL_PROJ_NAME].astype(str)
            mask = df['_proj_valid'].isin(invalid)
            df.loc[mask, '_proj_valid'] = df.loc[mask, COL_ORDER_ID].astype(str)
            # _proj_group: 占位符统一替换为常量，有真实项目名的保留原始值
            df['_proj_group'] = df[COL_PROJ_NAME].astype(str)
            df.loc[df['_proj_group'].isin(invalid), '_proj_group'] = '__NOPROJ__'
        else:
            df['_proj_valid'] = df[COL_ORDER_ID].astype(str)
            df['_proj_group'] = '__NOPROJ__'

        # S1: 同日拆单分组特征 — (采购人, 供应商, SKU, 日, 项目, 地址, 收货人)
        # 项目用原始值(_proj_valid会让"-"被order_id替换而阻断分组,同人项目不会有跨人风险)
        s1_key = [COL_SMTR_NAME, COL_SUP_NAME, COL_SKU_NAME, 'submit_day',
                  COL_PROJ_NAME, '_address_normalized', COL_CONSIGNER]
        s1_key = [c for c in s1_key if c in df.columns]
        if len(s1_key) >= 4:
            df['same_day_order_count'] = df.groupby(s1_key, dropna=False)[COL_ORDER_ID].transform('size')
            df['same_day_merged_amount'] = df.groupby(s1_key, dropna=False)[COL_SUB_TTL].transform('sum')

        # S3: 收货人拆单分组特征 — (收货人, 地址, SKU, 项目, 日)
        s3_key = ['_consigner_valid', COL_SKU_NAME, '_proj_valid', 'submit_day']
        s3_key = [c for c in s3_key if c in df.columns]
        if len(s3_key) >= 3:
            df['consigner_day_order_count'] = df.groupby(s3_key, dropna=False)[COL_ORDER_ID].transform('size')
            df['consigner_day_amount'] = df.groupby(s3_key, dropna=False)[COL_SUB_TTL].transform('sum')

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
