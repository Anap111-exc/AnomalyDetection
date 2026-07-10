# -*- coding: utf-8 -*-
"""
规则引擎：RuleEngine 类 + 13条规则函数
规则优先于 ML，命中确定性违规直接打高分
"""

import pandas as pd
import numpy as np
from config import *


class RuleEngine:
    """规则引擎核心类"""
    def __init__(self):
        self.rules = {}

    def register(self, dim, rule_func):
        """注册一条规则到指定检测器维度"""
        self.rules.setdefault(dim, []).append(rule_func)

    def execute(self, df, dim=None):
        """执行规则，返回 df + 分数列"""
        dims = [dim] if dim else list(self.rules.keys())
        for d in dims:
            score_col = f'{d}_score'
            if score_col not in df.columns:
                df[score_col] = 0.0
            for rule_func in self.rules.get(d, []):
                df = rule_func(df)
        return df


# ======================== 辅助函数 ========================

def _apply_score(df, mask, dim, score_val, reason):
    """在 mask 命中的行上设置分数和原因"""
    score_col = f'{dim}_score'
    reason_col = f'{dim}_rule_reason'
    if score_col not in df.columns:
        df[score_col] = 0.0
    if reason_col not in df.columns:
        df[reason_col] = ''
    df.loc[mask, score_col] = np.maximum(df.loc[mask, score_col], score_val)
    df.loc[mask, 'is_violated'] = np.maximum(
        df.loc[mask, 'is_violated'] if 'is_violated' in df.columns else 0,
        100 if score_val >= 100 else 0
    )
    df.loc[mask, reason_col] = df.loc[mask, reason_col].fillna('') + ';' + reason
    return df


# ======================== 价格规则 ========================

def rule_p1_multiple_supplier_gap(df):
    """P1: 同商品名称同月多供应商价差超阈值 → 仅标记买贵的行=100分（低价笔数≥2才触发，防单笔离群值误报）"""
    if COL_SUP_NAME not in df.columns:
        return df
    if COL_SKU_NAME not in df.columns:
        return df
    grp = df.groupby([COL_SKU_NAME, 'submit_month'])
    _max = grp[COL_TAX_PRICE].transform('max')
    _min = grp[COL_TAX_PRICE].transform('min')
    _cnt = grp[COL_SUP_NAME].transform('nunique')
    # 低价端至少2笔，避免单笔异常值带崩全部
    _min_cnt = (df[COL_TAX_PRICE] == _min).groupby([df[COL_SKU_NAME], df['submit_month']]).transform('sum')
    mask = (_cnt >= 2) & (_min > 0) & (_min_cnt >= 2) & \
           (_max > _min * RULE_PRICE_GAP_RATIO) & \
           (df[COL_TAX_PRICE] == _max)
    return _apply_score(df, mask, 'price', 100, 'P1:同SKU多供应商价差>20%')


def rule_p2_small_sample_deviation(df):
    """P2: 小样本SKU价格偏高 → 50分起（跳过首笔，防止品类均价回填误判）"""
    # 首笔无历史数据，sku_avg_price 被品类均价填充，不可比
    has_history = df['sku_purchase_count'] >= 1
    mask = has_history & \
           (df['is_small_sample'] == 1) & \
           (df['price_deviation_rate'] > RULE_SMALL_SAMPLE_DEVIATION)  # 只标高价偏离
    score = np.minimum(df.loc[mask, 'price_deviation_rate'].abs() * 100, 100)
    score_col = 'price_score'
    if score_col not in df.columns:
        df[score_col] = 0.0
    df.loc[mask, score_col] = np.maximum(df.loc[mask, score_col], np.maximum(score, 50))
    if 'price_rule_reason' not in df.columns:
        df['price_rule_reason'] = ''
    df.loc[mask, 'price_rule_reason'] = df.loc[mask, 'price_rule_reason'].fillna('') + ';P2:小样本SKU偏离>30%'
    return df


def rule_p3_history_max_break(df):
    """P3: 突破历史最高价130% → 85分（跳过首笔，防止品类回填误判）"""
    has_history = df['sku_purchase_count'] >= 1
    mask = has_history & \
           (df['sku_sample_count'] >= 5) & \
           (df['sku_max_price'] > 0) & \
           (df[COL_TAX_PRICE] > df['sku_max_price'] * RULE_HISTORY_MAX_BREAK)
    return _apply_score(df, mask, 'price', 85, 'P3:突破历史最高价130%')


def rule_p4_extreme_low_price(df):
    """P4: 极端低价(<历史中位价50%) → 75分（跳过首笔，防止品类回填误判）"""
    has_history = df['sku_purchase_count'] >= 1
    mask = has_history & \
           (df['sku_sample_count'] >= 5) & \
           (df['sku_median_price'] > 0) & \
           (df[COL_TAX_PRICE] < df['sku_median_price'] * RULE_EXTREME_LOW_RATIO)
    df = _apply_score(df, mask, 'price', 75, 'P4:极端低价(质量风险/关联交易)')
    if 'price_direction' not in df.columns:
        df['price_direction'] = ''
    df.loc[mask, 'price_direction'] = 'low_price_risk'
    return df


# ======================== 数量规则 ========================

def rule_q1_qty_exceeds_history(df):
    """Q1: 单笔数量超历史峰值150% → 90分（仅小样本，大样本由KDE覆盖）"""
    mask = (df['sku_sample_count'] < MIN_SKU_SAMPLES) & \
           (df['sku_sample_count'] >= 5) & \
           (df['sku_max_pur_qty'] > 0) & \
           (df[COL_PUR_QTY] > df['sku_max_pur_qty'] * RULE_QTY_EXCEED_RATIO)
    return _apply_score(df, mask, 'qty', 90, 'Q1:数量超历史峰值150%')


def rule_q2_buyer_daily_total(df):
    """Q2: 采购人日总量超过历史99分位 → 70分兜底（时间保护）"""
    if 'buyer_daily_total_qty' not in df.columns:
        return df
    if COL_SUBMIT_TIME not in df.columns:
        return df
    df_sorted = df.sort_values(COL_SUBMIT_TIME)
    df_sorted['_q2_99pct'] = (
        df_sorted.groupby(COL_SMTR_NAME)['buyer_daily_total_qty']
        .transform(lambda x: x.expanding(min_periods=3).quantile(RULE_QTY_DAILY_PERCENTILE).shift(1))
    )
    mask = (df_sorted['_q2_99pct'] > 0) & \
           (df_sorted['buyer_daily_total_qty'] > df_sorted['_q2_99pct'] * 1.2)
    df_sorted = _apply_score(df_sorted, mask, 'qty', 70, 'Q2:采购人日总量异常')
    # 回写到原df
    df['qty_score'] = df_sorted['qty_score']
    if 'qty_rule_reason' in df_sorted.columns:
        df['qty_rule_reason'] = df_sorted['qty_rule_reason']
    df_sorted.drop(columns=['_q2_99pct'], inplace=True, errors='ignore')
    return df


# ======================== 拆单规则 ========================

def rule_s1_same_day_split(df):
    """S1: 同日拆单 → 100分（品类动态阈值）"""
    if 'same_day_order_count' not in df.columns or 'same_day_merged_amount' not in df.columns:
        return df
    if 'split_threshold' not in df.columns:
        return df
    mask = (df['same_day_order_count'] >= 2) & \
           (df['same_day_merged_amount'] > df['split_threshold']) & \
           (df[COL_SUB_TTL] < df['split_threshold'])
    df = _apply_score(df, mask, 'split', 100, 'S1:同日拆单')
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[mask, 'split_type'] = 'same_day'
    return df


def rule_s2_cross_day_split(df):
    """S2: 跨天拆单(7天滑窗) → 80分（品类动态阈值，滑窗优化）"""
    if 'same_day_order_count' not in df.columns:
        return df
    if COL_ORDER_ID not in df.columns:
        return df
    if 'split_threshold' not in df.columns:
        return df
    df_sorted = df.sort_values(COL_SUBMIT_TIME)

    split_flags = pd.Series(False, index=df_sorted.index)

    for grp_key, grp in df_sorted.groupby([COL_SMTR_NAME, COL_SUP_NAME, COL_SKU_NAME]):
        n = len(grp)
        if n < 2:
            continue
        times = grp[COL_SUBMIT_TIME].values
        amts = grp[COL_SUB_TTL].values
        ths = grp['split_threshold'].values
        idx_list = list(grp.index)

        j = 0
        window_sum = 0.0
        window_orders = set()  # track unique order_ids
        order_ids = grp[COL_ORDER_ID].values

        for i in range(n):
            ti = times[i]
            # 扩展窗口
            window_sum += amts[i]
            window_orders.add(str(order_ids[i]))
            # 收缩窗口
            while j < n and times[j] < ti - pd.Timedelta(days=RULE_SPLIT_WINDOW_DAYS):
                window_sum -= amts[j]
                window_orders.discard(str(order_ids[j]))
                j += 1

            cnt = len(window_orders)
            if cnt >= 2 and window_sum > ths[i] and amts[i] < ths[i]:
                split_flags[idx_list[i]] = True

    df_sorted = _apply_score(df_sorted, split_flags, 'split', 80, 'S2:跨天拆单')
    df['split_score'] = df_sorted['split_score']
    if 'split_rule_reason' in df_sorted.columns:
        df['split_rule_reason'] = df_sorted['split_rule_reason']
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[split_flags.loc[df.index].fillna(False).values, 'split_type'] = 'cross_day'
    return df


# ======================== 短期高频规则 ========================

def rule_s6_short_term_high_freq(df):
    """S6: 短期高频下单 → 7天内≥2笔=80分, 14天内≥3笔=70分（滑窗优化版）"""
    if COL_SKU_NAME not in df.columns or COL_SMTR_NAME not in df.columns:
        return df
    if COL_SUBMIT_TIME not in df.columns:
        return df

    df_sorted = df.sort_values(COL_SUBMIT_TIME)

    flags_7d = pd.Series(False, index=df_sorted.index)
    flags_14d = pd.Series(False, index=df_sorted.index)

    # 按 采购人+SKU 分组，组内滑窗计数
    for grp_key, grp in df_sorted.groupby([COL_SMTR_NAME, COL_SKU_NAME]):
        times = grp[COL_SUBMIT_TIME].values
        n = len(grp)
        if n < 2:
            continue

        # 双指针：对于每行i，找7天/14天窗口内的行数
        idx_list = list(grp.index)
        j_7 = 0
        j_14 = 0
        for i in range(n):
            ti = times[i]
            while j_7 < n and times[j_7] < ti - pd.Timedelta(days=7):
                j_7 += 1
            while j_14 < n and times[j_14] < ti - pd.Timedelta(days=14):
                j_14 += 1
            cnt_7 = i - j_7 + 1
            cnt_14 = i - j_14 + 1

            if cnt_7 >= 2:
                for k in range(j_7, i + 1):
                    flags_7d[idx_list[k]] = True
            if cnt_14 >= 3:
                for k in range(j_14, i + 1):
                    flags_14d[idx_list[k]] = True

    df_sorted = _apply_score(df_sorted, flags_7d, 'split', 80, 'S6a:7天内高频下单')
    df_sorted = _apply_score(df_sorted, flags_14d, 'split', 70, 'S6b:14天内高频下单')

    df['split_score'] = df_sorted['split_score']
    if 'split_rule_reason' in df_sorted.columns:
        df['split_rule_reason'] = df_sorted['split_rule_reason']
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[flags_7d.loc[df.index].fillna(False).values, 'split_type'] = '高频7d'
    df.loc[flags_14d.loc[df.index].fillna(False).values, 'split_type'] = '高频14d'
    return df


# ======================== 新增拆单规则（S3-S5：收货人/地址/项目维度） ========================

def rule_s3_consigner_split(df):
    """S3: 收货人维度拆单 → 80分（标记整组，不限于低于阈值的子单）"""
    for c in ['consigner_day_order_count','consigner_day_amount','split_threshold',
              '_consigner_valid','sku_name','submit_day']:
        if c not in df.columns:
            return df
    group_keys = ['_consigner_valid', 'sku_name', 'submit_day']
    df['_s3_flag'] = (df['consigner_day_order_count'] >= 2) & \
                     (df['consigner_day_amount'] > df['split_threshold'])
    mask = df.groupby(group_keys)['_s3_flag'].transform('any').fillna(False)
    df.drop(columns=['_s3_flag'], inplace=True)
    df = _apply_score(df, mask, 'split', 80, 'S3:收货人维度拆单')
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[mask, 'split_type'] = df.loc[mask, 'split_type'].apply(
        lambda x: 'consigner' if not x else x + ',consigner'
    )
    return df


def rule_s4_address_split(df):
    """S4: 收货地址维度拆单 → 80分（标记整组）"""
    for c in ['address_day_order_count','address_day_amount','split_threshold',
              '_address_valid','sku_name','submit_day']:
        if c not in df.columns:
            return df
    group_keys = ['_address_valid', 'sku_name', 'submit_day']
    df['_s4_flag'] = (df['address_day_order_count'] >= 2) & \
                     (df['address_day_amount'] > df['split_threshold'])
    mask = df.groupby(group_keys)['_s4_flag'].transform('any').fillna(False)
    df.drop(columns=['_s4_flag'], inplace=True)
    df = _apply_score(df, mask, 'split', 80, 'S4:收货地址维度拆单')
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[mask, 'split_type'] = df.loc[mask, 'split_type'].apply(
        lambda x: 'address' if not x else x + ',address'
    )
    return df


def rule_s5_project_split(df):
    """S5: 项目维度拆单 → 85分（标记整组，仅对有效项目名判）"""
    for c in ['proj_day_order_count','proj_day_amount','split_threshold',
              '_proj_valid','sku_name','submit_day',COL_PROJ_NAME]:
        if c not in df.columns:
            return df
    has_proj = ~df[COL_PROJ_NAME].isin(['-', '', 'N/A', '无', None])
    group_keys = ['_proj_valid', 'sku_name', 'submit_day']
    df['_s5_flag'] = has_proj & \
                     (df['proj_day_order_count'] >= 2) & \
                     (df['proj_day_amount'] > df['split_threshold'])
    mask = df.groupby(group_keys)['_s5_flag'].transform('any').fillna(False)
    df.drop(columns=['_s5_flag'], inplace=True)
    df = _apply_score(df, mask, 'split', 85, 'S5:项目维度拆单')
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[mask, 'split_type'] = df.loc[mask, 'split_type'].apply(
        lambda x: 'project' if not x else x + ',project'
    )
    return df


# ======================== 聚量规则 ========================

def rule_c1_supplier_concentration(df):
    """C1: 供应商高浓度 + 多供应商前置 → 70分兜底"""
    if 'buyer_sup_amount_ratio_30d' not in df.columns:
        return df
    if 'buyer_supplier_diversity' not in df.columns:
        return df
    mask = (df['buyer_sup_amount_ratio_30d'] > RULE_CONCEN_RATIO) & \
           (df['buyer_supplier_diversity'] >= 3) & \
           (df['price_deviation_rate'] > 0.4)
    return _apply_score(df, mask, 'concen', 70, 'C1:采购人30天内80%+金额集中于单一供应商且价格偏高')


def rule_c2_high_price_concen(df):
    """C2: 高价+高浓度组合 → 75分（需至少3笔交易，跳过首笔防止品类回填误判）"""
    if 'buyer_sup_amount_ratio_30d' not in df.columns:
        return df
    if 'buyer_sup_order_count_30d' not in df.columns:
        return df
    has_history = df['sku_purchase_count'] >= 1
    mask = has_history & \
           (df['buyer_sup_order_count_30d'] >= 3) & \
           (df['buyer_sup_amount_ratio_30d'] > RULE_CONCEN_PRICE_RATIO) & \
           (df['price_deviation_rate'] > RULE_CONCEN_PRICE_DEVIATION)
    return _apply_score(df, mask, 'concen', 75, 'C2:高价聚量组合')


def rule_c3_high_price_high_qty(df):
    """C3: 高价高量组合 → 80分（需至少3笔交易）"""
    if 'buyer_sup_amount_ratio_30d' not in df.columns:
        return df
    if 'buyer_sup_order_count_30d' not in df.columns:
        return df
    if 'price_score' not in df.columns or 'qty_score' not in df.columns:
        return df
    mask = (df['buyer_sup_order_count_30d'] >= 3) & \
           (df['price_score'] > 60) & \
           (df['qty_score'] > 60) & \
           (df['buyer_sup_amount_ratio_30d'] > 0.5)
    return _apply_score(df, mask, 'concen', 80, 'C3:高价高量组合')


# ======================== 时序规则 ========================

def rule_t1_buyer_monthly_mutation(df):
    """T1: 采购人月单量环比突变(本月>上月×3) → 85分"""
    if 'buyer_monthly_order_count' not in df.columns:
        return df
    # 按 buyer+month 聚合
    buyer_monthly = df.groupby([COL_SMTR_NAME, 'submit_month']).size().reset_index(name='_cnt')
    buyer_monthly = buyer_monthly.sort_values([COL_SMTR_NAME, 'submit_month'])
    buyer_monthly['_prev_cnt'] = buyer_monthly.groupby(COL_SMTR_NAME)['_cnt'].shift(1)

    mut_mask = buyer_monthly['_prev_cnt'] > 0
    buyer_monthly['_is_mut'] = False
    # 正向突变: 本月 > 上月 × 3
    buyer_monthly.loc[mut_mask, '_is_mut'] = (
        buyer_monthly.loc[mut_mask, '_cnt'] > buyer_monthly.loc[mut_mask, '_prev_cnt'] * RULE_TS_BUYER_RATIO
    )
    # 负向突变: 本月 < 上月 × 0.3
    neg_mask = mut_mask & ~buyer_monthly['_is_mut']
    buyer_monthly.loc[neg_mask, '_is_mut'] = (
        buyer_monthly.loc[neg_mask, '_cnt'] < buyer_monthly.loc[neg_mask, '_prev_cnt'] * 0.3
    )

    mutation_months = set(
        buyer_monthly[buyer_monthly['_is_mut']].apply(
            lambda r: (r[COL_SMTR_NAME], r['submit_month']), axis=1
        )
    )

    mask = df.apply(
        lambda r: (r[COL_SMTR_NAME], r['submit_month']) in mutation_months, axis=1
    )
    return _apply_score(df, mask, 'ts', 85, 'T1:采购人月单量环比突变')


def rule_t2_sku_price_mutation(df):
    """T2: SKU月均价环比突变>50% → 80分"""
    if COL_SKU_ID not in df.columns:
        return df
    sku_monthly = df.groupby([COL_SKU_ID, 'submit_month'])[COL_TAX_PRICE].mean().reset_index()
    sku_monthly = sku_monthly.sort_values([COL_SKU_ID, 'submit_month'])
    sku_monthly['_prev'] = sku_monthly.groupby(COL_SKU_ID)[COL_TAX_PRICE].shift(1)

    mut_mask = (sku_monthly['_prev'] > 0) & (
        (sku_monthly[COL_TAX_PRICE] - sku_monthly['_prev']).abs() > sku_monthly['_prev'] * RULE_TS_SKU_PRICE_RATIO
    )

    mutation_set = set(
        sku_monthly[mut_mask].apply(lambda r: (r[COL_SKU_ID], r['submit_month']), axis=1)
    )

    mask = df.apply(lambda r: (r[COL_SKU_ID], r['submit_month']) in mutation_set, axis=1)
    return _apply_score(df, mask, 'ts', 80, 'T2:SKU月均价突变>50%')


def rule_i1_sparse_data_flag(df):
    """I1: 5维度均<30 且 CV>1.0 → 仅标记"""
    if 'sku_price_cv' not in df.columns:
        return df
    # 5维度均 < 30
    low_all = True
    for dim in ['price', 'qty', 'split', 'concen', 'ts']:
        col = f'{dim}_score'
        if col in df.columns:
            low_all = low_all & (df[col] < 30)
    mask = low_all & (df['sku_price_cv'] > 1.0)
    if 'invis_attention' not in df.columns:
        df['invis_attention'] = False
    df.loc[mask, 'invis_attention'] = True
    return df


# ======================== 注册所有规则 ========================

def register_all_rules(engine):
    """向规则引擎注册全部13条规则"""
    engine.register('price', rule_p1_multiple_supplier_gap)
    engine.register('price', rule_p2_small_sample_deviation)
    engine.register('price', rule_p3_history_max_break)
    engine.register('price', rule_p4_extreme_low_price)
    engine.register('qty', rule_q1_qty_exceeds_history)
    engine.register('qty', rule_q2_buyer_daily_total)
    engine.register('split', rule_s1_same_day_split)
    engine.register('split', rule_s2_cross_day_split)
    engine.register('split', rule_s3_consigner_split)
    engine.register('split', rule_s4_address_split)
    engine.register('split', rule_s5_project_split)
    engine.register('split', rule_s6_short_term_high_freq)
    engine.register('concen', rule_c1_supplier_concentration)
    engine.register('concen', rule_c2_high_price_concen)
    engine.register('concen', rule_c3_high_price_high_qty)
    engine.register('ts', rule_t1_buyer_monthly_mutation)
    engine.register('ts', rule_t2_sku_price_mutation)
    engine.register('invis', rule_i1_sparse_data_flag)
