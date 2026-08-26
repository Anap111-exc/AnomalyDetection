# -*- coding: utf-8 -*-
"""
规则引擎：RuleEngine 类 + 13条规则函数
规则优先于 ML，命中确定性违规直接打高分
"""

import pandas as pd
import numpy as np
from config import *

NS_PER_DAY = 86_400_000_000_000


def _time_ns(series):
    """时间列/数组转 int64 纳秒（统一UTC→naive→ns精度，兼容老版本numpy/pandas与us/ns精度差异）"""
    return pd.to_datetime(series, utc=True).dt.tz_convert(None).astype('datetime64[ns]').astype('int64').values


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
    if 'is_violated' not in df.columns:
        df['is_violated'] = 0
    df.loc[mask, score_col] = np.maximum(df.loc[mask, score_col], score_val)
    df.loc[mask, 'is_violated'] = np.maximum(
        df.loc[mask, 'is_violated'],
        100 if score_val >= 100 else 0
    )
    df.loc[mask, reason_col] = df.loc[mask, reason_col].fillna('') + ';' + reason
    return df


# ======================== 价格规则 ========================

def rule_p1_multiple_supplier_gap(df):
    """P1: 同商品名称同月多供应商价差超阈值 → 仅标记买贵的行=100分
    （低价笔数≥2才触发，防单笔离群值误报；高价笔须与低价笔在±N天内并存，区分"涨价"与"真供应商价差"）"""
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
    base = (_cnt >= 2) & (_min > 0) & (_min_cnt >= 2) & \
           (_max > _min * RULE_PRICE_GAP_RATIO) & \
           (df[COL_TAX_PRICE] == _max)

    # 时间重叠约束：高价行前后RULE_P1_OVERLAP_DAYS天内须存在低价行
    # （涨价场景低价笔集中在涨价前、高价笔在涨价后，时间不重叠则不标；真供应商并存则交错出现照标）
    if base.any() and COL_SUBMIT_TIME in df.columns:
        overlap = pd.Series(False, index=df.index)
        low = df[df[COL_TAX_PRICE] == _min]
        for sku, sub in df[base].groupby(COL_SKU_NAME):
            low_times = _time_ns(low.loc[low[COL_SKU_NAME] == sku, COL_SUBMIT_TIME])
            if len(low_times) == 0:
                continue
            sub_sorted = sub.sort_values(COL_SUBMIT_TIME)
            times = _time_ns(sub_sorted[COL_SUBMIT_TIME])
            for t, idx in zip(times, sub_sorted.index):
                has_low = ((low_times >= t - RULE_P1_OVERLAP_DAYS * NS_PER_DAY) &
                           (low_times <= t + RULE_P1_OVERLAP_DAYS * NS_PER_DAY)).any()
                overlap.loc[idx] = has_low
        mask = base & overlap
    else:
        mask = base
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


def rule_p4_break_prev_median(df):
    """P4: 突破前向常态基准价15% → 85分
    基准=该行之前前10笔价格中位数（近期常态，时间穿越），不足3笔回退全历史前向中位数
    段首过滤：同价连续段只保留段内前N笔（与Prophet段首过滤一致，价格水平调整后的后续同价单不重复标）"""
    if 'sku_prev_price_10' not in df.columns:
        return df
    has_history = df['sku_purchase_count'] >= 1
    seg_ok = df['sku_price_seg_rank'] <= PROPHET_JUMP_KEEP_N if 'sku_price_seg_rank' in df.columns else True
    mask = has_history & seg_ok & \
           (df['sku_prev_price_10'] > 0) & \
           (df[COL_TAX_PRICE] > df['sku_prev_price_10'] * RULE_PREV_MEDIAN_BREAK)
    return _apply_score(df, mask, 'price', 85, 'P4:突破前向常态基准价15%')


# ======================== 数量规则 ========================

def rule_q1_qty_exceeds_history(df):
    """Q1: 小样本(5≤样本<10)数量超全量中位数2.5倍 → 90分
    （基准用前向中位数而非历史峰值：峰值随大单跳变导致同量不同判；中位数稳健）"""
    if 'sku_qty_median' not in df.columns:
        return df
    has_history = df['sku_purchase_count'] >= 1
    mask = has_history & \
           (df['sku_sample_count'] < MIN_SKU_SAMPLES) & \
           (df['sku_sample_count'] >= 5) & \
           (df['sku_qty_median'] > 0) & \
           (df[COL_PUR_QTY] > df['sku_qty_median'] * RULE_QTY_EXCEED_RATIO)
    return _apply_score(df, mask, 'qty', 90, 'Q1:数量超全量中位数4倍')

def rule_q2_buyer_daily_total(df):
    """Q2已废弃"""
    return df


def rule_q3_qty_cross_check(df):
    """Q3: 数量纵横双超 → 90分（强信号已移除，业务确认只保留双超口径）
    纵向=超所属二级部门该SKU采购量中位数×3（本单位内）；横向=超该SKU全部订单数量中位数×10（全局）"""
    for c in ['sku_qty_median', 'dept_sku_qty_median']:
        if c not in df.columns:
            return df
    has_history = df['sku_purchase_count'] >= 1
    longi = (df[COL_PUR_QTY] > df['sku_qty_median'] * RULE_QTY_HIST_RATIO) & \
            (df['sku_qty_median'] > 0)
    later = (df[COL_PUR_QTY] > df['dept_sku_qty_median'] * RULE_QTY_DEPT_RATIO) & \
            (df['dept_sku_qty_median'] > 0)
    both = has_history & longi & later
    df = _apply_score(df, both, 'qty', 90, 'Q3:数量纵横双超(纵向同级3倍+横向全局10倍)')
    return df


# ======================== 拆单规则 ========================

def rule_s1_same_day_split(df):
    """S1: 同日拆单 → 100分（品类动态阈值）
    拆单定义：单笔都低于常规单笔金额、合计超过常规单笔金额（化整为零）"""
    if 'same_day_order_count' not in df.columns or 'same_day_merged_amount' not in df.columns:
        return df
    if 'split_threshold' not in df.columns:
        return df
    group_cond = (df['same_day_order_count'] >= 2) & \
                 (df['same_day_merged_amount'] > df['split_threshold'])
    # 单笔条件：组内每一笔的单笔金额都低于常规单笔金额（缺失特征时回退仅合计判定）
    if 'same_day_max_amount' in df.columns:
        group_cond = group_cond & (df['same_day_max_amount'] < df['split_threshold'])
    mask = group_cond
    df = _apply_score(df, mask, 'split', 100, 'S1:同日拆单')
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    df.loc[mask, 'split_type'] = df.loc[mask, 'split_type'].apply(
        lambda x: 'same_day' if pd.isna(x) or x == '' else x + ',same_day'
    )
    return df


def rule_s2_cross_day_split(df):
    """S2: 跨天拆单(7天滑窗) → 80分（分组加项目+地址+收货人）"""
    if 'same_day_order_count' not in df.columns:
        return df
    if 'split_threshold' not in df.columns:
        return df
    df_sorted = df.sort_values(COL_SUBMIT_TIME)
    split_flags = pd.Series(False, index=df_sorted.index)

    s2_key = [COL_SMTR_NAME, COL_SUP_NAME, COL_SKU_NAME,
              '_proj_group', '_address_normalized', COL_CONSIGNER]
    s2_key = [c for c in s2_key if c in df_sorted.columns]
    if len(s2_key) < 3:
        return df

    for grp_key, grp in df_sorted.groupby(s2_key, dropna=False):
        n = len(grp)
        if n < 2:
            continue

        grp_idx = grp.index
        grp_sorted = grp.sort_values(COL_SUBMIT_TIME)

        # 向量化：用 rolling('7D', closed='both') 求滑窗合计/笔数
        grp_rolled = grp_sorted.set_index(COL_SUBMIT_TIME)
        roll_sum = (
            grp_rolled[COL_SUB_TTL]
            .rolling('7D', closed='both')
            .sum()
            .values
        )
        roll_cnt = (
            grp_rolled[COL_SUB_TTL]
            .rolling('7D', closed='both')
            .count()
            .values
        )

        # 触发行：窗口内>=2笔 且 合计>阈值 且 当前笔<阈值
        is_trigger = (
            (roll_cnt >= 2) &
            (roll_sum > grp_sorted['split_threshold'].values) &
            (grp_sorted[COL_SUB_TTL].values < grp_sorted['split_threshold'].values)
        )

        # 对每个触发行，将窗口内所有行都标记（回传传播）
        grp_mask = np.zeros(n, dtype=bool)
        if is_trigger.any():
            times = _time_ns(grp_sorted[COL_SUBMIT_TIME])
            for idx in np.where(is_trigger)[0]:
                left = np.searchsorted(times, times[idx] - RULE_SPLIT_WINDOW_DAYS * NS_PER_DAY)
                grp_mask[left:idx+1] = True

        split_flags.loc[grp_idx] = grp_mask

    df_sorted = _apply_score(df_sorted, split_flags, 'split', 80, 'S2:跨天拆单')
    df['split_score'] = df_sorted['split_score']
    if 'split_rule_reason' in df_sorted.columns:
        df['split_rule_reason'] = df_sorted['split_rule_reason']
    if 'split_type' not in df.columns:
        df['split_type'] = ''
    cross_mask = split_flags.loc[df.index].fillna(False).values
    df.loc[cross_mask, 'split_type'] = df.loc[cross_mask, 'split_type'].apply(
        lambda x: 'cross_day' if pd.isna(x) or x == '' else x + ',cross_day'
    )
    return df


# ======================== 短期高频规则 ========================

def rule_s6_short_term_high_freq(df):
    """S6: 短期高频 → 7天内≥2笔=80分, 14天内≥3笔=70分（分组加地址+项目，不含收货人）"""
    if COL_SKU_NAME not in df.columns or COL_SMTR_NAME not in df.columns:
        return df
    if COL_SUBMIT_TIME not in df.columns:
        return df

    df_sorted = df.sort_values(COL_SUBMIT_TIME)

    flags_7d = pd.Series(False, index=df_sorted.index)
    flags_14d = pd.Series(False, index=df_sorted.index)

    # S6用_proj_group：占位符统一归组，有真实项目名的按项目区分
    s6_key = [COL_SMTR_NAME, COL_SKU_NAME,
              '_address_normalized', '_proj_group', COL_CONSIGNER]
    s6_key = [c for c in s6_key if c in df_sorted.columns]
    if len(s6_key) < 2:
        return df

    for grp_key, grp in df_sorted.groupby(s6_key, dropna=False):
        times = _time_ns(grp[COL_SUBMIT_TIME])
        n = len(grp)
        if n < 2:
            continue
        idx_list = list(grp.index)
        j_7 = 0
        j_14 = 0
        for i in range(n):
            ti = times[i]
            while j_7 < n and times[j_7] < ti - 7 * NS_PER_DAY:
                j_7 += 1
            while j_14 < n and times[j_14] < ti - 14 * NS_PER_DAY:
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
    flags_7d_idx = flags_7d.loc[df.index].fillna(False).values
    flags_14d_idx = flags_14d.loc[df.index].fillna(False).values
    df.loc[flags_7d_idx, 'split_type'] = df.loc[flags_7d_idx, 'split_type'].apply(
        lambda x: '高频7d' if pd.isna(x) or x == '' else x + ',高频7d'
    )
    df.loc[flags_14d_idx, 'split_type'] = df.loc[flags_14d_idx, 'split_type'].apply(
        lambda x: '高频14d' if pd.isna(x) or x == '' else x + ',高频14d'
    )
    return df


# ======================== 新增拆单规则（S3-S5：收货人/地址/项目维度） ========================

def rule_s3_consigner_split(df):
    """S3: 收货人拆单 → 80分（分组加项目+地址，标记整组）"""
    for c in ['consigner_day_order_count','consigner_day_amount','split_threshold']:
        if c not in df.columns:
            return df
    group_keys = ['_consigner_valid', 'sku_name', '_proj_valid', 'submit_day']
    group_keys = [c for c in group_keys if c in df.columns]
    if len(group_keys) < 3:
        return df
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


# ======================== 聚量规则 ========================

def rule_c2_high_price_concen(df):
    """C2: 高价+高浓度组合 → 95分（买过≥5种不同SKU才有统计意义，跳过首笔）"""
    if 'buyer_sup_amount_ratio_30d' not in df.columns:
        return df
    if 'buyer_sup_order_count_30d' not in df.columns:
        return df
    if 'buyer_sku_diversity' not in df.columns:
        return df
    has_history = df['sku_purchase_count'] >= 1
    # 价格偏离基准：优先用前10笔近期常态价（防全历史均值被旧价格主导造成压线误判），否则回退全历史偏离率
    if 'sku_prev_price_10' in df.columns:
        dev = (df[COL_TAX_PRICE] - df['sku_prev_price_10']) / df['sku_prev_price_10'].clip(lower=0.01)
    else:
        dev = df['price_deviation_rate']
    mask = has_history & \
           (df['sku_sample_count'] >= 5) & \
           (df['buyer_sup_order_count_30d'] >= 3) & \
           (df['buyer_sup_amount_ratio_30d'] > RULE_CONCEN_PRICE_RATIO) & \
           (dev > RULE_CONCEN_PRICE_DEVIATION) & \
           (df['buyer_sku_diversity'] >= 5)
    return _apply_score(df, mask, 'concen', 95, 'C2:高价供应商依赖')


def rule_c3_high_price_high_qty(df):
    """C3: 高价高量组合 → 100分（需至少3笔交易）"""
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
    return _apply_score(df, mask, 'concen', 100, 'C3:高价高量组合')


def rule_c4_price_jump_high_qty(df):
    """C4: 首次涨价+大量 → 100分（价格首次跳涨>20%且数量>历史均值10倍，需非首笔）"""
    has_history = df['sku_purchase_count'] >= 1
    price_jump = df[COL_TAX_PRICE] > df['sku_max_price'] * 1.2
    qty_surge = df[COL_PUR_QTY] > df['sku_avg_pur_qty'] * 10
    mask = has_history & \
           (df['sku_sample_count'] >= 5) & \
           price_jump & qty_surge & (df['sku_max_price'] > 0)
    return _apply_score(df, mask, 'concen', 100, 'C4:首次涨价+数量暴增')

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
    engine.register('price', rule_p4_break_prev_median)
    engine.register('qty', rule_q1_qty_exceeds_history)  # Q1: 小样本超全量中位数4倍(业务确认保留)
    engine.register('qty', rule_q2_buyer_daily_total)
    engine.register('qty', rule_q3_qty_cross_check)
    engine.register('split', rule_s1_same_day_split)
    engine.register('split', rule_s2_cross_day_split)
    engine.register('split', rule_s3_consigner_split)
    engine.register('split', rule_s6_short_term_high_freq)
    engine.register('concen', rule_c2_high_price_concen)
    engine.register('concen', rule_c3_high_price_high_qty)
    engine.register('concen', rule_c4_price_jump_high_qty)
    engine.register('ts', rule_t1_buyer_monthly_mutation)
    engine.register('ts', rule_t2_sku_price_mutation)
    engine.register('invis', rule_i1_sparse_data_flag)
