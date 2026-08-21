# -*- coding: utf-8 -*-
"""价格异常检测器：规则(P1-P4) + per-SKU KDE核密度估计"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from config import *


def price_detector(df, rule_engine):
    print("  [price_detector] 执行中...")

    _NS_DAY = 86_400_000_000_000

    if 'price_score' not in df.columns:
        df['price_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='price')

    # Step 2: per-SKU KDE
    rule_hit = df['price_score'] >= 80
    large_sample = df['sku_sample_count'] >= MIN_SKU_SAMPLES
    df_ml = df[large_sample & ~rule_hit].copy()

    if len(df_ml) == 0:
        print(f"    KDE跳过")
        return df

    df_list = []
    for grp_key, grp in df_ml.groupby(COL_SKU_NAME):
        if len(grp) < 5:
            df_list.append(grp)
            continue

        prices = grp[COL_TAX_PRICE].values.reshape(-1, 1)
        try:
            from scipy import stats
            # 低波动SKU跳过：价格基本无波动时KDE排名是随机噪声（cv<0.03直接跳过）
            mean_price = float(np.mean(prices))
            cv = float(np.std(prices, ddof=0)) / mean_price if mean_price > 0 else 0
            if cv < PRICE_KDE_MIN_CV:
                scores = np.zeros(len(grp))
            else:
                kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(prices)
                # 分位数法：拟合分布 → 求 CDF 的 RULE_PRICE_KDE_PERCENTILE 分位价格作为阈值（与数量KDE一致）
                p_max = float(np.max(prices))
                grid = np.linspace(0, max(p_max * 1.5, 10), 2000).reshape(-1, 1)
                log_dens = kde.score_samples(grid)
                dens = np.exp(log_dens)
                cdf = np.cumsum(dens)
                cdf = cdf / cdf[-1]
                idx_p = np.searchsorted(cdf, RULE_PRICE_KDE_PERCENTILE)
                threshold_kde = float(grid[idx_p][0])
                # 窗口兜底：阈值不低于"该行±30天窗口内价格中位×1.05"（按天粒度减少边界敏感，不足3行回退全史中位）
                try:
                    grp_sorted = grp.sort_values(COL_SUBMIT_TIME)
                    # 时间转 int64 纳秒（统一ns精度）并按天floor，同一天的行窗口完全一致
                    times = pd.to_datetime(grp_sorted[COL_SUBMIT_TIME], utc=True).dt.tz_convert(None).astype('datetime64[ns]').astype('int64').values
                    times_day = times - times % _NS_DAY
                    p_sorted = grp_sorted[COL_TAX_PRICE].values
                    hist_med = float(np.median(prices))
                    thresholds = np.empty(len(grp_sorted))
                    for i, t in enumerate(times_day):
                        lo = np.searchsorted(times_day, t - 30 * _NS_DAY)
                        hi = np.searchsorted(times_day, t + 30 * _NS_DAY, side='right')
                        win = p_sorted[lo:hi]
                        win_med = float(np.median(win)) if len(win) >= 3 else hist_med
                        thresholds[i] = max(threshold_kde, win_med * PRICE_KDE_MIN_RATIO)
                    threshold = pd.Series(thresholds, index=grp_sorted.index).reindex(grp.index).values
                except Exception:
                    threshold = np.full(len(grp), threshold_kde)
                # 价格超过阈值才给分，分数按超阈值比例映射（最高100分）
                raw_scores = np.where(prices.flatten() > threshold, prices.flatten() / threshold * 100, 0)
                scores = np.minimum(raw_scores, 100).round(1)
        except Exception:
            scores = np.zeros(len(grp))

        grp['price_score_ml'] = scores
        df_list.append(grp)

    df_ml = pd.concat(df_list)
    if 'price_score_ml' in df_ml.columns:
        df.loc[df_ml.index, 'price_score_ml'] = df_ml['price_score_ml']
    if 'price_score_ml' not in df.columns:
        df['price_score_ml'] = 0.0
    df['price_score_ml'] = df['price_score_ml'].fillna(0)
    df['price_score'] = np.maximum(df['price_score'], df['price_score_ml'].fillna(0))
    # 标注来源，让结果显示时能区分 KDE 与规则/Prophet
    kde_mask = df['price_score_ml'] > 0
    if 'price_rule_reason' not in df.columns:
        df['price_rule_reason'] = ''
    df.loc[kde_mask, 'price_rule_reason'] = \
        df.loc[kde_mask, 'price_rule_reason'].fillna('') + ';KDE:价格分布离群'

    print(f"    KDE处理 {len(df_ml)} 行, 规则命中 {(rule_hit).sum()} 行")
    return df
