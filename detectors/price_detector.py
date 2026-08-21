# -*- coding: utf-8 -*-
"""价格异常检测器：规则(P1-P4) + per-SKU KDE核密度估计"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from config import *


def price_detector(df, rule_engine):
    print("  [price_detector] 执行中...")

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
                log_density = kde.score_samples(prices)
                ranks = stats.rankdata(-log_density)
                pct = ranks / len(ranks) * 100
                scores = np.where(pct >= 95, pct, 0)
                # 业务底线：排除"不高反低"的左侧离群点，高于基准价1.05倍才允许给分
                # 基准价 = max(该行前后30天价格中位数, 全历史中位数)
                # （全历史下限防止价格水平切换后旧价残留单被未来低价基准误判；同期窗口不足3行回退全历史中位数）
                try:
                    grp_sorted = grp.sort_values(COL_SUBMIT_TIME)
                    # 时间转 int64 纳秒（统一ns精度，规避老版本numpy/pandas的datetime搜索兼容与us/ns差异）
                    times = pd.to_datetime(grp_sorted[COL_SUBMIT_TIME], utc=True).dt.tz_convert(None).astype('datetime64[ns]').astype('int64').values
                    _NS_DAY = 86_400_000_000_000
                    p_sorted = grp_sorted[COL_TAX_PRICE].values
                    hist_med = float(np.median(prices))
                    bases = np.empty(len(grp_sorted))
                    for i, t in enumerate(times):
                        lo = np.searchsorted(times, t - 30 * _NS_DAY)
                        hi = np.searchsorted(times, t + 30 * _NS_DAY, side='right')
                        win = p_sorted[lo:hi]
                        bases[i] = max(np.median(win), hist_med) if len(win) >= 3 else hist_med
                    median_price = pd.Series(bases, index=grp_sorted.index).reindex(grp.index).values
                except Exception:
                    median_price = np.full(len(grp), np.median(prices))
                trivial_mask = grp[COL_TAX_PRICE].values <= median_price * PRICE_KDE_MIN_RATIO
                scores[trivial_mask] = 0
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
