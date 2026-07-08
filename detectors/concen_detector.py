# -*- coding: utf-8 -*-
"""高价聚量检测器：规则(C1-C2) + 全局 IsolationForest"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ml_utils import normalize_anomaly_scores
from config import *


CONCEN_FEATURES = [
    'buyer_daily_order_count',
    'buyer_sup_amount_ratio_30d',
    'buyer_supplier_diversity',
    'buyer_sup_order_count_30d',
    'buyer_avg_order_amount',
    'price_deviation_rate',
    'price_zscore',
    'sup_price_premium',
]


def concen_detector(df, rule_engine):
    print("  [concen_detector] 执行中...")

    if 'concen_score' not in df.columns:
        df['concen_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='concen')

    # Step 2: 全局 IF
    features = [c for c in CONCEN_FEATURES if c in df.columns]

    if len(features) < 2:
        print("    ML跳过(特征不足)")
        return df

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        iso = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE
        )
        y_pred = iso.fit_predict(X_scaled)
        raw = -iso.score_samples(X_scaled)
        is_anom = (y_pred == -1)
        scores = np.zeros(len(df))
        if is_anom.sum() > 0:
            scores[is_anom] = normalize_anomaly_scores(raw[is_anom], 40, 100)
        scores[~is_anom] = 0
        df['concen_score_ml'] = scores
    except Exception:
        df['concen_score_ml'] = 50

    df['concen_score_ml'] = df['concen_score_ml'].fillna(50)
    # 交易笔数不足3笔的，浓度特征不可靠（ratio天然0或1），ML分置0
    low_tx_mask = df['buyer_sup_order_count_30d'].fillna(0) < 3
    df.loc[low_tx_mask, 'concen_score_ml'] = 0
    df['concen_score'] = np.maximum(df['concen_score'], df['concen_score_ml'])

    print(f"    规则命中 {(df['concen_score'] >= 70).sum()} 行")
    return df
