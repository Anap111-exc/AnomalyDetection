# -*- coding: utf-8 -*-
"""隐形异常检测器：规则(I1标记) + 全局IF+LOF + SHAP"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from ml_utils import normalize_anomaly_scores
from config import *

# SHAP 可选
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# 排除列（非数值或ID/分组key）
INVIS_EXCLUDE = {
    COL_ORDER_ID, COL_ORDER_NO, COL_ORD_ITEM_ID, COL_SKU_ID, COL_SKU_CODE, COL_SKU_NAME,
    COL_SMTR_NAME, COL_SUP_NAME, COL_DEPT, COL_SEC_DEPT, COL_COMPANY, COL_STORE_NAME,
    COL_MEASURE_UNIT, COL_ONE_CAT, COL_TWO_CAT, COL_THREE_CAT, COL_FOUR_CAT,
    COL_MATERIAL_TYPE, COL_MATERIAL_ZONE, COL_STORE_PZN_LEVEL,
    COL_IS_POVT, COL_ORD_TYPE, COL_ORD_STATUS, COL_SEC_DEPT_TYPE,
    'submit_day', 'submit_month', 'submit_week',
    'price_score', 'qty_score', 'split_score', 'concen_score', 'ts_score', 'invis_score',
    'price_score_ml', 'qty_score_ml', 'concen_score_ml', 'ts_score_ml',
    'price_direction', 'split_type', 'amount_deviation_flag',
    'is_small_sample', 'is_violated', 'is_whitelisted', 'whitelist_reason',
    'rule_reason', 'price_rule_reason', 'qty_rule_reason', 'split_rule_reason',
    'concen_rule_reason', 'invis_rule_reason', 'invis_attention',
}

PRICE_FEATURES = {'tax_price', 'price_deviation_rate', 'price_zscore',
                   'price_vs_max', 'price_vs_median', 'price_vs_category'}
QTY_FEATURES = {'pur_qty', 'qty_deviation_rate', 'qty_zscore'}


def invis_detector(df, rule_engine):
    print("  [invis_detector] 执行中...")

    if 'invis_score' not in df.columns:
        df['invis_score'] = 0.0

    # Step 1: 规则引擎（只有标记规则I1）
    df = rule_engine.execute(df, dim='invis')

    # Step 2: 构建全特征空间
    all_num = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in all_num if c not in INVIS_EXCLUDE and not c.startswith('_')]
    feature_cols = feature_cols[:50]  # 限制特征数

    if len(feature_cols) < 3:
        print("    特征不足，跳过ML")
        return df

    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: 全局 IF
    try:
        iso = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION_INVIS,
            random_state=IF_RANDOM_STATE
        )
        y_pred_if = iso.fit_predict(X_scaled)
        raw_if = -iso.score_samples(X_scaled)
        is_anom = (y_pred_if == -1)
        if_score = np.zeros(len(df))
        if is_anom.sum() > 0:
            if_score[is_anom] = normalize_anomaly_scores(raw_if[is_anom], 40, 100)
        if_score[~is_anom] = 0
    except Exception:
        if_score = np.zeros(len(df))

    # Step 4: 全局 LOF (novelty=False → fit_predict)
    try:
        n_neighbors = min(LOF_N_NEIGHBORS, len(df) - 1, max(2, int(len(df) * (1 - IF_CONTAMINATION_INVIS)) - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                                 contamination=IF_CONTAMINATION_INVIS)
        y_pred_lof = lof.fit_predict(X_scaled)
        raw_lof = -lof.negative_outlier_factor_
        is_anom_lof = (y_pred_lof == -1)
        lof_score = np.zeros(len(df))
        if is_anom_lof.sum() > 0:
            lof_score[is_anom_lof] = normalize_anomaly_scores(raw_lof[is_anom_lof], 40, 100)
        lof_score[~is_anom_lof] = 0
    except Exception:
        lof_score = np.zeros(len(df))

    df['invis_score_ml'] = ((if_score + lof_score) / 2).round(1)
    df['invis_score'] = np.maximum(df['invis_score'], df['invis_score_ml'].fillna(0))
    # 标注来源，让结果显示时能区分 ML 与规则
    ml_mask = df['invis_score_ml'] > 0
    if 'invis_rule_reason' not in df.columns:
        df['invis_rule_reason'] = ''
    df.loc[ml_mask, 'invis_rule_reason'] = \
        df.loc[ml_mask, 'invis_rule_reason'].fillna('') + ';IF+LOF:特征空间离群'

    # Step 5: SHAP 解释（仅高分行）
    df['invis_shap_top5'] = ''
    df['invis_overlap_price'] = False
    df['invis_overlap_qty'] = False

    high_mask = df['invis_score_ml'] > 60
    if HAS_SHAP and high_mask.sum() > 0:
        try:
            explainer = shap.TreeExplainer(iso)
            shap_vals = explainer.shap_values(X_scaled[high_mask])
            for i, idx in enumerate(df.index[high_mask]):
                top5 = np.argsort(np.abs(shap_vals[i]))[-5:][::-1]
                top5_names = [feature_cols[j] for j in top5]
                df.at[idx, 'invis_shap_top5'] = ','.join(top5_names)

                top5_set = set(top5_names)
                df.at[idx, 'invis_overlap_price'] = len(top5_set & PRICE_FEATURES) >= 3
                df.at[idx, 'invis_overlap_qty'] = len(top5_set & QTY_FEATURES) >= 3
        except Exception:
            pass

    print(f"    IF+LOF完成, SHAP解释 {high_mask.sum()} 行")
    return df
