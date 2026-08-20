# -*- coding: utf-8 -*-
"""审计闭环：Top-N 输出 + 统计摘要"""

import os
import pandas as pd
from config import *


def generate_audit_report(df, top_n=AUDIT_TOP_N):
    """输出Top-N高风险订单"""
    now_str = pd.Timestamp.now().strftime('%Y%m%d')

    # Top-N
    audit_cols = [
        COL_ORDER_ID, COL_ORD_ITEM_ID, COL_SMTR_NAME, COL_SUP_NAME,
        COL_SKU_NAME, COL_TAX_PRICE, COL_PUR_QTY, COL_SUB_TTL,
        'anomaly_type', 'anomaly_detail', 'risk_level',
        'whitelist_reason',
    ]
    audit_cols = [c for c in audit_cols if c in df.columns]
    # Top-N: 按异常行的 boosted max 排序
    if 'final_score' in df.columns:
        audit_df = df.nlargest(top_n, 'final_score')[audit_cols]
    else:
        boosted = [f'boosted_{d}' for d in ['price','qty','split','concen','ts','invis']]
        boosted = [c for c in boosted if c in df.columns]
        if boosted:
            df['_audit_sort'] = df[boosted].max(axis=1)
            audit_df = df.nlargest(top_n, '_audit_sort')[audit_cols]
            df.drop(columns=['_audit_sort'], inplace=True)
        else:
            audit_df = df[df['risk_level'] == '异常'].head(top_n)[audit_cols]
    import time
    ts = int(time.time())
    audit_path = os.path.join(AUDIT_DIR, f'audit_top{top_n}_{ts}.xlsx')
    audit_df.to_excel(audit_path, index=False)
    print(f"  审计报告: {audit_path}")

    # 统计摘要
    total_orders = df[COL_ORDER_ID].nunique() if COL_ORDER_ID in df.columns else len(df)
    print(f"\n  ===== 异常检测摘要 =====")
    print(f"  总订单数: {total_orders}")
    anomaly_cnt = (df['risk_level'] == '异常').sum()
    normal_cnt = (df['risk_level'] == '无异常').sum()
    print(f"  异常: {anomaly_cnt} ({anomaly_cnt/len(df)*100:.1f}%)")
    print(f"  无异常: {normal_cnt} ({normal_cnt/len(df)*100:.1f}%)")
    print(f"  异常类型分布:")
    for t, c in df['anomaly_type'].value_counts().head(6).items():
        print(f"    {t}: {c}")
