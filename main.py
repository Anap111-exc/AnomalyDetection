# -*- coding: utf-8 -*-
"""主入口：串联全部流程"""

import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import (
    price_detector, qty_detector, split_detector,
    concen_detector, invis_detector,
)
from graph.build_graph import build_dim_graph
from graph.graph_algo import run_louvain, run_pagerank
from graph.boost import compute_boost, map_boost_to_orders
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist
from audit import generate_audit_report
from drifts import check_drift


def main():
    print("=" * 50)
    print("内采异常检测系统 v2.0")
    print("=" * 50)

    # 0. 初始化规则引擎
    rule_engine = RuleEngine()
    register_all_rules(rule_engine)
    print(f"  规则引擎: 已注册 {sum(len(v) for v in rule_engine.rules.values())} 条规则")

    # 1. 读取数据
    print("\n[1/10] 读取数据...")
    #导入数据，INPUT_FILE替换为数据所在文件夹地址
    input_path = INPUT_FILE

    df = pd.read_excel(input_path)
    total_rows = len(df)
    print(f"  读取 {total_rows} 行, {len(df.columns)} 列")
    print(f"  列名: {list(df.columns)[:10]}...")

    # 2. 数据清洗
    print("\n[2/10] 数据清洗...")
    df = data_process(df)

    # 3. 特征工程
    print("\n[3/10] 特征工程...")
    df = feature_engineer(df)

    # 4. 白名单
    print("\n[4/10] 白名单...")
    whitelist, wl_set = load_whitelist()
    df = apply_whitelist(df, whitelist, wl_set)
    print(f"  白名单实体: {len(wl_set)} 个")

    # 5. 6个检测器
    print("\n[5/10] 运行检测器...")
    df = price_detector(df, rule_engine)
    df = qty_detector(df, rule_engine)
    df = split_detector(df, rule_engine)
    df = concen_detector(df, rule_engine)
    df = invis_detector(df, rule_engine)

    # 6. 图算法
    print("\n[6/10] 图算法提权...")
    graph_dims = ['price', 'qty', 'concen', 'invis']
    for dim in graph_dims:
        try:
            G = build_dim_graph(df, dim, whitelist=wl_set)
            if G.number_of_nodes() == 0:
                print(f"    [{dim}] 图为空，跳过")
                continue

            partition, communities = run_louvain(G)
            pr = run_pagerank(G)

            # 收集节点原始分
            node_scores = {}
            score_col = f'{dim}_score'
            if score_col in df.columns:
                for _, row in df.iterrows():
                    for entity in [str(row.get(COL_SMTR_NAME, '')),
                                   str(row.get(COL_SUP_NAME, '')),
                                   str(row.get(COL_SKU_NAME, ''))]:
                        if entity and entity in node_scores:
                            node_scores[entity] = max(node_scores[entity], row[score_col])
                        elif entity:
                            node_scores[entity] = row[score_col]

            node_boost = compute_boost(partition, pr, communities, node_scores,
                                       whitelist=wl_set)
            df = map_boost_to_orders(df, node_boost, dim)
            print(f"    [{dim}] 节点={G.number_of_nodes()}, 边={G.number_of_edges()}, 社区={len(communities)}")
        except Exception as e:
            print(f"    [{dim}] 图计算失败: {e}")
            df[f'graph_{dim}_boost'] = 0

    # 7. 融合
    print("\n[7/10] 加权融合...")
    df = compute_final_score(df)

    # 8. 漂移检查
    print("\n[8/10] 漂移检查...")
    check_drift(df)

    # 9. 审计报告
    print("\n[9/10] 审计报告...")
    generate_audit_report(df)

    # 10. 输出
    print("\n[10/10] 保存结果...")
    output_cols = [
        COL_ORDER_ID, COL_ORD_ITEM_ID, COL_SUBMIT_TIME,
        COL_SMTR_NAME, COL_DEPT, COL_SUP_NAME, COL_SKU_NAME,
        COL_TAX_PRICE, COL_PUR_QTY, COL_SUB_TTL,
        'price_score', 'qty_score', 'split_score', 'concen_score',
        'ts_score', 'invis_score',
    ]
    # 图boost列
    output_cols += [f'graph_{dim}_boost' for dim in graph_dims]
    output_cols += ['invis_shap_top5',
        'anomaly_type', 'anomaly_detail', 'risk_level',
        'is_whitelisted', 'whitelist_reason',
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_excel(OUTPUT_FILE, index=False)

    print(f"  [OK] {total_rows}行 -> {len(df)}行 -> {OUTPUT_FILE}")
    anomaly_cnt = (df['risk_level'] == '异常').sum()
    normal_cnt = (df['risk_level'] == '无异常').sum()
    print(f"  异常: {anomaly_cnt} ({anomaly_cnt/len(df)*100:.1f}%)")
    print(f"  无异常: {normal_cnt} ({normal_cnt/len(df)*100:.1f}%)")


if __name__ == '__main__':
    main()
