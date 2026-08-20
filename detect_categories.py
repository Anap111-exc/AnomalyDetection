# -*- coding: utf-8 -*-
"""
内采异常采购检测 - 品类批量检测入口（ti-one 平台版）

功能：
  1. 检测单一品类：python detect_categories.py 茶叶
  2. 检测多个品类：python detect_categories.py "标签机,茶叶,牛奶（不含鲜奶）"
     （品类名用英文逗号分隔，自动逐个检测并分别输出）

目录约定：
  代码目录  : /AnomalyDetection/
  原始数据  : /ad_dataload/            （自动定位第一个 .xlsx）
  结果输出  : /dataresult/品类专项/     （文件名为 <品类名>-内采异常数据分析.xlsx）

品类匹配：
  优先精确匹配 four_cat_name == 品类名；若 0 行则回退"包含"匹配（如 标签机 匹配 标签机/标牌机）。
"""
import pandas as pd, numpy as np, os, sys, time, glob, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.environ.get("ANOMALY_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import price_detector, qty_detector, split_detector, concen_detector
from graph.build_graph import build_dim_graph
from graph.graph_algo import run_louvain, run_pagerank
from graph.boost import compute_boost, map_boost_to_orders
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist


def locate_data_file():
    """定位数据目录下的第一个数据文件（.xlsx 或 .csv）"""
    for ext in ('*.xlsx', '*.csv'):
        files = sorted(glob.glob(os.path.join(DATA_DIR, ext)))
        if files:
            return files[0]
    raise FileNotFoundError(f"未在 {DATA_DIR} 找到数据文件(.xlsx/.csv)")


def read_data(path):
    """读取数据文件（按扩展名选择解析方式）"""
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)


# 品类名 → 实际类目关键词映射（品类名可能是文件夹名/习惯叫法，与 four_cat 不完全一致）
# 未列入映射的品类名，会自动按"精确 → 包含"匹配
CATEGORY_KEYWORDS = {
    '显示器支架': '支架',
    '柜式空调-挂式空调': '家用空调',
    '柜式空调': '家用空调',
    '挂式空调': '家用空调',
    '排插插排': '排插',
    '标签机': '标签机',
    '牛奶（不含鲜奶）': '牛奶（不含鲜奶）',
    '打印机及复印机配件': '打印机及复印机配件',
    '茶叶': '茶叶',
    '移动硬盘': '移动硬盘',
    '油色谱在线监测装置维护': '油色谱在线监测装置维护',
}


def match_category(df, cat_name):
    """品类匹配：映射表关键词 → 精确 → 包含"""
    kw = CATEGORY_KEYWORDS.get(cat_name, cat_name)
    # 优先精确匹配 four_cat
    exact = df[df[COL_FOUR_CAT] == kw]
    if len(exact) > 0:
        return exact
    # 回退包含匹配
    return df[df[COL_FOUR_CAT].astype(str).str.contains(kw, na=False)]


def run_single_category(df_all, cat_name, engine, wl, wl_set):
    """对单个品类跑全流程检测，返回结果 DataFrame"""
    cat = match_category(df_all, cat_name)
    print(f"  品类[{cat_name}]: {len(cat)} 行, {cat['sku_name'].nunique()} SKU", flush=True)
    if len(cat) == 0:
        print(f"    !! 未匹配到任何订单，跳过")
        return None

    cat = apply_whitelist(cat, wl, wl_set)
    cat = engine.execute(cat)
    cat = price_detector(cat, engine)
    cat = qty_detector(cat, engine)
    cat = split_detector(cat, engine)
    cat = concen_detector(cat, engine)

    for dim in ['price', 'qty', 'concen']:
        try:
            G = build_dim_graph(cat, dim, whitelist=wl_set)
            if G.number_of_nodes() == 0:
                continue
            partition, communities = run_louvain(G)
            pr = run_pagerank(G)
            node_scores = {}
            score_col = f'{dim}_score'
            for _, row in cat.iterrows():
                for entity in [str(row.get(COL_SMTR_NAME, '')), str(row.get(COL_SUP_NAME, '')), str(row.get(COL_SKU_NAME, ''))]:
                    if entity:
                        node_scores[entity] = max(node_scores.get(entity, 0), row[score_col])
            node_boost = compute_boost(partition, pr, communities, node_scores, whitelist=wl_set)
            cat = map_boost_to_orders(cat, node_boost, dim)
        except Exception as e:
            print(f"    [{dim}] 图失败: {e}")
            cat[f'graph_{dim}_boost'] = 0

    cat = compute_final_score(cat)
    return cat


def export_result(cat, cat_name):
    """输出到 /dataresult/品类专项/<品类名>-内采异常数据分析.xlsx"""
    os.makedirs(RESULT_ROOT, exist_ok=True)
    out_path = os.path.join(RESULT_ROOT, f"{cat_name}-内采异常数据分析.xlsx")
    cols = [c for c in ['order_id', 'ord_item_id', 'submit_time',
            'smtr_name', 'thd_dept_short_name', 'sec_dept_short_name', 'sup_short_name',
            'consigner_name', 'rec_address', 'sku_name', 'proj_name', 'four_cat_name',
            'tax_price', 'pur_qty', 'sub_ttl',
            'anomaly_type', 'risk_level',
            'price_type', 'qty_type', 'split_type', 'concen_type'] if c in cat.columns]
    cat[cols].to_excel(out_path, index=False)
    n_anom = int((cat['risk_level'] == '异常').sum())
    print(f"  输出: {out_path} | 共{len(cat)}行, 异常{n_anom}行 ({n_anom/len(cat)*100:.1f}%)", flush=True)
    return out_path


def main():
    if len(sys.argv) < 2:
        # 交互模式：直接输入品类名（逗号分隔）
        print("请输入待检测品类名（多个用英文逗号分隔，如：标签机,茶叶,排插插排）：")
        inp = input().strip()
        cat_names = [c.strip() for c in inp.split(',') if c.strip()]
        if not cat_names:
            print("未输入有效品类，退出。")
            sys.exit(1)
    else:
        cat_names = [c.strip() for c in sys.argv[1].split(',') if c.strip()]
    print(f"待检测品类({len(cat_names)}个): {cat_names}")

    # 读取原始数据 + 全量特征工程（只做一次）
    raw_path = locate_data_file()
    print(f"读取数据: {raw_path}")
    df_all = data_process(read_data(raw_path))
    df_all = feature_engineer(df_all)

    wl, wl_set = load_whitelist()
    engine = RuleEngine(); register_all_rules(engine)

    for cat_name in cat_names:
        print(f"\n{'='*50}\n检测品类: {cat_name}\n{'='*50}", flush=True)
        t0 = time.time()
        try:
            cat = run_single_category(df_all, cat_name, engine, wl, wl_set)
            if cat is not None:
                export_result(cat, cat_name)
            print(f"  耗时 {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  品类[{cat_name}] 检测失败: {e}")

    print("\n全部完成!")


if __name__ == "__main__":
    main()