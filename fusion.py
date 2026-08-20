# -*- coding: utf-8 -*-
"""简化融合：规则优先 → ML+图兜底"""

import numpy as np
import pandas as pd
from config import *

DIM_CN = {
    'price': '价格异常',
    'qty': '数量异常',
    'split': '拆单',
    'concen': '高价聚量',
    'ts': '时序异常',
    'invis': '隐形异常',
}


def _dim_order(d):
    """主类型并列时优先级：高价聚量(双重信号)最高，其余按维度顺序"""
    return 5 if d == 'concen' else DIMS.index(d) if d in DIMS else 0


PRICE_REASON_CN = {
    'P1': '多供应商价差',
    'P2': '小样本偏离',
    'P3': '突破历史最高价',
    'P4': '突破前向常态基准价',
    'Prophet': '时序价格异常',
    'KDE': '价格分布离群',
}


def _extract_price_reason(reason):
    """从price_rule_reason中提取触发规则/算法名（含中文名）"""
    if pd.isna(reason) or reason == '':
        return ''
    reasons = []
    for r in str(reason).split(';'):
        r = r.strip()
        if 'P1:' in r: reasons.append('P1')
        elif 'P2:' in r: reasons.append('P2')
        elif 'P3:' in r: reasons.append('P3')
        elif 'P4:' in r: reasons.append('P4')
        elif 'Prophet:' in r: reasons.append('Prophet')
        elif 'KDE:' in r: reasons.append('KDE')
    tags = [f'{k}:{PRICE_REASON_CN[k]}' for k in dict.fromkeys(reasons) if k in PRICE_REASON_CN]
    return '[' + '+'.join(tags) + ']' if tags else ''


def _tag(reasons, cn_map):
    """按顺序去重，生成 '缩写:中文名+...' 标注"""
    tags = []
    for k in dict.fromkeys(reasons):
        if k in cn_map:
            tags.append(f'{k}:{cn_map[k]}')
    return '[' + '+'.join(tags) + ']' if tags else ''


QTY_REASON_CN = {
    'Q1': '超常态4倍',
    'Q3': '纵横双超',
    'KDE': '95%分布离群',
}


def _extract_qty_reason(reason):
    """从qty_rule_reason中提取触发规则/算法名（含中文名）"""
    if pd.isna(reason) or reason == '':
        return ''
    reasons = []
    for r in str(reason).split(';'):
        r = r.strip()
        if 'Q1:' in r: reasons.append('Q1')
        elif 'Q3:' in r: reasons.append('Q3')
        elif 'KDE:' in r: reasons.append('KDE')
    return _tag(reasons, QTY_REASON_CN)


SPLIT_REASON_CN = {
    'S1': '同日拆单',
    'S2': '跨天拆单',
    'S3': '收货人拆单',
    'S6': '短期高频',
}


def _extract_split_reason(reason):
    """从split_rule_reason中提取触发规则号(S1/S2/S3/S6)，含中文名"""
    if pd.isna(reason) or reason == '':
        return ''
    reasons = []
    for r in str(reason).split(';'):
        r = r.strip()
        if r.startswith('S1:'): reasons.append('S1')
        elif r.startswith('S2:'): reasons.append('S2')
        elif r.startswith('S3:'): reasons.append('S3')
        elif r.startswith('S6'): reasons.append('S6')
    return _tag(reasons, SPLIT_REASON_CN)


CONCEN_REASON_CN = {
    'C2': '高价供应商依赖',
    'C3': '高价高量组合',
    'C4': '首次涨价+数量暴增',
    'Prophet': '时序价格异常',
}


def _extract_concen_reason(reason):
    """从concen_rule_reason中提取触发规则/算法名（含中文名）"""
    if pd.isna(reason) or reason == '':
        return ''
    reasons = []
    for r in str(reason).split(';'):
        r = r.strip()
        if 'C2:' in r: reasons.append('C2')
        elif 'C3:' in r: reasons.append('C3')
        elif 'C4:' in r: reasons.append('C4')
        elif 'Prophet:' in r: reasons.append('Prophet')
    return _tag(reasons, CONCEN_REASON_CN)


def _extract_invis_reason(reason):
    """从invis_rule_reason中提取触发模型名（含中文名）"""
    if pd.isna(reason) or reason == '':
        return ''
    reasons = []
    for r in str(reason).split(';'):
        r = r.strip()
        if r.startswith('IF'): reasons.append('IF+LOF')
    return _tag(reasons, {'IF+LOF': '特征空间离群'})


def _dim_label(d, row):
    """维度类型标签：类型名 + 触发规则/模型标注"""
    if d == 'price':
        return '价格异常' + _extract_price_reason(row.get('price_rule_reason', ''))
    if d == 'qty':
        return '数量异常' + _extract_qty_reason(row.get('qty_rule_reason', ''))
    if d == 'split':
        # 规则号 + 拆单方式，如 拆单[S6](高频7d,same_day)
        label = '拆单' + _extract_split_reason(row.get('split_rule_reason', ''))
        mode = row.get('split_mode', '')
        if isinstance(mode, str) and mode:
            uniq = ','.join(dict.fromkeys(x for x in mode.split(',') if x))
            label += f'({uniq})'
        return label
    if d == 'concen':
        reason = str(row.get('concen_rule_reason', ''))
        # C2规则语义为"高价+供应商依赖"，与"高价聚量"（时序突变+数量聚集）区分展示
        if 'C2:' in reason:
            return '高价供应商依赖' + _extract_concen_reason(row.get('concen_rule_reason', ''))
        return '高价聚量' + _extract_concen_reason(row.get('concen_rule_reason', ''))
    if d == 'invis':
        return '隐形异常' + _extract_invis_reason(row.get('invis_rule_reason', ''))
    return DIM_CN.get(d, d)


def _qty_type_name(reason):
    """数量异常主类型：anomaly_type 保持纯类型名，细分形式（Q3纵横双超/Q1超常态/KDE分布离群）在 qty_type 列展示"""
    return '数量异常'


def compute_final_score(df):
    """规则引擎命中→直接异常；未命中→ML+图任一维度超阈值→异常
    输出: risk_level, anomaly_type(主类型), 各维度类型列 {dim}_type"""

    # Step 1: 分维度 boosted_score (上限 100)
    for dim in DIMS:
        boost_col = f'graph_{dim}_boost'
        score_col = f'{dim}_score'
        if boost_col in df.columns:
            df[f'boosted_{dim}'] = (df[score_col].fillna(0) + df[boost_col].fillna(0)).clip(upper=100)
        else:
            df[f'boosted_{dim}'] = df[score_col].fillna(0).clip(upper=100)

    # Step 2: 判定
    df['is_violated'] = df.get('is_violated', pd.Series(0, index=df.index)).fillna(0)
    df['is_whitelisted'] = df.get('is_whitelisted', pd.Series(0, index=df.index)).fillna(0)

    # 拆单方式备份（split_type 列随后被覆盖为拆单判定标签）
    if 'split_type' in df.columns:
        df['split_mode'] = df['split_type']

    # 分维度类型列初始化（score>0 即记录嫌疑及来源）
    for d in DIMS:
        df[f'{d}_type'] = ''

    risk_levels = []
    anomaly_types = []

    for idx, row in df.iterrows():
        if row['is_whitelisted'] == 1:
            risk_levels.append('无异常')
            anomaly_types.append('')
            continue

        # 各维度类型列（含来源标注）
        for d in DIMS:
            if row.get(f'{d}_score', 0) > 0:
                df.at[idx, f'{d}_type'] = _dim_label(d, row)

        if row['is_violated'] == 100:
            risk_levels.append('异常')
            hit = [d for d in DIMS if row.get(f'{d}_score', 0) >= 80]
            # 主类型 = 分数最高维度（并列时高价聚量优先），anomaly_type 只写维度类型
            hit.sort(key=lambda d: (row.get(f'{d}_score', 0), _dim_order(d)), reverse=True)
            atype = DIM_CN[hit[0]] if hit else '规则命中'
            # 数量异常按触发来源细分主类型（Q3横纵双超/Q1超常态/KDE分布离群）
            if hit and hit[0] == 'qty':
                atype = _qty_type_name(row.get('qty_rule_reason', ''))
            # C2规则主类型显示"高价供应商依赖"（非"高价聚量"）
            if hit and hit[0] == 'concen' and 'C2:' in str(row.get('concen_rule_reason', '')):
                atype = '高价供应商依赖'
            anomaly_types.append(atype)
            continue

        # ML+图判定（按维度使用不同阈值）
        max_dim = DIMS[0]
        max_val = 0
        for d in DIMS:
            v = row.get(f'boosted_{d}', 0)
            threshold = SPLIT_BINARY_THRESHOLD if d == 'split' else BINARY_THRESHOLD
            better = v > max_val or (v == max_val and _dim_order(d) > _dim_order(max_dim))
            if v >= threshold and better:
                max_val = v
                max_dim = d

        if max_val > 0:
            risk_levels.append('异常')
            # anomaly_type 只写维度类型，具体触发规则/模型在分维度列展示
            atype = DIM_CN[max_dim]
            # 数量异常按触发来源细分主类型（Q3横纵双超/Q1超常态/KDE分布离群）
            if max_dim == 'qty':
                atype = _qty_type_name(row.get('qty_rule_reason', ''))
            # C2规则主类型显示"高价供应商依赖"（非"高价聚量"）
            if max_dim == 'concen' and 'C2:' in str(row.get('concen_rule_reason', '')):
                atype = '高价供应商依赖'
            # 高价聚量标签污染修复：只有规则命中的才标"高价聚量"
            if max_dim == 'concen':
                concen_reason = row.get('concen_rule_reason', '')
                if pd.isna(concen_reason) or concen_reason == '':
                    atype = '隐形异常'
            anomaly_types.append(atype)
        else:
            risk_levels.append('无异常')
            anomaly_types.append('')

    df['risk_level'] = risk_levels
    df['anomaly_type'] = anomaly_types

    # 清理
    for d in DIMS:
        df.drop(columns=[f'boosted_{d}'], inplace=True, errors='ignore')

    return df
