# -*- coding: utf-8 -*-
"""概念漂移监控（简化版）"""

import json
import os
import numpy as np
from config import BASE_DIR

BASELINE_FILE = os.path.join(BASE_DIR, 'drift_baseline.json')


def check_drift(df):
    """比较当前批次分数分布与历史基线"""
    current = {}
    for dim in ['price', 'qty', 'concen']:
        col = f'{dim}_score'
        if col in df.columns:
            current[f'{dim}_mean'] = float(df[col].mean())
            current[f'{dim}_std'] = float(df[col].std())

    if not os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, 'w') as f:
            json.dump(current, f, indent=2)
        print("  drift baseline saved")
        return

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    for dim in ['price', 'qty', 'concen']:
        cur_mean = current.get(f'{dim}_mean', 0)
        his_mean = baseline.get(f'{dim}_mean', cur_mean)
        if his_mean > 0:
            delta = abs(cur_mean - his_mean) / his_mean * 100
            if delta > 20:
                print(f"  [WARN] {dim} score drift {delta:.1f}%")

    # 更新基线为当前值（指数移动平均）
    alpha = 0.3
    for k in current:
        baseline[k] = alpha * current[k] + (1 - alpha) * baseline.get(k, current[k])
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
