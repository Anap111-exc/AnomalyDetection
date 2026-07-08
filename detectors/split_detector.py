# -*- coding: utf-8 -*-
"""拆单检测器：纯规则(S1-S2)，不跑ML"""

import pandas as pd
import numpy as np
from config import *


def split_detector(df, rule_engine):
    print("  [split_detector] 执行中...（纯规则）")

    if 'split_score' not in df.columns:
        df['split_score'] = 0.0

    df = rule_engine.execute(df, dim='split')

    # split_score 保持 0 或 80/100
    print(f"    规则命中 {(df['split_score'] > 0).sum()} 行")
    return df
