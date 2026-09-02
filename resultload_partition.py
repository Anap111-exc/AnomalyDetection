# -*- coding: utf-8 -*-
"""
分区回写脚本（定时任务用）
- 结果表重建为【按订单月份分区】的分区表
- 回写按 submit 月份分组，逐月 INSERT OVERWRITE PARTITION：
    老月份（已存在）→ 覆盖 = 更新；新月份（不存在）→ 新建分区 = 插入
- batch 加大到 1000（单条 SQL ~0.5MB），40 万行全量回写预计 5~10 分钟
用法：
  export LIVY_URL="http://172.**.**.**:****"
  python resultload_partition.py /home/tione/notebook/dataresult/全量/全品类-全量标注.xlsx
"""
import os, sys, io, json, time, requests
import pandas as pd, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LIVY = os.environ.get("LIVY_URL", "http://172.**.**.**:****")
H = {"Content-Type": "application/json"}

TABLE = "ads_hznc.ads_hznc_trd_rpt_anomaly_result_df"

CREATE_SQL = f"""
CREATE TABLE {TABLE} (
  order_id STRING, ord_item_id STRING, submit_time STRING,
  smtr_name STRING, thd_dept_short_name STRING, sec_dept_short_name STRING,
  sup_short_name STRING, consigner_name STRING, rec_address STRING,
  sku_name STRING, proj_name STRING, four_cat_name STRING,
  tax_price DOUBLE, pur_qty DOUBLE, sub_ttl DOUBLE,
  anomaly_type STRING, risk_level STRING,
  price_type STRING, qty_type STRING, split_type STRING, concen_type STRING,
  detect_time STRING
)
PARTITIONED BY (submit_month STRING)
"""


class LivySQL:
    def __init__(self, url):
        self.url = url
        r = requests.post(f"{url}/sessions", headers=H, json={"kind": "sql"}, timeout=30)
        self.sid = r.json()["id"]
        for _ in range(40):
            if requests.get(f"{url}/sessions/{self.sid}", timeout=10).json()["state"] == "idle":
                break
            time.sleep(3)
        print(f"Livy SQL 会话就绪, ID = {self.sid}", flush=True)

    def sql(self, sql_text):
        r = requests.post(f"{self.url}/sessions/{self.sid}/statements",
                          headers=H, json={"code": sql_text}, timeout=300)
        stmt = r.json()["id"]
        for _ in range(120):
            res = requests.get(f"{self.url}/sessions/{self.sid}/statements/{stmt}", timeout=10).json()
            if res["state"] == "available":
                break
            time.sleep(2)
        out = res.get("output", {})
        if out.get("status") != "ok":
            raise Exception("SQL 失败: " + str(out))
        data = out.get("data", {})
        if "application/json" in data:
            j = data["application/json"]
            if isinstance(j, str):
                j = json.loads(j)
            return pd.DataFrame(j["data"], columns=[f["name"] for f in j["schema"]["fields"]])
        return None


def build_insert_overwrite_sql(df, table, month, batch=1000):
    """按月分区覆盖：INSERT OVERWRITE TABLE t PARTITION(month='...') VALUES ..."""
    cols = list(df.columns)
    col_sql = ", ".join(f"`{c}`" for c in cols)
    stmts = []
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        rows = []
        for _, row in part.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("NULL")
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    vals.append(repr(float(v)) if isinstance(v, float) else str(v))
                else:
                    s = str(v).replace("\n", " ").replace("\r", " ") \
                              .replace("\\", "\\\\").replace("'", "\\'")
                    vals.append(f"'{s}'")
            rows.append("(" + ", ".join(vals) + ")")
        stmts.append(
            f"INSERT OVERWRITE TABLE {table} PARTITION (submit_month='{month}') "
            f"({col_sql}) VALUES " + ", ".join(rows)
        )
    return stmts


def main():
    result_file = sys.argv[1] if len(sys.argv) >= 2 else \
        "/home/tione/notebook/dataresult/全量/全品类-全量标注.xlsx"

    spark = LivySQL(LIVY)

    print("1) 重建分区表（DROP + CREATE）...", flush=True)
    spark.sql(f"DROP TABLE IF EXISTS {TABLE}")
    spark.sql(CREATE_SQL)

    print(f"2) 读取结果: {result_file}", flush=True)
    df = pd.read_excel(result_file)
    df["submit_month"] = df["submit_time"].astype(str).str[:7]      # '2026-01'
    df["detect_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"   共 {len(df)} 行，涉及 {df['submit_month'].nunique()} 个订单月份", flush=True)

    t0 = time.time()
    total_stmts = 0
    for month, grp in df.groupby("submit_month"):
        grp = grp.drop(columns=["submit_month"])
        stmts = build_insert_overwrite_sql(grp, TABLE, month, batch=1000)
        for stmt in stmts:
            spark.sql(stmt)
            total_stmts += 1
        print(f"   {month}: {len(grp)} 行，{len(stmts)} 条语句，累计 {total_stmts} 条", flush=True)

    print(f"3) 写入完成，{len(df)} 行，{total_stmts} 条语句，耗时 {time.time()-t0:.0f}s", flush=True)

    print("4) 验证：", flush=True)
    spark.sql(f"SELECT COUNT(1) AS c FROM {TABLE}").show()
    spark.sql(f"SELECT submit_month, COUNT(1) c FROM {TABLE} GROUP BY submit_month ORDER BY submit_month").show()


if __name__ == "__main__":
    main()