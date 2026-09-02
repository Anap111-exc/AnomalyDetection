# -*- coding: utf-8 -*-
"""
滚动取数脚本（定时任务用）
- 只取 dwd_hznc_trd_pur_ord_det_df 的【ds 最大分区】（最新全量快照，消除跨分区重复，约省 2/3 数据量）
- 取数窗口：默认最近 180 天（取数日 -180 天 ~ 取数日），避免全量抽取的时间成本
- 按天循环 + row_number 分页（每批 1000 行，绕 Livy 上限；已修复旧脚本"每页查两次写两遍"的 bug）
用法：
  export LIVY_URL="http://172.**.**.**:****"
  python dataload_rolling.py                      # 默认窗口 180 天，输出 pur_ord_rolling.csv
  python dataload_rolling.py 90                   # 指定窗口 90 天
  python dataload_rolling.py 180 /home/tione/notebook/ad_dataload/pur_ord_rolling.csv
"""
import os, sys, csv, io, json, time, glob, requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LIVY = os.environ.get("LIVY_URL", "http://172.**.**.**:****")
H = {"Content-Type": "application/json"}

TARGET_TABLE = "cdm_hznc.dwd_hznc_trd_pur_ord_det_df"
FIELDS = """order_id, order_no, ord_item_id, submit_time,
    ord_type_name, ord_status_name, smtr_name, thd_dept_short_name,
    sec_dept_short_name, sec_dept_type, material_type, store_pzn_level,
    sup_short_name, company_name, store_name, sku_id, sku_code, sku_name,
    material_type_name, is_povt_alevt, one_cat_name, two_cat_name,
    three_cat_name, four_cat_name, measure_unit_name, pur_qty,
    origin_tax_price, tax_price, sub_ttl, apply_people_name,
    consigner_name, proj_name, rec_address"""


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
                          headers=H, json={"code": sql_text}, timeout=120)
        stmt = r.json()["id"]
        for _ in range(60):
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


def main():
    window_days = int(sys.argv[1]) if len(sys.argv) >= 2 else 180
    save_path = sys.argv[2] if len(sys.argv) >= 3 else "/home/tione/notebook/ad_dataload/pur_ord_rolling.csv"

    spark = LivySQL(LIVY)

    # 安全覆盖
    if os.path.exists(save_path):
        os.remove(save_path)
        print("已删除旧文件，准备干净覆盖", flush=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 窗口起点 = 今天 - window_days
    start_day = (pd.Timestamp.now() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"取数窗口: {start_day} ~ {today}（{window_days} 天）", flush=True)

    # 窗口内所有业务日
    date_df = spark.sql(f"""
        SELECT DISTINCT date_format(submit_time, 'yyyy-MM-dd') as dt
        FROM {TARGET_TABLE}
        WHERE submit_time >= '{start_day}' AND submit_time <= '{today} 23:59:59'
        ORDER BY dt
    """)
    all_days = date_df["dt"].tolist()
    print(f"窗口内共 {len(all_days)} 个业务日", flush=True)

    batch = 1000
    written = 0
    for day in all_days:
        # 只取 ds 最大分区（最新全量快照，消除跨分区重复）
        cnt = int(spark.sql(
            f"SELECT COUNT(*) c FROM {TARGET_TABLE} "
            f"WHERE submit_time LIKE '{day}%' "
            f"AND ds = (SELECT max(ds) FROM {TARGET_TABLE})"
        ).iloc[0]["c"])
        if cnt == 0:
            continue

        for off in range(0, cnt, batch):
            start, end = off + 1, off + batch
            sql = f"""
            SELECT * FROM (
                SELECT {FIELDS},
                       row_number() over (order by ord_item_id) as rn
                FROM {TARGET_TABLE}
                WHERE submit_time LIKE '{day}%'
                  AND ds = (SELECT max(ds) FROM {TARGET_TABLE})
            ) t WHERE rn >= {start} AND rn <= {end}
            """
            df = spark.sql(sql)
            if df is None or df.empty:
                break
            df.drop(columns=["rn"], inplace=True)
            df.to_csv(save_path, mode="a", index=False, header=(written == 0), quoting=csv.QUOTE_ALL)
            written += len(df)
        print(f"  {day} 抽完，当日 {cnt} 行，累计 {written} 行", flush=True)

    print(f"抽取完成，落盘 {written} 行 -> {save_path}", flush=True)
    if written > 0:
        print("校验首行:", open(save_path, encoding="utf-8", errors="replace").readline()[:120], flush=True)


if __name__ == "__main__":
    main()