# -*- coding: utf-8 -*-
"""每日排程:美股收盤後執行,讓白天所有使用者請求都只讀資料庫。

  1. 更新價格:股票池 + 所有使用者追蹤中的標的(Tiingo/yfinance 輪替 + 抽樣交叉驗證)
  2. 批次結算:推進所有使用者的建議追蹤狀態機
  3. 系統掃描:以預設參數跑一次全市場掃描存檔,使用者按「探索市場」秒回

用法:
  python daily_job.py            # 立即執行一次
  python daily_job.py --loop     # 常駐,每日 UTC 22:30(美股收盤後)自動執行
"""
import math
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import db
import datasource
import tracker
from universe import UNIVERSE

RUN_AT_UTC = (22, 30)        # 美股收盤(20:00/21:00 UTC)後的安全時點
DEFAULT_SCAN_PARAMS = {      # 與前端預設一致,使用者用預設參數探索時直接命中
    "target": 0.15, "max_hold": 252, "wait": 63,
    "stop": 0.15, "trail": 0.08, "cash_rate": 0.04,
    "conditional": True, "fat_tails": True, "div_tax": 0.30,
}
DEFAULT_YEARS = 10


def _sanitize(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def run_once():
    t0 = time.time()
    tickers = sorted(set(UNIVERSE) | set(db.tracked_tickers()))
    print(f"[daily] 開始:更新 {len(tickers)} 檔價格")
    stats = datasource.refresh_for_cron(tickers, DEFAULT_YEARS)
    print(f"[daily] 價格更新完成:成功 {stats['ok']},失敗 {stats['failed'] or '無'}")

    tracker.settle_all()

    print("[daily] 跑系統掃描(預設參數)…")
    from engine import discover_scan
    pk = db.params_key("scan", {**DEFAULT_SCAN_PARAMS, "years": DEFAULT_YEARS})
    out = _sanitize(discover_scan(UNIVERSE, DEFAULT_SCAN_PARAMS,
                                  n_paths=2000, years=DEFAULT_YEARS))
    db.scan_result_set(pk, out)
    print(f"[daily] 完成,共 {time.time()-t0:.0f} 秒;"
          f"掃描 {len(out['results'])} 檔(timing="
          f"{sum(1 for r in out['results'] if r['category']=='timing')})")


def _seconds_until_next_run() -> float:
    now = time.gmtime()
    today_run = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                             RUN_AT_UTC[0], RUN_AT_UTC[1], 0, 0, 0, 0)) - time.timezone
    wait = today_run - time.time()
    if wait <= 0:
        wait += 86400
    return wait


def loop():
    print(f"[daily] 常駐模式,每日 {RUN_AT_UTC[0]:02d}:{RUN_AT_UTC[1]:02d} UTC 執行")
    while True:
        wait = _seconds_until_next_run()
        print(f"[daily] 下次執行於 {wait/3600:.1f} 小時後")
        time.sleep(wait)
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001
            print(f"[daily] 執行失敗:{exc}")
            time.sleep(600)      # 失敗後稍候,避免緊迴圈


if __name__ == "__main__":
    if "--loop" in sys.argv:
        loop()
    else:
        run_once()
