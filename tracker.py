# -*- coding: utf-8 -*-
"""建議追蹤(track record):記錄工具給出的每筆建議,之後自動對答案。

「跟著投資」的信任不該來自回測,而來自事後可查的成績單。
狀態機(與引擎出場規則一致,皆以收盤判定):
  timing 建議:waiting(等限價觸發)→ active(持有中)→ done(出場)/ expired(等待逾期)
  hold 建議:active(建議日現價買入,持續對答案,手動移除)
"""
import os
import sqlite3
import threading
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker.db")
_lock = threading.Lock()
_last_refresh = 0.0
REFRESH_TTL = 60  # 秒

SCHEMA = """CREATE TABLE IF NOT EXISTS recs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  ticker TEXT NOT NULL,
  name TEXT,
  kind TEXT NOT NULL,
  dip REAL, entry_price REAL, variant TEXT,
  target REAL, stop REAL, trail REAL,
  wait INTEGER, max_hold INTEGER,
  spot_at_rec REAL, expected_ann REAL, expected_days REAL,
  status TEXT NOT NULL,
  entry_date TEXT, exit_date TEXT, exit_price REAL, exit_reason TEXT,
  last_price REAL, last_date TEXT, current_return REAL
)"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute(SCHEMA)


def add_rec(d: dict) -> int:
    today = dt.date.today().isoformat()
    kind = d.get("kind", "timing")
    status = "active" if (kind == "hold" or float(d.get("dip") or 0) <= 0) else "waiting"
    row = {
        "created_at": today,
        "ticker": str(d["ticker"]).upper()[:12],
        "name": str(d.get("name") or "")[:80],
        "kind": kind,
        "dip": float(d.get("dip") or 0),
        "entry_price": float(d["entry_price"]),
        "variant": d.get("variant", "fixed"),
        "target": float(d.get("target") or 0),
        "stop": float(d.get("stop") or 0),
        "trail": float(d.get("trail") or 0),
        "wait": int(d.get("wait") or 63),
        "max_hold": int(d.get("max_hold") or 252),
        "spot_at_rec": float(d.get("spot_at_rec") or 0),
        "expected_ann": d.get("expected_ann"),
        "expected_days": d.get("expected_days"),
        "status": status,
        "entry_date": today if status == "active" else None,
    }
    cols = ", ".join(row)
    ph = ", ".join("?" * len(row))
    with _lock, _conn() as c:
        cur = c.execute(f"INSERT INTO recs ({cols}) VALUES ({ph})", list(row.values()))
        return cur.lastrowid


def remove_rec(rid: int):
    with _lock, _conn() as c:
        c.execute("DELETE FROM recs WHERE id=?", (rid,))


def _advance(row: dict, sub: pd.Series) -> dict:
    """以建議日(不含)之後的收盤序列推進單筆狀態機,回傳要更新的欄位。"""
    upd = {}
    if len(sub) == 0:
        return upd
    closes = sub.to_numpy(dtype=float)
    dates = [d.strftime("%Y-%m-%d") for d in sub.index]
    upd["last_price"] = float(closes[-1])
    upd["last_date"] = dates[-1]

    if row["kind"] == "hold":
        upd["current_return"] = float(closes[-1] / row["entry_price"] - 1)
        return upd

    status = row["status"]
    entry_i = None
    if status == "waiting":
        hit = np.where(closes <= row["entry_price"])[0]
        if len(hit) and hit[0] < row["wait"]:
            entry_i = int(hit[0])
            status = "active"
            upd["status"], upd["entry_date"] = "active", dates[entry_i]
        elif len(closes) >= row["wait"]:
            upd["status"] = "expired"
            upd["current_return"] = None
            return upd
        else:
            return upd
    elif status == "active":
        entry_i = dates.index(row["entry_date"]) if row["entry_date"] in dates else 0

    if status != "active":
        return upd

    # 持有中:逐日套用出場規則(與引擎一致,收盤判定)
    use_stop = row["variant"] in ("fixed_stop", "trail_stop")
    use_trail = row["variant"] in ("trail", "trail_stop")
    e = row["entry_price"]
    hit, runmax = False, 0.0
    for i in range(entry_i + 1, len(closes)):
        px = closes[i]
        rel = px / e
        held = i - entry_i
        exit_price = reason = None
        if not hit and rel >= 1 + row["target"]:
            if use_trail:
                hit, runmax = True, px
            else:
                exit_price, reason = e * (1 + row["target"]), "達標"
        elif hit and use_trail:
            runmax = max(runmax, px)
            if px <= runmax * (1 - row["trail"]):
                exit_price, reason = px, "移動停利"
        if exit_price is None and use_stop and not hit and rel <= 1 - row["stop"]:
            exit_price, reason = px, "停損"
        if exit_price is None and held >= row["max_hold"]:
            exit_price, reason = px, "持有期滿"
        if exit_price is not None:
            upd.update(status="done", exit_date=dates[i], exit_price=float(exit_price),
                       exit_reason=reason, current_return=float(exit_price / e - 1))
            return upd
    upd["current_return"] = float(closes[-1] / e - 1)
    return upd


def refresh_all():
    """更新所有未結案紀錄(60 秒節流)。"""
    global _last_refresh
    if time.time() - _last_refresh < REFRESH_TTL:
        return
    with _conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM recs WHERE status IN ('waiting','active')")]
    if not rows:
        _last_refresh = time.time()
        return
    tickers = sorted({r["ticker"] for r in rows})
    start = min(r["created_at"] for r in rows)
    try:
        df = yf.download(tickers, start=start, interval="1d", auto_adjust=True,
                         progress=False, group_by="ticker", threads=True)
    except Exception:
        return
    for r in rows:
        try:
            col = df[r["ticker"]]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
            sub = col.dropna()
            sub = sub[sub.index.strftime("%Y-%m-%d") > r["created_at"]]
            upd = _advance(r, sub)
        except Exception:
            continue
        if upd:
            sets = ", ".join(f"{k}=?" for k in upd)
            with _lock, _conn() as c:
                c.execute(f"UPDATE recs SET {sets} WHERE id=?", [*upd.values(), r["id"]])
    _last_refresh = time.time()


def list_recs(refresh: bool = True) -> dict:
    if refresh:
        refresh_all()
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM recs ORDER BY id DESC")]
    closed = [r for r in rows if r["status"] == "done"]
    wins = [r for r in closed if (r["current_return"] or 0) > 0]
    summary = {
        "total": len(rows),
        "open": sum(1 for r in rows if r["status"] in ("waiting", "active")),
        "closed": len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "avg_return": (sum(r["current_return"] or 0 for r in closed) / len(closed))
                      if closed else None,
    }
    return {"rows": rows, "summary": summary}


init_db()
