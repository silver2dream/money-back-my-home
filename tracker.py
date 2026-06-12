# -*- coding: utf-8 -*-
"""建議追蹤(track record):記錄工具給出的每筆建議,之後自動對答案。

產品化版本:
  - 每筆紀錄綁定 user_id(Supabase Auth 的 sub;本機模式為 'local')
  - 結算資料一律來自 price_eod 表(由每日排程更新),不直接呼叫外部資料源
  - settle_all() 供每日排程批次結算全部使用者;list_recs() 對過舊資料做輕量補刷

狀態機(與引擎出場規則一致,皆以收盤判定):
  timing 建議:waiting(等限價觸發)→ active(持有中)→ done(出場)/ expired(等待逾期)
  hold 建議:active(建議日現價買入,持續對答案,手動移除)
"""
import datetime as dt
import time

import numpy as np
from sqlalchemy import and_, delete, select

import db
from db import engine, price_eod, price_meta, recommendations


def add_rec(user_id: str, d: dict) -> int:
    today = dt.date.today().isoformat()
    kind = d.get("kind", "timing")
    status = "active" if (kind == "hold" or float(d.get("dip") or 0) <= 0) else "waiting"
    row = {
        "user_id": user_id,
        "created_at": today,
        "ticker": str(d["ticker"]).upper()[:12],
        "name": str(d.get("name") or "")[:120],
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
    with engine.begin() as c:
        rid = c.execute(recommendations.insert().values(**row)).inserted_primary_key[0]
    return int(rid)


def remove_rec(user_id: str, rid: int):
    with engine.begin() as c:
        c.execute(delete(recommendations).where(
            and_(recommendations.c.id == rid, recommendations.c.user_id == user_id)))


def _closes_since(ticker: str, after_date: str):
    """price_eod 中該標的「日期 > after_date」的收盤序列。回傳 (closes, dates)。"""
    with engine.connect() as c:
        rows = c.execute(select(price_eod.c.date, price_eod.c.close)
                         .where(and_(price_eod.c.ticker == ticker,
                                     price_eod.c.date > after_date))
                         .order_by(price_eod.c.date)).all()
    return [r.close for r in rows], [r.date for r in rows]


def _advance(row: dict, closes: list, dates: list) -> dict:
    """以建議日之後的收盤序列推進單筆狀態機,回傳要更新的欄位。"""
    upd = {}
    if not closes:
        return upd
    upd["last_price"] = float(closes[-1])
    upd["last_date"] = dates[-1]

    if row["kind"] == "hold":
        upd["current_return"] = float(closes[-1] / row["entry_price"] - 1)
        return upd

    status = row["status"]
    entry_i = None
    if status == "waiting":
        hit = [i for i, p in enumerate(closes) if p <= row["entry_price"]]
        if hit and hit[0] < row["wait"]:
            entry_i = hit[0]
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

    use_stop = row["variant"] in ("fixed_stop", "trail_stop")
    use_trail = row["variant"] in ("trail", "trail_stop")
    e = row["entry_price"]
    hit_flag, runmax = False, 0.0
    for i in range(entry_i + 1, len(closes)):
        px = closes[i]
        rel = px / e
        held = i - entry_i
        exit_price = reason = None
        if not hit_flag and rel >= 1 + row["target"]:
            if use_trail:
                hit_flag, runmax = True, px
            else:
                exit_price, reason = e * (1 + row["target"]), "達標"
        elif hit_flag and use_trail:
            runmax = max(runmax, px)
            if px <= runmax * (1 - row["trail"]):
                exit_price, reason = px, "移動停利"
        if exit_price is None and use_stop and not hit_flag and rel <= 1 - row["stop"]:
            exit_price, reason = px, "停損"
        if exit_price is None and held >= row["max_hold"]:
            exit_price, reason = px, "持有期滿"
        if exit_price is not None:
            upd.update(status="done", exit_date=dates[i], exit_price=float(exit_price),
                       exit_reason=reason, current_return=float(exit_price / e - 1))
            return upd
    upd["current_return"] = float(closes[-1] / e - 1)
    return upd


def _settle_rows(rows: list):
    """對一批未結案紀錄執行狀態機並寫回。"""
    for r in rows:
        closes, dates = _closes_since(r["ticker"], r["created_at"])
        upd = _advance(r, closes, dates)
        if upd:
            with engine.begin() as c:
                c.execute(recommendations.update()
                          .where(recommendations.c.id == r["id"]).values(**upd))


def settle_all(log=print) -> int:
    """每日排程:批次結算所有使用者的未結案紀錄(資料來自 price_eod)。"""
    with engine.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(
            select(recommendations).where(
                recommendations.c.status.in_(["waiting", "active"])))]
    _settle_rows(rows)
    log(f"[settle] 已結算 {len(rows)} 筆未結案紀錄")
    return len(rows)


def list_recs(user_id: str, refresh: bool = True) -> dict:
    """單一使用者的紀錄;refresh 時對價格過舊(>26h)的追蹤標的做輕量補刷再結算。"""
    with engine.connect() as c:
        open_rows = [dict(r._mapping) for r in c.execute(
            select(recommendations).where(and_(
                recommendations.c.user_id == user_id,
                recommendations.c.status.in_(["waiting", "active"]))))]

    if refresh and open_rows:
        stale = set()
        with engine.connect() as c:
            for t in {r["ticker"] for r in open_rows}:
                m = c.execute(select(price_meta.c.updated_at)
                              .where(price_meta.c.ticker == t)).first()
                if not m or time.time() - m.updated_at > 26 * 3600:
                    stale.add(t)
        if stale:
            from datasource import get_history
            for t in stale:
                try:
                    get_history(t, 10, force_refresh=True)
                except Exception:
                    pass
        _settle_rows(open_rows)

    with engine.connect() as c:
        rows = [dict(r._mapping) for r in c.execute(
            select(recommendations).where(recommendations.c.user_id == user_id)
            .order_by(recommendations.c.id.desc()))]
    closed = [r for r in rows if r["status"] == "done"]
    wins = [r for r in closed if (r["current_return"] or 0) > 0]
    return {"rows": rows, "summary": {
        "total": len(rows),
        "open": sum(1 for r in rows if r["status"] in ("waiting", "active")),
        "closed": len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "avg_return": (sum(r["current_return"] or 0 for r in closed) / len(closed))
                      if closed else None,
    }}
