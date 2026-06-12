# -*- coding: utf-8 -*-
"""資料庫層:SQLAlchemy Core,開發用 SQLite、生產用 Postgres(Supabase),由
DATABASE_URL 環境變數切換,SQL 寫法保持兩庫相容(日期一律存 ISO 字串)。

表結構:
  recommendations  使用者建議追蹤(唯一 per-user 的表,user_id 來自 Supabase Auth)
  price_eod        每日調整後收盤價(全域共享,由每日排程更新)
  price_meta       每檔標的的名稱/殖利率/資料來源/更新時間
  analysis_cache   分析結果快取(多 worker 共享,取代記憶體 dict)
  scan_results     市場掃描結果(全域共享,同參數 24 小時內直接重用)
  scan_jobs        掃描任務狀態(單列,多 worker 下取代記憶體 _djob)
"""
import json
import os
import time

from sqlalchemy import (Column, Float, Integer, MetaData, PrimaryKeyConstraint,
                        String, Table, Text, create_engine, delete, select, text)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(BASE_DIR, "local.db"))
if DATABASE_URL.startswith("postgres://"):          # Heroku/Supabase 舊式 scheme
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
metadata = MetaData()

recommendations = Table(
    "recommendations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column("created_at", String(10), nullable=False),
    Column("ticker", String(12), nullable=False),
    Column("name", String(120)),
    Column("kind", String(12), nullable=False),
    Column("dip", Float), Column("entry_price", Float), Column("variant", String(16)),
    Column("target", Float), Column("stop", Float), Column("trail", Float),
    Column("wait", Integer), Column("max_hold", Integer),
    Column("spot_at_rec", Float), Column("expected_ann", Float), Column("expected_days", Float),
    Column("status", String(12), nullable=False),
    Column("entry_date", String(10)), Column("exit_date", String(10)),
    Column("exit_price", Float), Column("exit_reason", String(20)),
    Column("last_price", Float), Column("last_date", String(10)),
    Column("current_return", Float),
)

price_eod = Table(
    "price_eod", metadata,
    Column("ticker", String(12), nullable=False),
    Column("date", String(10), nullable=False),
    Column("close", Float, nullable=False),
    PrimaryKeyConstraint("ticker", "date"),
)

price_meta = Table(
    "price_meta", metadata,
    Column("ticker", String(12), primary_key=True),
    Column("name", String(120)),
    Column("currency", String(8)),
    Column("div_yield", Float),
    Column("source", String(16)),
    Column("updated_at", Float),     # epoch 秒
)

analysis_cache = Table(
    "analysis_cache", metadata,
    Column("cache_key", String(80), primary_key=True),
    Column("payload", Text, nullable=False),
    Column("expires_at", Float, nullable=False),
)

scan_results = Table(
    "scan_results", metadata,
    Column("params_key", String(80), primary_key=True),
    Column("payload", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)

scan_jobs = Table(
    "scan_jobs", metadata,
    Column("job_id", String(16), primary_key=True),   # 固定 'global'(系統級掃描)
    Column("state", Text, nullable=False),            # JSON:phase/done/total/...
    Column("heartbeat", Float, nullable=False),
)


def init_db():
    metadata.create_all(engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.connect() as c:
            c.execute(text("PRAGMA journal_mode=WAL"))


def params_key(prefix: str, payload: dict) -> str:
    """參數 → 確定性快取鍵(伺服器與每日排程共用,確保掃描結果能互相命中)。"""
    import hashlib
    body = json.dumps(payload, sort_keys=True)
    return prefix + ":" + hashlib.sha1(body.encode()).hexdigest()[:32]


# ---------------------------------------------------------------- 分析快取

def cache_get(key: str):
    with engine.connect() as c:
        row = c.execute(select(analysis_cache.c.payload, analysis_cache.c.expires_at)
                        .where(analysis_cache.c.cache_key == key)).first()
    if row and row.expires_at > time.time():
        return json.loads(row.payload)
    return None


def cache_set(key: str, payload: dict, ttl: int = 900):
    body = json.dumps(payload, ensure_ascii=False)
    with engine.begin() as c:
        c.execute(delete(analysis_cache).where(analysis_cache.c.cache_key == key))
        c.execute(analysis_cache.insert().values(
            cache_key=key, payload=body, expires_at=time.time() + ttl))
        # 順手清過期項,避免表無限長大
        c.execute(delete(analysis_cache).where(analysis_cache.c.expires_at < time.time()))


# ---------------------------------------------------------------- 掃描結果與任務狀態

def scan_result_get(params_key: str, max_age: float = 86400):
    with engine.connect() as c:
        row = c.execute(select(scan_results.c.payload, scan_results.c.created_at)
                        .where(scan_results.c.params_key == params_key)).first()
    if row and time.time() - row.created_at < max_age:
        return json.loads(row.payload)
    return None


def scan_result_set(params_key: str, payload: dict):
    with engine.begin() as c:
        c.execute(delete(scan_results).where(scan_results.c.params_key == params_key))
        c.execute(scan_results.insert().values(
            params_key=params_key, payload=json.dumps(payload, ensure_ascii=False),
            created_at=time.time()))
        c.execute(delete(scan_results).where(scan_results.c.created_at < time.time() - 7 * 86400))


def scan_job_get():
    with engine.connect() as c:
        row = c.execute(select(scan_jobs.c.state, scan_jobs.c.heartbeat)
                        .where(scan_jobs.c.job_id == "global")).first()
    if not row:
        return None
    state = json.loads(row.state)
    state["_heartbeat"] = row.heartbeat
    return state


def scan_job_set(state: dict):
    body = json.dumps({k: v for k, v in state.items() if not k.startswith("_")},
                      ensure_ascii=False)
    with engine.begin() as c:
        c.execute(delete(scan_jobs).where(scan_jobs.c.job_id == "global"))
        c.execute(scan_jobs.insert().values(job_id="global", state=body, heartbeat=time.time()))


# ---------------------------------------------------------------- 價格資料存取

def prices_save(ticker: str, dates: list, closes, meta: dict):
    """整檔覆寫(調整後價格在除息/拆分後整條歷史都會變,不能增量)。"""
    rows = [{"ticker": ticker, "date": d, "close": float(p)} for d, p in zip(dates, closes)]
    with engine.begin() as c:
        c.execute(delete(price_eod).where(price_eod.c.ticker == ticker))
        for i in range(0, len(rows), 1000):
            c.execute(price_eod.insert(), rows[i:i + 1000])
        c.execute(delete(price_meta).where(price_meta.c.ticker == ticker))
        c.execute(price_meta.insert().values(
            ticker=ticker, name=meta.get("name", ticker),
            currency=meta.get("currency", "USD"),
            div_yield=float(meta.get("div_yield") or 0.0),
            source=meta.get("source", ""), updated_at=time.time()))


def prices_load(ticker: str, max_age_hours: float = 30):
    """從 DB 讀取;資料過舊或不存在回 None。回傳 (closes list, dates list, meta dict)。"""
    with engine.connect() as c:
        m = c.execute(select(price_meta).where(price_meta.c.ticker == ticker)).first()
        if not m or time.time() - m.updated_at > max_age_hours * 3600:
            return None
        rows = c.execute(select(price_eod.c.date, price_eod.c.close)
                         .where(price_eod.c.ticker == ticker)
                         .order_by(price_eod.c.date)).all()
    if len(rows) < 2:
        return None
    dates = [r.date for r in rows]
    closes = [r.close for r in rows]
    meta = {"ticker": ticker, "name": m.name or ticker, "currency": m.currency or "USD",
            "div_yield": m.div_yield or 0.0, "source": m.source or "db",
            "as_of": dates[-1]}
    return closes, dates, meta


def tracked_tickers() -> list:
    """所有使用者未結案紀錄涉及的標的(每日排程需要更新它們)。"""
    with engine.connect() as c:
        rows = c.execute(select(recommendations.c.ticker).distinct()
                         .where(recommendations.c.status.in_(["waiting", "active"]))).all()
    return [r.ticker for r in rows]


init_db()
