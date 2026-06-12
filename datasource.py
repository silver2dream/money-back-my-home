# -*- coding: utf-8 -*-
"""資料來源層:多源容錯 + DB 快取 + 交叉驗證。

策略(對應授權與限流現實):
  - 單檔分析(精度優先):DB 快取 → Tiingo(含息 adjClose,免費層 50 檔/時、500 檔/月,
    僅供後端內部計算 = 條款內的 internal use)→ yfinance → Stooq
  - 批量掃描(量優先):DB 快取 → yfinance 批量 → Stooq 逐檔補
  - 每日排程:Tiingo/yfinance 輪替逐檔刷新 + 抽樣交叉驗證
  使用者請求永遠先吃 DB;外部呼叫集中在排程與冷啟動。

設 TIINGO_API_KEY 環境變數即啟用 Tiingo(強烈建議,免費註冊);未設則退 yfinance。
註:Stooq 已對程式化存取部署反爬(2026-06 實測回驗證頁),保留實作僅作末位備援。
"""
import io
import math
import os
import threading
import time

import numpy as np
import pandas as pd
import requests

import db

TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "").strip()
TIINGO_HOURLY_BUDGET = 45          # 官方上限 50 檔/時,留安全邊際
_tiingo_lock = threading.Lock()
_tiingo_window = {"start": 0.0, "count": 0}

UA = {"User-Agent": "Mozilla/5.0 (stock-analyzer; personal research)"}


class DataSourceError(Exception):
    """資料層錯誤(帶給使用者看的訊息)。"""


def _start_date(years: int) -> str:
    t = time.time() - years * 365.25 * 86400
    return time.strftime("%Y-%m-%d", time.gmtime(t))


# ---------------------------------------------------------------- Tiingo

def _tiingo_allow() -> bool:
    if not TIINGO_KEY:
        return False
    with _tiingo_lock:
        now = time.time()
        if now - _tiingo_window["start"] > 3600:
            _tiingo_window.update(start=now, count=0)
        if _tiingo_window["count"] >= TIINGO_HOURLY_BUDGET:
            return False
        _tiingo_window["count"] += 1
        return True


def _fetch_tiingo(ticker: str, years: int) -> dict:
    if not _tiingo_allow():
        raise DataSourceError("Tiingo 額度暫滿或未設定 API key")
    r = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
        params={"startDate": _start_date(years), "token": TIINGO_KEY},
        headers=UA, timeout=20)
    if r.status_code != 200:
        raise DataSourceError(f"Tiingo HTTP {r.status_code}")
    rows = r.json()
    if not isinstance(rows, list) or len(rows) < 2:
        raise DataSourceError("Tiingo 無資料")
    dates, closes, raw_closes, divs = [], [], [], []
    for row in rows:
        adj = row.get("adjClose")
        if adj is None or adj <= 0:
            continue
        dates.append(str(row["date"])[:10])
        closes.append(float(adj))
        raw_closes.append(float(row.get("close") or adj))
        divs.append(float(row.get("divCash") or 0.0))
    if len(closes) < 2:
        raise DataSourceError("Tiingo 有效資料不足")
    # 殖利率:近一年實際配息 ÷ 最新未調整價
    paid = sum(divs[-252:])
    div_yield = min(max(paid / raw_closes[-1], 0.0), 0.15) if raw_closes[-1] > 0 else 0.0
    name = ticker
    try:
        m = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}",
                         params={"token": TIINGO_KEY}, headers=UA, timeout=10).json()
        name = m.get("name") or ticker
    except Exception:
        pass
    return {"dates": dates, "closes": closes, "name": name, "currency": "USD",
            "div_yield": div_yield, "source": "tiingo"}


# ---------------------------------------------------------------- Stooq

def _fetch_stooq(ticker: str, years: int) -> dict:
    sym = ticker.lower().replace(".", "-") + ".us"
    r = requests.get("https://stooq.com/q/d/l/",
                     params={"s": sym, "i": "d",
                             "d1": _start_date(years).replace("-", ""),
                             "d2": time.strftime("%Y%m%d")},
                     headers=UA, timeout=20)
    body = r.text
    if r.status_code != 200 or not body.startswith("Date"):
        raise DataSourceError("Stooq 無資料或達每日限額")
    df = pd.read_csv(io.StringIO(body))
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    if len(df) < 2:
        raise DataSourceError("Stooq 有效資料不足")
    # Stooq 的 Close 預設即為含息+拆分調整後價格;它不提供名稱與股息明細
    old = db.prices_load(ticker, max_age_hours=24 * 3650)  # 保留既有 meta 的名稱/殖利率
    return {"dates": [str(d)[:10] for d in df["Date"]],
            "closes": [float(x) for x in df["Close"]],
            "name": old[2]["name"] if old else ticker,
            "currency": "USD",
            "div_yield": old[2]["div_yield"] if old else 0.0,
            "source": "stooq"}


# ---------------------------------------------------------------- yfinance(末位備援)

def _fetch_yfinance(ticker: str, years: int) -> dict:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True)
    if df is None or df.empty or "Close" not in df:
        raise DataSourceError("yfinance 無資料")
    closes = df["Close"].dropna()
    closes = closes[closes > 0]
    if len(closes) < 2:
        raise DataSourceError("yfinance 有效資料不足")
    div_yield = 0.0
    try:
        if "Dividends" in df.columns:
            paid = float(df["Dividends"].iloc[-252:].sum())
            div_yield = float(np.clip(paid / float(closes.iloc[-1]), 0.0, 0.15))
    except Exception:
        pass
    name, currency = ticker, "USD"
    try:
        info = tk.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
    except Exception:
        pass
    return {"dates": [d.strftime("%Y-%m-%d") for d in closes.index],
            "closes": [float(x) for x in closes],
            "name": name, "currency": currency, "div_yield": div_yield,
            "source": "yfinance"}


_FETCHERS = {"tiingo": _fetch_tiingo, "stooq": _fetch_stooq, "yfinance": _fetch_yfinance}


# ---------------------------------------------------------------- 公開介面

def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t or len(t) > 12:
        raise DataSourceError("請輸入有效的股票代碼(例如 AAPL、NVDA、SPY)。")
    return t


def _try_sources(ticker: str, years: int, order: list) -> dict:
    errors = []
    for src in order:
        try:
            return _FETCHERS[src](ticker, years)
        except Exception as exc:
            errors.append(f"{src}: {exc}")
    raise DataSourceError(f"「{ticker}」所有資料來源皆失敗({'; '.join(errors[:3])})。"
                          "請確認代碼是否正確,或稍後再試。")


def get_history(ticker: str, years: int = 10, force_refresh: bool = False):
    """單檔歷史(精度優先)。回傳 (closes ndarray, dates list, meta dict)。"""
    ticker = _normalize(ticker)
    if not force_refresh:
        hit = db.prices_load(ticker)
        if hit:
            closes, dates, meta = hit
            return np.asarray(closes, dtype=float), dates, meta
    data = _try_sources(ticker, years, ["tiingo", "yfinance", "stooq"])
    db.prices_save(ticker, data["dates"], data["closes"], data)
    meta = {"ticker": ticker, "name": data["name"], "currency": data["currency"],
            "div_yield": data["div_yield"], "source": data["source"],
            "as_of": data["dates"][-1]}
    return np.asarray(data["closes"], dtype=float), data["dates"], meta


def get_history_many(tickers: list, years: int = 10, progress=None) -> dict:
    """批量歷史(掃描用)。回傳 {ticker: (closes ndarray, dates, meta)};失敗的不在內。
    DB 快取優先;缺的先用 yfinance 批量,個別失敗再走 Stooq。"""
    out, missing = {}, []
    for t in tickers:
        hit = db.prices_load(t)
        if hit:
            out[t] = (np.asarray(hit[0], dtype=float), hit[1], hit[2])
        else:
            missing.append(t)

    if missing:
        import yfinance as yf
        CHUNK = 30
        for ci in range(0, len(missing), CHUNK):
            chunk = missing[ci: ci + CHUNK]
            if progress:
                progress("download", ci, len(missing), f"{chunk[0]} … {chunk[-1]}")
            try:
                df = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                                 progress=False, group_by="ticker", threads=True)
            except Exception:
                df = None
            for t in chunk:
                got = None
                if df is not None:
                    try:
                        col = df[t]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
                        c = col.dropna()
                        c = c[c > 0]
                        if len(c) >= 2:
                            got = {"dates": [d.strftime("%Y-%m-%d") for d in c.index],
                                   "closes": [float(x) for x in c],
                                   "name": t, "currency": "USD", "div_yield": 0.0,
                                   "source": "yfinance"}
                    except Exception:
                        got = None
                if got is None:
                    try:
                        got = _fetch_stooq(t, years)
                    except Exception:
                        continue
                db.prices_save(t, got["dates"], got["closes"], got)
                meta = {"ticker": t, "name": got["name"], "currency": got["currency"],
                        "div_yield": got["div_yield"], "source": got["source"],
                        "as_of": got["dates"][-1]}
                out[t] = (np.asarray(got["closes"], dtype=float), got["dates"], meta)
    return out


def cross_check(ticker: str) -> str:
    """雙源交叉驗證:Tiingo vs Stooq 最新共同收盤差 > 1.5% 回警告字串,否則空字串。"""
    try:
        a = _fetch_tiingo(ticker, 1)
        b = _fetch_stooq(ticker, 1)
    except Exception:
        return ""
    common = set(a["dates"]) & set(b["dates"])
    if not common:
        return ""
    d = max(common)
    pa = a["closes"][a["dates"].index(d)]
    pb = b["closes"][b["dates"].index(d)]
    diff = abs(pa - pb) / pb if pb else 0
    if diff > 0.015:
        return (f"{ticker} 雙源收盤價差異 {diff*100:.1f}%({d}:Tiingo {pa:.2f} vs "
                f"Stooq {pb:.2f}),請人工確認。")
    return ""


def refresh_for_cron(tickers: list, years: int = 10, log=print) -> dict:
    """每日排程刷新:Tiingo/yfinance 輪替逐檔強制更新,回統計。"""
    ok, failed = 0, []
    for i, t in enumerate(sorted(set(tickers))):
        rotation = (["tiingo", "yfinance", "stooq"] if i % 2 == 0
                    else ["yfinance", "tiingo", "stooq"])
        try:
            data = _try_sources(_normalize(t), years, rotation)
            db.prices_save(t, data["dates"], data["closes"], data)
            ok += 1
        except Exception as exc:
            failed.append(t)
            log(f"[cron] {t} 更新失敗:{exc}")
        time.sleep(0.5)            # 溫和節流
    warnings = []
    for t in sorted(set(tickers))[:10]:    # 抽樣交叉驗證
        w = cross_check(t)
        if w:
            warnings.append(w)
            log("[cron][warn] " + w)
    return {"ok": ok, "failed": failed, "warnings": warnings}
