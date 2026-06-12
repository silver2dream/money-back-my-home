# -*- coding: utf-8 -*-
"""美股蒙地卡羅策略分析 — Flask 伺服器(產品化版)

狀態全部落在資料庫(分析快取、掃描任務、掃描結果),多 gunicorn worker 下行為正確;
使用者身分由 auth 層解析(AUTH_MODE=none 本機單人 / supabase 生產多使用者)。
"""
import hashlib
import json
import math
import sys
import threading
import time

import pandas as pd
from flask import Flask, g, jsonify, request, send_from_directory
from sqlalchemy import select

import db
import tracker
from auth import public_config, require_user
from engine import EngineError, analyze, discover_scan
from rotation import rotation_run
from universe import UNIVERSE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__, static_folder="static")

CACHE_TTL = 900          # 分析結果快取 15 分鐘
SCAN_TTL = 86400         # 掃描結果同參數 24 小時內重用
JOB_STALE = 300          # 掃描任務心跳逾時(秒),視為殭屍


def _sanitize(obj):
    """JSON 不接受 NaN/Inf,遞迴轉成 None。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------- 速率限制(每 worker 近似)

_rl_lock = threading.Lock()
_rl: dict = {}


def rate_limit(bucket: str, per_minute: int):
    def deco(f):
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            key = (getattr(g, "user_id", "anon"), bucket)
            now = time.time()
            with _rl_lock:
                window = [t for t in _rl.get(key, []) if now - t < 60]
                if len(window) >= per_minute:
                    return jsonify({"error": "操作太頻繁,請稍候再試。"}), 429
                window.append(now)
                _rl[key] = window
            return f(*args, **kwargs)
        return wrapper
    return deco


_params_key = db.params_key


def _parse_strategy_params(args):
    return {
        "target": float(args.get("target", 15)) / 100.0,
        "max_hold": int(args.get("max_hold", 252)),
        "wait": int(args.get("wait", 63)),
        "stop": float(args.get("stop", 15)) / 100.0,
        "trail": float(args.get("trail", 8)) / 100.0,
        "cash_rate": float(args.get("cash", 4)) / 100.0,
        "conditional": args.get("cond", "1") != "0",
        "fat_tails": args.get("fat", "1") != "0",
        "div_tax": float(args.get("divtax", 30)) / 100.0,
    }


# ---------------------------------------------------------------- 基本路由

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/config")
def api_config():
    return jsonify(public_config())


@app.route("/api/analyze")
@require_user
@rate_limit("analyze", 10)
def api_analyze():
    try:
        ticker = request.args.get("ticker", "").strip().upper()
        p = _parse_strategy_params(request.args)
        n_paths = int(request.args.get("paths", 5000))
        years = int(request.args.get("years", 10))
    except ValueError:
        return jsonify({"error": "參數格式錯誤,請檢查輸入值。"}), 400

    key = _params_key("an", {**p, "ticker": ticker, "paths": n_paths, "years": years})
    hit = db.cache_get(key)
    if hit:
        return jsonify(hit)

    try:
        result = _sanitize(analyze(
            ticker, p["target"], p["max_hold"], p["wait"], n_paths, years,
            p["stop"], p["trail"], p["cash_rate"],
            p["conditional"], p["fat_tails"], p["div_tax"]))
    except EngineError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("analyze failed")
        return jsonify({"error": f"分析失敗:{exc}"}), 500

    db.cache_set(key, result, CACHE_TTL)
    return jsonify(result)


# ---------------------------------------------------------------- 市場探索(任務狀態與結果都在 DB)

def _run_discover(p: dict, years: int, params_key: str):
    state = {"running": True, "finished": False, "phase": "download",
             "done": 0, "total": len(UNIVERSE), "current": "", "params_key": params_key}

    def cb(phase, done, total, label):
        state.update(phase=phase, done=done, total=total, current=label)
        db.scan_job_set(state)

    try:
        out = _sanitize(discover_scan(UNIVERSE, p, n_paths=2000, years=years, progress=cb))
        db.scan_result_set(params_key, out)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("discover failed")
        db.scan_result_set(params_key,
                           {"results": [], "errors": [{"ticker": "*", "reason": str(exc)}],
                            "clusters": []})
    finally:
        state.update(running=False, finished=True, phase="done")
        db.scan_job_set(state)


@app.route("/api/discover/start", methods=["POST"])
@require_user
@rate_limit("discover", 4)
def discover_start():
    try:
        p = _parse_strategy_params(request.args)
        years = int(request.args.get("years", 10))
    except ValueError:
        return jsonify({"error": "參數格式錯誤,請檢查輸入值。"}), 400
    pk = _params_key("scan", {**p, "years": years})

    if db.scan_result_get(pk, SCAN_TTL):       # 同參數 24h 內 → 直接標記完成
        db.scan_job_set({"running": False, "finished": True, "phase": "done",
                         "done": len(UNIVERSE), "total": len(UNIVERSE),
                         "current": "", "params_key": pk})
        return jsonify({"ok": True, "cached": True, "total": len(UNIVERSE)})

    job = db.scan_job_get()
    if job and job.get("running") and time.time() - job.get("_heartbeat", 0) < JOB_STALE:
        return jsonify({"error": "市場掃描已在進行中。"}), 409

    db.scan_job_set({"running": True, "finished": False, "phase": "download",
                     "done": 0, "total": len(UNIVERSE), "current": "", "params_key": pk})
    threading.Thread(target=_run_discover, args=(p, years, pk), daemon=True).start()
    return jsonify({"ok": True, "total": len(UNIVERSE)})


@app.route("/api/discover/status")
@require_user
def discover_status():
    job = db.scan_job_get()
    if not job:
        return jsonify({"running": False, "finished": False, "phase": "", "done": 0,
                        "total": 0, "current": "", "results": None, "errors": [],
                        "clusters": []})
    if job.get("running") and time.time() - job.get("_heartbeat", 0) > JOB_STALE:
        job.update(running=False, finished=False,
                   error="掃描程序中斷,請重新啟動掃描。")
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    out.setdefault("results", None)
    out.setdefault("errors", [])
    out.setdefault("clusters", [])
    if job.get("finished") and job.get("params_key"):
        result = db.scan_result_get(job["params_key"], SCAN_TTL)
        if result:
            out.update(results=result.get("results"), errors=result.get("errors", []),
                       clusters=result.get("clusters", []))
    return jsonify(out)


# ---------------------------------------------------------------- 組合輪動(資料來自 price_eod)

def _load_series_map(tickers: list) -> dict:
    out = {}
    with db.engine.connect() as c:
        for t in tickers:
            rows = c.execute(select(db.price_eod.c.date, db.price_eod.c.close)
                             .where(db.price_eod.c.ticker == t)
                             .order_by(db.price_eod.c.date)).all()
            if len(rows) >= 2:
                out[t] = pd.Series([r.close for r in rows],
                                   index=pd.to_datetime([r.date for r in rows]))
    return out


@app.route("/api/rotation")
@require_user
@rate_limit("rotation", 6)
def api_rotation():
    try:
        p = _parse_strategy_params(request.args)
        variant = request.args.get("variant", "trail_stop")
        max_pos = max(1, min(20, int(request.args.get("max_pos", 5))))
        tickers = [t.strip().upper() for t in request.args.get("tickers", "").split(",") if t.strip()]
    except ValueError:
        return jsonify({"error": "參數格式錯誤。"}), 400
    if not tickers:
        tickers = list(UNIVERSE.keys())
    series = _load_series_map(tickers)
    if len(series) < 2:
        return jsonify({"error": "價格資料不足,請先執行「探索市場」讓系統下載歷史資料。"}), 400
    try:
        out = _sanitize(rotation_run(series, list(series.keys()), p, variant, max_pos))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("rotation failed")
        return jsonify({"error": f"輪動回測失敗:{exc}"}), 500
    return jsonify(out)


# ---------------------------------------------------------------- 建議追蹤(per-user)

@app.route("/api/track/add", methods=["POST"])
@require_user
@rate_limit("track", 30)
def track_add():
    try:
        rid = tracker.add_rec(g.user_id, request.get_json(force=True))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"記錄失敗:{exc}"}), 400
    return jsonify({"ok": True, "id": rid})


@app.route("/api/track/list")
@require_user
@rate_limit("track", 30)
def track_list():
    try:
        return jsonify(_sanitize(tracker.list_recs(g.user_id)))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("track list failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/track/remove", methods=["POST"])
@require_user
def track_remove():
    try:
        tracker.remove_rec(g.user_id, int(request.args.get("id", 0)))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("US Stock Monte Carlo Analyzer -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
