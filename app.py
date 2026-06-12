# -*- coding: utf-8 -*-
"""美股蒙地卡羅策略分析 — Flask 伺服器"""
import math
import sys
import time
import threading

from flask import Flask, jsonify, request, send_from_directory

import tracker
from engine import EngineError, analyze, discover_scan
from rotation import rotation_run
from universe import UNIVERSE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__, static_folder="static")

CACHE_TTL = 900  # 同參數結果快取 15 分鐘,避免重複下載資料
_cache: dict = {}
_lock = threading.Lock()

# 市場探索背景任務(單一任務即可,本機單人使用)
_djob = {"running": False, "finished": False, "phase": "", "done": 0,
         "total": 0, "current": "", "results": None, "errors": [], "clusters": []}
_djob_lock = threading.Lock()
_scan_store = {"series": None, "params": None}   # 掃描原始價格序列(供輪動回測)


def _sanitize(obj):
    """JSON 不接受 NaN/Inf,遞迴轉成 None。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze")
def api_analyze():
    try:
        ticker = request.args.get("ticker", "").strip().upper()
        target = float(request.args.get("target", 15)) / 100.0
        max_hold = int(request.args.get("max_hold", 252))
        wait = int(request.args.get("wait", 63))
        n_paths = int(request.args.get("paths", 5000))
        years = int(request.args.get("years", 10))
        stop = float(request.args.get("stop", 15)) / 100.0
        trail = float(request.args.get("trail", 8)) / 100.0
        cash = float(request.args.get("cash", 4)) / 100.0
        conditional = request.args.get("cond", "1") != "0"
        fat_tails = request.args.get("fat", "1") != "0"
        div_tax = float(request.args.get("divtax", 30)) / 100.0
    except ValueError:
        return jsonify({"error": "參數格式錯誤,請檢查輸入值。"}), 400

    key = (ticker, round(target, 4), max_hold, wait, n_paths, years,
           round(stop, 4), round(trail, 4), round(cash, 4),
           conditional, fat_tails, round(div_tax, 4))
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return jsonify(hit[1])

    try:
        result = _sanitize(analyze(ticker, target, max_hold, wait, n_paths, years,
                                   stop, trail, cash, conditional, fat_tails, div_tax))
    except EngineError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("analyze failed")
        return jsonify({"error": f"分析失敗:{exc}"}), 500

    with _lock:
        _cache[key] = (time.time(), result)
    return jsonify(result)


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


def _run_discover(p, years):
    def cb(phase, done, total, label):
        with _djob_lock:
            _djob.update(phase=phase, done=done, total=total, current=label)
    try:
        out = discover_scan(UNIVERSE, p, n_paths=2000, years=years, progress=cb)
        _scan_store["series"] = out.pop("series", None)
        _scan_store["params"] = p
        out = _sanitize(out)
        with _djob_lock:
            _djob.update(results=out["results"], errors=out["errors"],
                         clusters=out.get("clusters", []))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("discover failed")
        with _djob_lock:
            _djob["errors"] = [{"ticker": "*", "reason": str(exc)}]
    finally:
        with _djob_lock:
            _djob.update(running=False, finished=True, phase="done")


@app.route("/api/discover/start", methods=["POST"])
def discover_start():
    try:
        p = _parse_strategy_params(request.args)
        years = int(request.args.get("years", 10))
    except ValueError:
        return jsonify({"error": "參數格式錯誤,請檢查輸入值。"}), 400
    with _djob_lock:
        if _djob["running"]:
            return jsonify({"error": "市場掃描已在進行中。"}), 409
        _djob.update(running=True, finished=False, phase="download", done=0,
                     total=len(UNIVERSE), current="", results=None, errors=[])
    threading.Thread(target=_run_discover, args=(p, years), daemon=True).start()
    return jsonify({"ok": True, "total": len(UNIVERSE)})


@app.route("/api/discover/status")
def discover_status():
    with _djob_lock:
        return jsonify(dict(_djob))


@app.route("/api/rotation")
def api_rotation():
    """組合輪動回測(需先完成市場探索,使用其下載的價格資料)。"""
    if _scan_store["series"] is None:
        return jsonify({"error": "請先執行「探索市場」,輪動回測使用其下載的歷史資料。"}), 400
    try:
        p = _parse_strategy_params(request.args)
        variant = request.args.get("variant", "trail_stop")
        max_pos = max(1, min(20, int(request.args.get("max_pos", 5))))
        tickers = [t.strip().upper() for t in request.args.get("tickers", "").split(",") if t.strip()]
    except ValueError:
        return jsonify({"error": "參數格式錯誤。"}), 400
    if not tickers:
        tickers = list(_scan_store["series"].keys())
    try:
        out = _sanitize(rotation_run(_scan_store["series"], tickers, p, variant, max_pos))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("rotation failed")
        return jsonify({"error": f"輪動回測失敗:{exc}"}), 500
    return jsonify(out)


@app.route("/api/track/add", methods=["POST"])
def track_add():
    try:
        rid = tracker.add_rec(request.get_json(force=True))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"記錄失敗:{exc}"}), 400
    return jsonify({"ok": True, "id": rid})


@app.route("/api/track/list")
def track_list():
    try:
        return jsonify(_sanitize(tracker.list_recs()))
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("track list failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/track/remove", methods=["POST"])
def track_remove():
    try:
        tracker.remove_rec(int(request.args.get("id", 0)))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("US Stock Monte Carlo Analyzer -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
