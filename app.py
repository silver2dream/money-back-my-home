# -*- coding: utf-8 -*-
"""美股蒙地卡羅策略分析 — Flask 伺服器"""
import math
import sys
import time
import threading

from flask import Flask, jsonify, request, send_from_directory

from engine import EngineError, analyze

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__, static_folder="static")

CACHE_TTL = 900  # 同參數結果快取 15 分鐘,避免重複下載資料
_cache: dict = {}
_lock = threading.Lock()


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
    except ValueError:
        return jsonify({"error": "參數格式錯誤,請檢查輸入值。"}), 400

    key = (ticker, round(target, 4), max_hold, wait, n_paths, years,
           round(stop, 4), round(trail, 4), round(cash, 4))
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return jsonify(hit[1])

    try:
        result = _sanitize(analyze(ticker, target, max_hold, wait, n_paths, years,
                                   stop, trail, cash))
    except EngineError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("analyze failed")
        return jsonify({"error": f"分析失敗:{exc}"}), 500

    with _lock:
        _cache[key] = (time.time(), result)
    return jsonify(result)


if __name__ == "__main__":
    print("US Stock Monte Carlo Analyzer -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
