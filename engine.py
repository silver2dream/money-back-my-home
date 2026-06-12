# -*- coding: utf-8 -*-
"""
蒙地卡羅股票策略引擎 v2

策略模型:
  在「等待窗口」內掛限價單於 現價 × (1 - 回檔%),觸發後持有,依出場規則出場。
  四種出場規則(一次全算,前端切換顯示):
    fixed       固定目標:收盤 >= 進場價×(1+目標) 以目標價賣出
    fixed_stop  固定目標 + 停損:收盤 <= 進場價×(1-停損) 以收盤價砍出
    trail       移動停利:觸及目標後不賣,改追蹤最高收盤,回落 trail% 以收盤價出場
    trail_stop  移動停利 + 停損(停損僅在達標前有效)

關鍵指標「帳戶年化」:同一資金窗口 T = 等待 + 最長持有,
  未觸發或出場後的閒置資金以現金利率計息,對全部路徑取幾何平均後年化;
  與「單純持有」基準(同窗口)直接可比,等回檔策略的機會成本因此入帳。

驗證三層:
  1. 全期回測(in-sample):同規則在全部歷史上逐段實測
  2. walk-forward:前半資料建模擬 → 後半資料實測(樣本外)
  3. 壓力測試:只從歷史最差一年的報酬抽樣重新模擬
"""
import math
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
BLOCK_SIZE = 21          # bootstrap 區塊長度(約一個月),保留短期自相關
DIP_LEVELS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]
EXIT_KEYS = ("fixed", "fixed_stop", "trail", "trail_stop")
HIST_BINS = 12
BACKTEST_STEP = 5


class EngineError(Exception):
    """帶給使用者看的錯誤訊息。"""


# ---------------------------------------------------------------- 資料層

def fetch_history(ticker: str, years: int = 10):
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 12:
        raise EngineError("請輸入有效的股票代碼(例如 AAPL、NVDA、SPY)。")
    tk = yf.Ticker(ticker)
    try:
        df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True)
    except Exception as exc:
        raise EngineError(f"下載 {ticker} 資料失敗:{exc}") from exc
    if df is None or df.empty or "Close" not in df:
        raise EngineError(f"找不到「{ticker}」的歷史資料,請確認代碼是否正確(美股代碼,例如 AAPL)。")

    closes = df["Close"].dropna()
    closes = closes[closes > 0]
    if len(closes) < TRADING_DAYS:
        raise EngineError(f"「{ticker}」歷史資料不足一年({len(closes)} 個交易日),樣本太少無法可靠模擬。")

    name, currency = ticker, "USD"
    try:
        fi = tk.fast_info
        currency = getattr(fi, "currency", None) or "USD"
    except Exception:
        pass
    try:
        info = tk.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
    except Exception:
        pass

    dates = [d.strftime("%Y-%m-%d") for d in closes.index]
    meta = {"ticker": ticker, "name": name, "currency": currency, "as_of": dates[-1]}
    return closes.to_numpy(dtype=float), dates, meta


def basic_stats(closes: np.ndarray) -> dict:
    log_ret = np.diff(np.log(closes))
    ann_return = math.exp(log_ret.mean() * TRADING_DAYS) - 1
    ann_vol = float(log_ret.std(ddof=1) * math.sqrt(TRADING_DAYS))
    running_max = np.maximum.accumulate(closes)
    max_dd = float((closes / running_max - 1).min())
    return {
        "years": round(len(log_ret) / TRADING_DAYS, 1),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": max_dd,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
    }


# ---------------------------------------------------------------- 模擬

def simulate_paths(log_ret: np.ndarray, spot: float, n_paths: int, horizon: int,
                   seed: int = 42) -> np.ndarray:
    """Block bootstrap。回傳 (n_paths, horizon+1),[:,0] = spot。"""
    rng = np.random.default_rng(seed)
    block = min(BLOCK_SIZE, len(log_ret))
    n_blocks = math.ceil(horizon / block)
    starts = rng.integers(0, len(log_ret) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    cum = np.cumsum(log_ret[idx], axis=1)
    prices = np.empty((n_paths, horizon + 1))
    prices[:, 0] = spot
    prices[:, 1:] = spot * np.exp(cum)
    return prices


# ---------------------------------------------------------------- 出場規則(核心,模擬與回測共用)

def exit_outcomes(rel: np.ndarray, target: float, stop: float, trail: float) -> dict:
    """rel: (n, max_hold+1) 進場後相對價格序列(成本 = 1,rel[:,0] 為進場日收盤)。
    回傳 {variant: (ret, days, hit)};hit = 曾觸及目標價。
    達標以目標價(限價)成交;停損/移動停利以當日收盤成交(含跳空,偏保守)。"""
    n, W = rel.shape
    cols = np.arange(W)[None, :]
    rows = np.arange(n)
    BIG = W + 9
    end_day = W - 1

    hit_mask = (rel >= 1.0 + target) & (cols >= 1)
    j_hit = np.where(hit_mask.any(1), hit_mask.argmax(1), BIG)

    stop_mask = (rel <= 1.0 - stop) & (cols >= 1) & (cols < np.minimum(j_hit, BIG)[:, None])
    j_stop = np.where(stop_mask.any(1), stop_mask.argmax(1), BIG)

    # 移動停利:達標日起追蹤最高收盤,回落 trail% 出場
    after = cols >= np.clip(j_hit, 0, W)[:, None]      # j_hit=BIG → 全 False
    runmax = np.maximum.accumulate(np.where(after, rel, -np.inf), axis=1)
    tr_mask = (rel <= runmax * (1.0 - trail)) & (cols > np.clip(j_hit, 0, W)[:, None])
    j_tr = np.where(tr_mask.any(1), tr_mask.argmax(1), BIG)

    def pack(j_exit, target_fill_mask, hit):
        days = np.where(j_exit < BIG, j_exit, end_day)
        ret = rel[rows, days] - 1.0
        if target_fill_mask is not None:
            ret = np.where(target_fill_mask, target, ret)
        return ret.astype(float), days.astype(float), hit

    out = {}
    hit_f = j_hit < BIG
    out["fixed"] = pack(j_hit, hit_f, hit_f)

    hit_fs = hit_f & (j_hit < j_stop)
    out["fixed_stop"] = pack(np.minimum(j_hit, j_stop), hit_fs, hit_fs)

    j_e = np.where(hit_f, np.where(j_tr < BIG, j_tr, end_day), BIG)
    out["trail"] = pack(j_e, None, hit_f)

    stopped = j_stop < j_hit
    hit_ts = hit_f & ~stopped
    j_e = np.where(stopped, j_stop, np.where(hit_ts & (j_tr < BIG), j_tr, BIG))
    out["trail_stop"] = pack(j_e, None, hit_ts)
    return out


# ---------------------------------------------------------------- 統計彙整

def _hist(days_hit: np.ndarray, edges: np.ndarray):
    if len(days_hit) == 0:
        return [0] * (len(edges) - 1)
    counts, _ = np.histogram(days_hit, bins=edges)
    return [int(c) for c in counts]


def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None


def variant_stats(ret: np.ndarray, days: np.ndarray, hit: np.ndarray,
                  n_total: int, p: dict, edges: np.ndarray) -> dict:
    """單一(進場點位 × 出場規則)的統計。ret/days/hit 為「已觸發」路徑;
    n_total 為全部路徑數(含未觸發,其資金全程停泊現金)。"""
    n_trig = len(ret)
    T = p["wait"] + p["max_hold"]
    cash_d = (1.0 + p["cash_rate"]) ** (1.0 / TRADING_DAYS) - 1.0

    # 帳戶層級:同窗口 T,閒置時間計現金利息,幾何平均後年化
    r_cash_full = (1.0 + cash_d) ** T - 1.0
    if n_trig:
        r_acct = (1.0 + cash_d) ** (T - days) * (1.0 + ret) - 1.0
        log_sum = np.log1p(r_acct).sum() + math.log1p(r_cash_full) * (n_total - n_trig)
    else:
        log_sum = math.log1p(r_cash_full) * n_total
    account_ann = math.exp(log_sum / n_total * TRADING_DAYS / T) - 1.0

    if n_trig == 0:
        return {
            "hit_prob_entry": 0.0, "hit_prob_overall": 0.0, "loss_prob": None,
            "avg_return": None, "ret_p10": None, "days_p50": None, "days_p90": None,
            "avg_days": None, "account_ann": account_ann, "turn_ann": None,
            "kelly": 0.0, "hist_counts": _hist(np.array([]), edges),
        }

    days_hit = days[hit]
    avg_days = float(days.mean())
    avg_return = float(ret.mean())
    # 完美再部署年化(出場後立即有下一筆同品質機會的理論上限)
    turn_ann = avg_return / avg_days * TRADING_DAYS if avg_days > 0 else None
    # ¼ Kelly:單次交易超額報酬 / 變異數 / 4
    cash_period = (1.0 + cash_d) ** avg_days - 1.0
    var = float(ret.var())
    kelly = max(0.0, min(1.0, (avg_return - cash_period) / var / 4.0)) if var > 1e-12 else 0.0

    return {
        "hit_prob_entry": float(hit.mean()),
        "hit_prob_overall": float(hit.mean()) * n_trig / n_total,
        "loss_prob": float((ret < 0).mean()),
        "avg_return": avg_return,
        "ret_p10": _pct(ret, 10),
        "days_p50": _pct(days_hit, 50),
        "days_p90": _pct(days_hit, 90),
        "avg_days": avg_days,
        "account_ann": account_ann,
        "turn_ann": turn_ann,
        "kelly": kelly,
        "hist_counts": _hist(days_hit, edges),
    }


def evaluate_entry(prices: np.ndarray, spot: float, dip: float,
                   p: dict, edges: np.ndarray) -> dict:
    """單一進場點位 × 全部出場規則(模擬路徑)。"""
    n_paths, L = prices.shape
    wait, max_hold = p["wait"], p["max_hold"]
    entry_price = spot * (1.0 - dip)

    if dip <= 0:
        trig_rows = np.arange(n_paths)
        t_entry = np.zeros(n_paths, dtype=int)
    else:
        in_wait = prices[:, : wait + 1] <= entry_price
        trig_mask = in_wait.any(axis=1)
        trig_rows = np.where(trig_mask)[0]
        t_entry = in_wait.argmax(axis=1)[trig_rows]

    n_trig = len(trig_rows)
    variants = {}
    if n_trig:
        offsets = t_entry[:, None] + np.arange(max_hold + 1)[None, :]
        aligned = prices[trig_rows[:, None], offsets] / entry_price
        outcomes = exit_outcomes(aligned, p["target"], p["stop"], p["trail"])
        for k in EXIT_KEYS:
            ret, days, hit = outcomes[k]
            variants[k] = variant_stats(ret, days, hit, n_paths, p, edges)
    else:
        empty = np.array([])
        for k in EXIT_KEYS:
            variants[k] = variant_stats(empty, empty, np.array([], bool), n_paths, p, edges)

    return {
        "dip": dip,
        "entry_price": entry_price,
        "target_price": entry_price * (1.0 + p["target"]),
        "stop_price": entry_price * (1.0 - p["stop"]),
        "trigger_prob": n_trig / n_paths,
        "variants": variants,
    }


# ---------------------------------------------------------------- 基準

def benchmark_stats(prices: np.ndarray, spot: float, p: dict) -> dict:
    """單純持有(同窗口 T)的模擬分佈。"""
    T = p["wait"] + p["max_hold"]
    r = prices[:, min(T, prices.shape[1] - 1)] / spot - 1.0
    ann = math.exp(float(np.log1p(r).mean()) * TRADING_DAYS / T) - 1.0
    return {
        "ann": ann,
        "total_p50": _pct(r, 50), "total_p10": _pct(r, 10), "total_p90": _pct(r, 90),
        "loss_prob": float((r < 0).mean()),
        "window_days": T,
    }


# ---------------------------------------------------------------- 歷史回測(全期與 walk-forward 共用)

def backtest_full(closes: np.ndarray, p: dict, edges: np.ndarray) -> dict:
    """同規則在真實序列上逐段實測;只統計窗口完整的起點,避免存活偏差。"""
    M = len(closes)
    wait, max_hold = p["wait"], p["max_hold"]
    window = wait + max_hold
    last_start = M - 1 - window
    if last_start < 1:
        return {"available": False, "reason": "歷史資料長度不足以完成完整窗口回測。"}

    rows = []
    for dip in DIP_LEVELS:
        starts = list(range(0, last_start + 1, BACKTEST_STEP))
        te_list, rel_rows = [], []
        for t in starts:
            entry_price = closes[t] * (1.0 - dip)
            if dip <= 0:
                te = t
            else:
                seg = closes[t + 1: t + wait + 1]
                idx = np.nonzero(seg <= entry_price)[0]
                if len(idx) == 0:
                    continue
                te = t + 1 + int(idx[0])
            te_list.append(te)
            rel_rows.append(closes[te: te + max_hold + 1] / entry_price)

        n_starts, n_trig = len(starts), len(rel_rows)
        row = {"dip": dip, "samples": n_starts,
               "trigger_rate": n_trig / n_starts if n_starts else 0.0, "variants": {}}
        if n_trig:
            outcomes = exit_outcomes(np.vstack(rel_rows), p["target"], p["stop"], p["trail"])
            for k in EXIT_KEYS:
                ret, days, hit = outcomes[k]
                hr = float(hit.mean())
                row["variants"][k] = {
                    "hit_rate": hr,
                    "hit_overall": hr * n_trig / n_starts,
                    "days_p50": _pct(days[hit], 50),
                    "avg_return": float(ret.mean()),
                }
        else:
            for k in EXIT_KEYS:
                row["variants"][k] = {"hit_rate": None, "hit_overall": 0.0,
                                      "days_p50": None, "avg_return": None}
        rows.append(row)
    return {"available": True, "rows": rows}


def _mad(sim_entries: list, real_rows: list) -> dict:
    """各出場規則下,模擬與實測「整體達標率」的平均絕對偏差。"""
    out = {}
    for k in EXIT_KEYS:
        diffs = [abs(e["variants"][k]["hit_prob_overall"] - r["variants"][k]["hit_overall"])
                 for e, r in zip(sim_entries, real_rows)]
        out[k] = float(np.mean(diffs)) if diffs else None
    return out


def walk_forward(closes: np.ndarray, dates: list, p: dict,
                 n_paths: int, edges: np.ndarray) -> dict:
    """樣本外驗證:前半資料建立模擬 → 後半資料實測。"""
    M = len(closes)
    split = M // 2
    window = p["wait"] + p["max_hold"]
    if split < TRADING_DAYS * 2 or (M - split) < window + 10 * BACKTEST_STEP:
        return {"available": False,
                "reason": "資料長度不足(訓練段需 ≥2 年、驗證段需容納完整窗口),略過樣本外驗證。"}

    train_ret = np.diff(np.log(closes[:split]))
    sim = simulate_paths(train_ret, 1.0, n_paths, window, seed=7)
    sim_entries = [evaluate_entry(sim, 1.0, dip, p, edges) for dip in DIP_LEVELS]
    bt = backtest_full(closes[split:], p, edges)
    if not bt["available"]:
        return {"available": False, "reason": "驗證段長度不足。"}

    rows = []
    for e, r in zip(sim_entries, bt["rows"]):
        rows.append({
            "dip": e["dip"],
            "trigger_sim": e["trigger_prob"], "trigger_real": r["trigger_rate"],
            "samples": r["samples"],
            "variants": {k: {
                "hit_sim": e["variants"][k]["hit_prob_entry"],
                "hit_real": r["variants"][k]["hit_rate"],
                "overall_sim": e["variants"][k]["hit_prob_overall"],
                "overall_real": r["variants"][k]["hit_overall"],
                "days_sim": e["variants"][k]["days_p50"],
                "days_real": r["variants"][k]["days_p50"],
            } for k in EXIT_KEYS},
        })
    return {"available": True, "split_date": dates[split],
            "train_from": dates[0], "test_to": dates[-1],
            "rows": rows, "mad": _mad(sim_entries, bt["rows"])}


# ---------------------------------------------------------------- 壓力測試

def stress_test(closes: np.ndarray, dates: list, spot: float, p: dict,
                n_paths: int, edges: np.ndarray) -> dict:
    """只從歷史最差一年(滾動 252 日報酬最低的窗口)抽樣,重新模擬。"""
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < TRADING_DAYS * 2:
        return {"available": False, "reason": "資料不足兩年,無法切出壓力子段。"}
    cs = np.concatenate([[0.0], np.cumsum(log_ret)])
    roll = cs[TRADING_DAYS:] - cs[:-TRADING_DAYS]
    i0 = int(roll.argmin())
    sub = log_ret[i0: i0 + TRADING_DAYS]

    window = p["wait"] + p["max_hold"]
    sim = simulate_paths(sub, 1.0, n_paths, window, seed=11)
    rows = []
    for dip in DIP_LEVELS:
        e = evaluate_entry(sim, 1.0, dip, p, edges)
        rows.append({"dip": dip, "trigger_prob": e["trigger_prob"],
                     "variants": {k: {
                         "hit_overall": e["variants"][k]["hit_prob_overall"],
                         "account_ann": e["variants"][k]["account_ann"],
                     } for k in EXIT_KEYS}})
    bh = benchmark_stats(sim, 1.0, p)
    return {"available": True,
            "from": dates[i0 + 1], "to": dates[min(i0 + TRADING_DAYS, len(dates) - 1)],
            "worst_year_return": math.exp(float(roll[i0])) - 1.0,
            "bh_ann": bh["ann"], "rows": rows}


# ---------------------------------------------------------------- 主流程

def analyze(ticker: str, target: float, max_hold: int = 252, wait: int = 63,
            n_paths: int = 5000, years: int = 10, stop: float = 0.15,
            trail: float = 0.08, cash_rate: float = 0.04) -> dict:
    if not (0.005 <= target <= 5.0):
        raise EngineError("目標利潤需介於 0.5% 與 500% 之間。")
    if not (10 <= max_hold <= 1260):
        raise EngineError("最長持有天數需介於 10 與 1260 個交易日之間。")
    p = {
        "target": target,
        "max_hold": int(max_hold),
        "wait": int(np.clip(wait, 5, 252)),
        "stop": float(np.clip(stop, 0.02, 0.50)),
        "trail": float(np.clip(trail, 0.02, 0.50)),
        "cash_rate": float(np.clip(cash_rate, 0.0, 0.10)),
    }
    n_paths = int(np.clip(n_paths, 500, 20000))

    closes, dates, meta = fetch_history(ticker, years)
    spot = float(closes[-1])
    stats = basic_stats(closes)
    log_ret = np.diff(np.log(closes))
    edges = np.linspace(0, p["max_hold"], HIST_BINS + 1)

    horizon = p["wait"] + p["max_hold"]
    prices = simulate_paths(log_ret, spot, n_paths, horizon)

    entries = [evaluate_entry(prices, spot, dip, p, edges) for dip in DIP_LEVELS]
    bench = benchmark_stats(prices, spot, p)

    bt = backtest_full(closes, p, edges)
    if bt["available"]:
        bt["mad"] = _mad(entries, bt["rows"])
    wf = walk_forward(closes, dates, p, n_paths, edges)
    stress = stress_test(closes, dates, spot, p, n_paths, edges)

    # 各出場規則的推薦進場點位 = 帳戶年化最高者;全域最佳組合另計
    reco = {k: int(max(range(len(entries)),
                       key=lambda i: entries[i]["variants"][k]["account_ann"]))
            for k in EXIT_KEYS}
    best_k, best_i = max(((k, reco[k]) for k in EXIT_KEYS),
                         key=lambda ki: entries[ki[1]]["variants"][ki[0]]["account_ann"])
    best_ann = entries[best_i]["variants"][best_k]["account_ann"]

    tail = min(len(closes), TRADING_DAYS * 2)
    return {
        **meta,
        "current_price": spot,
        "stats": stats,
        "params": {**p, "n_paths": n_paths, "years_used": stats["years"]},
        "history": {"dates": dates[-tail:], "prices": closes[-tail:].tolist()},
        "fan": _fan_chart(prices),
        "hist_edges": [float(b) for b in edges],
        "benchmark": bench,
        "entries": entries,
        "reco": reco,
        "best": {"variant": best_k, "idx": best_i, "account_ann": best_ann,
                 "beats_benchmark": bool(best_ann > bench["ann"])},
        "backtest": bt,
        "walk_forward": wf,
        "stress": stress,
    }


def _fan_chart(prices: np.ndarray) -> dict:
    qs = np.percentile(prices, [5, 25, 50, 75, 95], axis=0)
    return {
        "days": list(range(prices.shape[1])),
        "p5": qs[0].tolist(), "p25": qs[1].tolist(), "p50": qs[2].tolist(),
        "p75": qs[3].tolist(), "p95": qs[4].tolist(),
    }


if __name__ == "__main__":
    import json, sys, time
    t0 = time.time()
    r = analyze(sys.argv[1] if len(sys.argv) > 1 else "AAPL",
                float(sys.argv[2]) / 100 if len(sys.argv) > 2 else 0.15)
    print(f"== {r['ticker']} spot={r['current_price']:.2f} "
          f"bh_sim_ann={r['benchmark']['ann']*100:.1f}% hist_ann={r['stats']['ann_return']*100:.1f}%")
    for e in r["entries"]:
        line = f"dip={e['dip']*100:>4.0f}% trig={e['trigger_prob']*100:5.1f}% | "
        line += " | ".join(
            f"{k}: hit={v['hit_prob_entry']*100 if v['hit_prob_entry'] else 0:5.1f}% "
            f"acct={v['account_ann']*100:6.2f}% kelly={v['kelly']*100:3.0f}%"
            for k, v in e["variants"].items())
        print(line)
    print("reco:", r["reco"], "best:", r["best"])
    if r["walk_forward"]["available"]:
        print("walk-forward mad:", {k: f"{v*100:.1f}pp" for k, v in r["walk_forward"]["mad"].items()})
    if r["stress"]["available"]:
        s = r["stress"]
        print(f"stress {s['from']}~{s['to']} ({s['worst_year_return']*100:.0f}%) bh_ann={s['bh_ann']*100:.1f}%")
        for row in s["rows"]:
            print("  dip", row["dip"], {k: f"{v['account_ann']*100:.1f}%" for k, v in row["variants"].items()})
    print(f"-- elapsed {time.time()-t0:.1f}s", file=sys.stderr)
