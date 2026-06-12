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
    """歷史資料(經 datasource 層:DB 快取 → Tiingo → yfinance → Stooq)。"""
    from datasource import DataSourceError, get_history
    try:
        closes, dates, meta = get_history(ticker, years)
    except DataSourceError as exc:
        raise EngineError(str(exc)) from exc
    if len(closes) < TRADING_DAYS:
        raise EngineError(f"「{meta.get('ticker', ticker)}」歷史資料不足一年"
                          f"({len(closes)} 個交易日),樣本太少無法可靠模擬。")
    return closes, dates, meta


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

COND_BLOCKS = 3   # 條件化只作用於前 3 個區塊(約一季)— 狀態資訊的可預測性隨時間衰減


def simulate_paths(log_ret: np.ndarray, spot: float, n_paths: int, horizon: int,
                   seed: int = 42, start_pool: np.ndarray = None,
                   fat_tails: bool = False) -> np.ndarray:
    """Block bootstrap。回傳 (n_paths, horizon+1),[:,0] = spot。

    start_pool:條件化 — 前 COND_BLOCKS 個區塊(約一季)的起點只從這些歷史索引抽,
                之後回歸無條件(避免把當前市場狀態鎖死整個模擬窗口);None = 全程無條件。
    fat_tails:疊加小幅 t(4) 噪音(scale = 15% 日波動)— 平滑分佈並加厚尾部,
               讓模擬能產生略超出歷史極值的情境(smoothed bootstrap)。"""
    rng = np.random.default_rng(seed)
    block = min(BLOCK_SIZE, len(log_ret))
    n_blocks = math.ceil(horizon / block)
    max_start = len(log_ret) - block
    pool = None
    if start_pool is not None:
        pool = start_pool[start_pool <= max_start]
        if len(pool) < 50:          # 條件樣本過少 → 退回無條件
            pool = None
    starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks))
    if pool is not None:
        nc = min(COND_BLOCKS, n_blocks)
        starts[:, :nc] = rng.choice(pool, size=(n_paths, nc))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    sim = log_ret[idx]
    if fat_tails:
        sim = sim + rng.standard_t(4, size=sim.shape) * (log_ret.std() * 0.15)
    cum = np.cumsum(sim, axis=1)
    prices = np.empty((n_paths, horizon + 1))
    prices[:, 0] = spot
    prices[:, 1:] = spot * np.exp(cum)
    return prices


# ---------------------------------------------------------------- 市場狀態(條件化模擬)

def regime_state(closes: np.ndarray) -> dict:
    """當前市場狀態:趨勢(現價 vs 200 日均線)× 波動(近 21 日年化波動 vs 全期滾動中位)。"""
    log_ret = np.diff(np.log(closes))
    ma_win = min(200, len(closes))
    ma200 = float(closes[-ma_win:].mean())
    vol21 = float(log_ret[-21:].std(ddof=0) * math.sqrt(TRADING_DAYS))
    roll = _rolling_vol(log_ret, 21)
    vol_med = float(np.nanmedian(roll))
    return {
        "trend": "bull" if closes[-1] > ma200 else "bear",
        "vol": "high" if vol21 > vol_med else "low",
        "ma200": ma200, "vol21": vol21, "vol_median": vol_med,
    }


def _rolling_vol(log_ret: np.ndarray, w: int) -> np.ndarray:
    """滾動 w 日年化波動(索引 t = 截至 log_ret[t] 的窗口);前 w-1 為 NaN。"""
    n = len(log_ret)
    out = np.full(n, np.nan)
    if n < w:
        return out
    c1 = np.concatenate([[0.0], np.cumsum(log_ret)])
    c2 = np.concatenate([[0.0], np.cumsum(log_ret ** 2)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
    out[w - 1:] = np.sqrt(var) * math.sqrt(TRADING_DAYS)
    return out


def _state_series(closes: np.ndarray):
    """逐日狀態序列(log_ret 座標 t = 截至 closes[t+1] 收盤,無前視):
    回傳 (bull bool 陣列, 滾動 21 日年化波動陣列)。"""
    log_ret = np.diff(np.log(closes))
    n = len(log_ret)
    cs = np.concatenate([[0.0], np.cumsum(closes)])
    t_arr = np.arange(n)
    px_idx = t_arr + 1
    win = np.minimum(px_idx + 1, 200)
    ma = (cs[px_idx + 1] - cs[px_idx + 1 - win]) / win
    bull = closes[px_idx] > ma
    vol = _rolling_vol(log_ret, 21)
    return bull, vol


def _pool_core(closes: np.ndarray, want_bull: bool, want_high: bool, vol_med: float):
    """指定狀態的歷史起點索引;不足 200 日逐步放寬(先棄趨勢,再棄全部)。"""
    bull, vol = _state_series(closes)
    valid = ~np.isnan(vol)
    t_arr = np.arange(len(bull))
    high = vol > vol_med
    pool = t_arr[valid & (bull == want_bull) & (high == want_high)]
    if len(pool) >= 200:
        return pool, ""
    pool = t_arr[valid & (high == want_high)]
    if len(pool) >= 200:
        return pool, "趨勢條件樣本不足,僅以波動狀態條件化"
    return None, "條件樣本不足,改用全部歷史(無條件)"


def regime_pool(closes: np.ndarray, state: dict):
    """當前狀態的歷史樣本池。回傳 (pool | None, sample_days, note)。"""
    pool, note = _pool_core(closes, state["trend"] == "bull",
                            state["vol"] == "high", state["vol_median"])
    return pool, (int(len(pool)) if pool is not None else 0), note


# ---------------------------------------------------------------- 估計不確定性(二階 bootstrap)

def uncertainty_bands(log_ret: np.ndarray, spot: float, p: dict, edges: np.ndarray,
                      fat_tails: bool, B: int = 12, n_paths: int = 800) -> dict:
    """二階 bootstrap:以大區塊(63 日)重抽「替代歷史」B 次,每個替代歷史重跑模擬,
    得到各(進場 × 出場)帳戶年化與持有基準年化的「相對中位數的 16/84 偏移量」。
    偏移量套在點估計上即為 68% 區間 — 反映 drift / 波動估計本身的抽樣不確定性,
    這是任何依賴十年歷史的點估計天生的模糊度。"""
    n = len(log_ret)
    T = p["wait"] + p["max_hold"]
    rng = np.random.default_rng(99)
    blk = min(63, n)
    n_blk = math.ceil(n / blk)
    acc = {k: [[] for _ in DIP_LEVELS] for k in EXIT_KEYS}
    bench_acc = []
    for b in range(B):
        starts = rng.integers(0, n - blk + 1, size=n_blk)
        alt = np.concatenate([log_ret[s: s + blk] for s in starts])[:n]
        prices = simulate_paths(alt, spot, n_paths, T, seed=1000 + b, fat_tails=fat_tails)
        for i, dip in enumerate(DIP_LEVELS):
            e = evaluate_entry(prices, spot, dip, p, edges)
            for k in EXIT_KEYS:
                acc[k][i].append(e["variants"][k]["account_ann"])
        bench_acc.append(benchmark_stats(prices, spot, p)["ann"])

    def offsets(vals):
        lo, med, hi = np.percentile(vals, [16, 50, 84])
        return [float(lo - med), float(hi - med)]

    return {
        "entries": {k: [offsets(acc[k][i]) for i in range(len(DIP_LEVELS))] for k in EXIT_KEYS},
        "bench": offsets(bench_acc),
        "B": B,
    }


# ---------------------------------------------------------------- 資料健檢

def data_health(dates: list, closes: np.ndarray) -> list:
    """輕量資料品質檢查,回傳警告字串清單。"""
    import datetime as _dt
    warns = []
    last = _dt.date.fromisoformat(dates[-1])
    if (_dt.date.today() - last).days > 7:
        warns.append(f"資料最後日期為 {dates[-1]},距今超過 7 天,可能未更新或已下市。")
    log_ret = np.diff(np.log(closes))
    big = np.where(np.abs(log_ret) > 0.40)[0]
    if len(big):
        days = ", ".join(dates[i + 1] for i in big[:3])
        warns.append(f"偵測到 {len(big)} 個單日 |漲跌| > 40% 的極端跳動({days}…),"
                     "可能為分割/資料異常,結果請謹慎解讀。")
    d = [_dt.date.fromisoformat(x) for x in dates]
    gaps = sum(1 for a, b in zip(d[:-1], d[1:]) if (b - a).days > 12)
    if gaps:
        warns.append(f"歷史資料存在 {gaps} 段超過 12 個日曆天的缺口(停牌或資料缺漏)。")
    return warns


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

def backtest_full(closes: np.ndarray, p: dict, edges: np.ndarray,
                  step: int = BACKTEST_STEP) -> dict:
    """同規則在真實序列上逐段實測;只統計窗口完整的起點,避免存活偏差。"""
    M = len(closes)
    wait, max_hold = p["wait"], p["max_hold"]
    window = wait + max_hold
    last_start = M - 1 - window
    if last_start < 1:
        return {"available": False, "reason": "歷史資料長度不足以完成完整窗口回測。"}

    rows = []
    for dip in DIP_LEVELS:
        starts = list(range(0, last_start + 1, step))
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
    """樣本外驗證:只用前半資料建立模擬 → 後半資料逐段實測。

    條件化模式下的正確做法:預先計算四種市場狀態(牛/熊 × 高/低波動)各自的
    條件預測表,test 段每個起點依「該起點當下的狀態」(無前視)查表,
    再以狀態出現頻率加權 — 驗證的是「實際使用時」工具給出的預測。"""
    M = len(closes)
    split = M // 2
    window = p["wait"] + p["max_hold"]
    if split < TRADING_DAYS * 2 or (M - split) < window + 10 * BACKTEST_STEP:
        return {"available": False,
                "reason": "資料長度不足(訓練段需 ≥2 年、驗證段需容納完整窗口),略過樣本外驗證。"}

    train = closes[:split]
    train_ret = np.diff(np.log(train))
    fat = p.get("fat_tails", False)
    bt = backtest_full(closes[split:], p, edges)
    if not bt["available"]:
        return {"available": False, "reason": "驗證段長度不足。"}

    conditional = bool(p.get("conditional"))
    if conditional:
        vol_med = regime_state(train)["vol_median"]
        sim_n = int(np.clip(n_paths // 2, 1000, 2500))
        preds = {}
        for tb in (True, False):
            for hv in (True, False):
                pool, _ = _pool_core(train, tb, hv, vol_med)
                sim = simulate_paths(train_ret, 1.0, sim_n, window, seed=7,
                                     start_pool=pool, fat_tails=fat)
                preds[(tb, hv)] = [evaluate_entry(sim, 1.0, dip, p, edges)
                                   for dip in DIP_LEVELS]
        # test 段每個回測起點的狀態(截至起點收盤,無前視)
        bull_all, vol_all = _state_series(closes)
        last_start = (M - split) - 1 - window
        from collections import Counter
        cnt = Counter()
        for t_local in range(0, last_start + 1, BACKTEST_STEP):
            li = max(split + t_local - 1, 21)
            cnt[(bool(bull_all[li]), bool(vol_all[li] > vol_med))] += 1

        def wavg(getter):
            num = den = 0.0
            for key, c in cnt.items():
                v = getter(preds[key])
                if v is None:
                    continue
                num += v * c
                den += c
            return num / den if den else None

        rows, mad = [], {k: [] for k in EXIT_KEYS}
        for di, dip in enumerate(DIP_LEVELS):
            r = bt["rows"][di]
            row = {"dip": dip, "samples": r["samples"],
                   "trigger_sim": wavg(lambda pr: pr[di]["trigger_prob"]),
                   "trigger_real": r["trigger_rate"], "variants": {}}
            for k in EXIT_KEYS:
                overall_sim = wavg(lambda pr: pr[di]["trigger_prob"]
                                   * pr[di]["variants"][k]["hit_prob_entry"])
                row["variants"][k] = {
                    "hit_sim": wavg(lambda pr: pr[di]["variants"][k]["hit_prob_entry"]),
                    "hit_real": r["variants"][k]["hit_rate"],
                    "overall_sim": overall_sim,
                    "overall_real": r["variants"][k]["hit_overall"],
                    "days_sim": wavg(lambda pr: pr[di]["variants"][k]["days_p50"]),
                    "days_real": r["variants"][k]["days_p50"],
                }
                if overall_sim is not None:
                    mad[k].append(abs(overall_sim - r["variants"][k]["hit_overall"]))
        return {"available": True, "split_date": dates[split],
                "train_from": dates[0], "test_to": dates[-1], "conditional": True,
                "state_mix": {f"{'bull' if tb else 'bear'}/{'high' if hv else 'low'}": c
                              for (tb, hv), c in cnt.items()},
                "rows": rows,
                "mad": {k: (float(np.mean(v)) if v else None) for k, v in mad.items()}}

    # 無條件版
    sim = simulate_paths(train_ret, 1.0, n_paths, window, seed=7, fat_tails=fat)
    sim_entries = [evaluate_entry(sim, 1.0, dip, p, edges) for dip in DIP_LEVELS]
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
            "train_from": dates[0], "test_to": dates[-1], "conditional": False,
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
    sim = simulate_paths(sub, 1.0, n_paths, window, seed=11,
                         fat_tails=p.get("fat_tails", False))
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
            trail: float = 0.08, cash_rate: float = 0.04,
            conditional: bool = True, fat_tails: bool = True,
            div_tax: float = 0.30) -> dict:
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
        "conditional": bool(conditional),
        "fat_tails": bool(fat_tails),
        "div_tax": float(np.clip(div_tax, 0.0, 0.50)),
    }
    n_paths = int(np.clip(n_paths, 500, 20000))

    closes, dates, meta = fetch_history(ticker, years)
    spot = float(closes[-1])
    stats = basic_stats(closes)
    warnings = data_health(dates, closes)
    edges = np.linspace(0, p["max_hold"], HIST_BINS + 1)

    # 股息預扣稅:前瞻模擬的報酬扣除 yield × 稅率(調整後價格已含全額股息再投資)
    log_ret = np.diff(np.log(closes))
    div_drag = 0.0
    if p["div_tax"] > 0 and meta.get("div_yield", 0) > 0:
        div_drag = math.log1p(-meta["div_yield"] * p["div_tax"]) / TRADING_DAYS
        log_ret = log_ret + div_drag

    # 市場狀態與條件化模擬
    state = regime_state(closes)
    pool, pool_days, pool_note = (None, 0, "")
    if p["conditional"]:
        pool, pool_days, pool_note = regime_pool(closes, state)
    regime = {"enabled": p["conditional"], "trend": state["trend"], "vol": state["vol"],
              "sample_days": pool_days, "note": pool_note,
              "active": bool(p["conditional"] and pool is not None)}

    horizon = p["wait"] + p["max_hold"]
    prices = simulate_paths(log_ret, spot, n_paths, horizon,
                            start_pool=pool, fat_tails=p["fat_tails"])
    entries = [evaluate_entry(prices, spot, dip, p, edges) for dip in DIP_LEVELS]
    bench = benchmark_stats(prices, spot, p)

    # in-sample 驗證需與「全期、無條件」的模擬對照才語義對等
    if regime["active"]:
        prices_u = simulate_paths(log_ret, spot, min(n_paths, 3000), horizon,
                                  fat_tails=p["fat_tails"])
        entries_u = [evaluate_entry(prices_u, spot, dip, p, edges) for dip in DIP_LEVELS]
    else:
        entries_u = entries

    bt = backtest_full(closes, p, edges)
    if bt["available"]:
        bt["mad"] = _mad(entries_u, bt["rows"])
        for e_u, row in zip(entries_u, bt["rows"]):     # 對照表的「模擬」欄(無條件)
            row["sim"] = {"trigger_prob": e_u["trigger_prob"],
                          "variants": {k: {"hit": e_u["variants"][k]["hit_prob_entry"],
                                           "days_p50": e_u["variants"][k]["days_p50"]}
                                       for k in EXIT_KEYS}}
    wf = walk_forward(closes, dates, p, n_paths, edges)
    stress = stress_test(closes, dates, spot, p, n_paths, edges)

    # 估計不確定性(68% 區間)— 偏移量套在點估計上,中心對齊、寬度誠實
    bands = uncertainty_bands(log_ret, spot, p, edges, p["fat_tails"])
    for i in range(len(entries)):
        for k in EXIT_KEYS:
            lo_off, hi_off = bands["entries"][k][i]
            v = entries[i]["variants"][k]
            v["ann_lo"] = v["account_ann"] + lo_off
            v["ann_hi"] = v["account_ann"] + hi_off
    bench["ann_lo"] = bench["ann"] + bands["bench"][0]
    bench["ann_hi"] = bench["ann"] + bands["bench"][1]

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
        "warnings": warnings,
        "regime": regime,
        "div_drag_ann": (math.exp(div_drag * TRADING_DAYS) - 1) if div_drag else 0.0,
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


# ---------------------------------------------------------------- 市場探索(粗掃)

def quick_one(closes: np.ndarray, ticker: str, name: str, p: dict,
              n_paths: int = 2000) -> dict:
    """單檔輕量分析:主模擬 + 全組合評比 + in-sample 偏差,供市場掃描排序分類。
    不含 walk-forward / 壓力測試(由前端對入圍標的呼叫完整分析補上)。"""
    spot = float(closes[-1])
    log_ret = np.diff(np.log(closes))
    edges = np.linspace(0, p["max_hold"], HIST_BINS + 1)
    T = p["wait"] + p["max_hold"]
    pool = None
    if p.get("conditional"):
        pool, _, _ = regime_pool(closes, regime_state(closes))
    prices = simulate_paths(log_ret, spot, n_paths, T, start_pool=pool,
                            fat_tails=p.get("fat_tails", False))

    entries = [evaluate_entry(prices, spot, dip, p, edges) for dip in DIP_LEVELS]
    bench = benchmark_stats(prices, spot, p)

    # 單純持有的 ¼ Kelly(供「適合持有」標的的倉位建議)
    r_bh = prices[:, T] / spot - 1.0
    cash_t = (1.0 + p["cash_rate"]) ** (T / TRADING_DAYS) - 1.0
    var = float(r_bh.var())
    kelly_bh = max(0.0, min(1.0, (float(r_bh.mean()) - cash_t) / var / 4.0)) if var > 1e-12 else 0.0

    best_k, best_i = max(((k, i) for k in EXIT_KEYS for i in range(len(entries))),
                         key=lambda ki: entries[ki[1]]["variants"][ki[0]]["account_ann"])
    e, v = entries[best_i], entries[best_i]["variants"][best_k]
    excess = v["account_ann"] - bench["ann"]

    bt = backtest_full(closes, p, edges, step=10)
    bt_mad = _mad(entries, bt["rows"])[best_k] if bt["available"] else None

    # 擇時優勢必須同時:贏過持有 0.5pp 以上、且帳戶年化至少比現金高 2pp —
    # 否則「贏過很爛的持有」只是五十步笑百步,對投資人正確的建議是留現金。
    cash_floor = p["cash_rate"] + 0.02
    if excess > 0.005 and v["account_ann"] >= cash_floor:
        category = "timing"
    elif bench["ann"] >= 0.08:         # 擇時無優勢但長期報酬夠好 → 持有
        category = "hold"
    else:
        category = "avoid"

    stats = basic_stats(closes)
    return {
        "ticker": ticker, "name": name,
        "current_price": spot,
        "ann_return": stats["ann_return"], "ann_vol": stats["ann_vol"],
        "years": stats["years"],
        "benchmark": {"ann": bench["ann"], "total_p10": bench["total_p10"],
                      "kelly_bh": kelly_bh},
        "best": {"variant": best_k, "idx": best_i, "dip": e["dip"],
                 "entry_price": e["entry_price"], "target_price": e["target_price"],
                 "stop_price": e["stop_price"], "trigger_prob": e["trigger_prob"],
                 "account_ann": v["account_ann"], "excess": excess,
                 "hit_prob_overall": v["hit_prob_overall"], "days_p50": v["days_p50"],
                 "kelly": v["kelly"], "ret_p10": v["ret_p10"]},
        "bt_mad": bt_mad,
        "category": category,
    }


def discover_scan(tickers: dict, p: dict, n_paths: int = 2000, years: int = 10,
                  progress=None) -> dict:
    """掃描整個股票池:datasource 批量取得(DB 快取優先)→ 逐檔 quick_one。"""
    from datasource import get_history_many
    items = list(tickers.items())
    total = len(items)
    data = get_history_many([t for t, _ in items], years, progress=progress)

    closes_map, errors, results = {}, [], []
    for t, _ in items:
        if t not in data:
            errors.append({"ticker": t, "reason": "無資料或下載失敗"})
    for i, (t, nm) in enumerate(items):
        if t not in data:
            continue
        closes, dates, meta = data[t]
        if len(closes) < TRADING_DAYS:
            errors.append({"ticker": t, "reason": "資料不足一年"})
            continue
        closes_map[t] = pd.Series(closes, index=pd.to_datetime(dates))
        if progress:
            progress("analyze", i, total, t)
        try:
            display = meta.get("name") if meta.get("name") not in (None, t) else nm
            results.append(quick_one(closes, t, display or nm, p, n_paths))
        except Exception as exc:
            errors.append({"ticker": t, "reason": f"分析失敗:{exc}"})

    clusters = correlation_clusters(closes_map, [r["ticker"] for r in results])
    return {"results": results, "errors": errors, "clusters": clusters}


def correlation_clusters(series_map: dict, tickers: list, thr: float = 0.8) -> list:
    """近一年日報酬相關 > thr 的連通群(≥2 檔)— 這些標的漲跌高度同步,應視為同一注。
    閾值 0.8:0.7 會讓大型股透過大盤 ETF 連通成單一巨群,失去產業辨識度。"""
    cols = {t: np.log(series_map[t]).diff() for t in tickers if t in series_map}
    if len(cols) < 2:
        return []
    df = pd.DataFrame(cols).tail(TRADING_DAYS)
    df = df.dropna(axis=1, thresh=int(TRADING_DAYS * 0.8))
    names = list(df.columns)
    if len(names) < 2:
        return []
    corr = df.corr().to_numpy()

    parent = list(range(len(names)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if corr[i, j] > thr:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i, t in enumerate(names):
        groups.setdefault(find(i), []).append(t)
    return sorted([g for g in groups.values() if len(g) >= 2], key=len, reverse=True)


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
