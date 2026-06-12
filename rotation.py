# -*- coding: utf-8 -*-
"""組合輪動回測:多檔標的、等回檔進場、規則出場、資金持續再部署。

單檔「帳戶年化」假設出場後資金閒置,是保守下限;這裡在真實歷史上模擬
「出場後資金立即等待下一個訊號、輪動到任何觸發的標的」的組合級結果 —
這才是把多檔掃描當成一個完整策略時「實際能拿到」的年化。

語義與單檔策略一致:
  進場訊號:收盤 ≤ 近 wait 日滾動最高收盤 × (1 - dip),且該檔無持倉、有空位
  出場:與單檔相同的四種規則(達標以目標價限價出場;停損/移停/到期以收盤)
  部位:等分 — 進場金額 = min(當時淨值 / 最大持倉數, 剩餘現金)
  現金:依年利率計日息
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DIP_LEVELS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]


def _equal_weight_hold(closes: np.ndarray, cash_d: float) -> np.ndarray:
    """等權買入持有基準:每檔 1/N 資金,於該檔第一個有效價日買入,之前吃現金日息。"""
    T, N = closes.shape
    nav = np.zeros(T)
    for j in range(N):
        col = closes[:, j]
        valid = np.where(~np.isnan(col))[0]
        if len(valid) == 0:
            nav += (1.0 / N) * (1 + cash_d) ** np.arange(1, T + 1)
            continue
        f = valid[0]
        unit = np.empty(T)
        unit[:f] = (1 + cash_d) ** np.arange(1, f + 1)          # 上市前吃利息
        base = (1 + cash_d) ** f
        px = pd.Series(col).ffill().to_numpy()
        unit[f:] = base * px[f:] / px[f]
        nav += unit / N
    return nav


def rotation_backtest(price_df: pd.DataFrame, dip: float, p: dict, variant: str,
                      max_pos: int = 5) -> dict:
    """單一回檔深度 × 單一出場規則的組合輪動回測。"""
    closes = price_df.to_numpy(dtype=float)
    T, N = closes.shape
    wait, max_hold = p["wait"], p["max_hold"]
    target, stop, trail = p["target"], p["stop"], p["trail"]
    use_stop = variant in ("fixed_stop", "trail_stop")
    use_trail = variant in ("trail", "trail_stop")
    cash_d = (1 + p["cash_rate"]) ** (1 / TRADING_DAYS) - 1

    roll_high = price_df.rolling(wait, min_periods=wait).max().to_numpy()
    signal = closes <= roll_high * (1 - dip)        # (T,N) bool,NaN 比較自然為 False

    cash, pos = 1.0, {}
    nav_series = np.empty(T)
    rets, holds = [], []
    expo_sum = 0.0

    for t in range(T):
        px = closes[t]
        # 出場
        for j in list(pos.keys()):
            o = pos[j]
            pj = px[j]
            if np.isnan(pj):
                continue
            o["last"] = pj
            rel = pj / o["entry"]
            held = t - o["day0"]
            exit_price = None
            if not o["hit"] and rel >= 1 + target:
                if use_trail:
                    o["hit"], o["runmax"] = True, pj
                else:
                    exit_price = o["entry"] * (1 + target)
            elif o["hit"] and use_trail:
                o["runmax"] = max(o["runmax"], pj)
                if pj <= o["runmax"] * (1 - trail):
                    exit_price = pj
            if exit_price is None and use_stop and not o["hit"] and rel <= 1 - stop:
                exit_price = pj
            if exit_price is None and held >= max_hold:
                exit_price = pj
            if exit_price is not None:
                cash += o["shares"] * exit_price
                rets.append(exit_price / o["entry"] - 1)
                holds.append(held)
                del pos[j]
        # 進場
        if len(pos) < max_pos and cash > 1e-9:
            hits = np.where(signal[t])[0]
            if len(hits):
                nav_now = cash + sum(o["shares"] * o["last"] for o in pos.values())
                for j in hits:
                    if len(pos) >= max_pos or cash <= 1e-9:
                        break
                    if j in pos:
                        continue
                    alloc = min(nav_now / max_pos, cash)
                    pos[j] = {"shares": alloc / px[j], "entry": px[j], "last": px[j],
                              "day0": t, "hit": False, "runmax": px[j]}
                    cash -= alloc
        # 日終
        cash *= 1 + cash_d
        nav = cash + sum(o["shares"] * o["last"] for o in pos.values())
        nav_series[t] = nav
        expo_sum += 1 - cash / nav

    yrs = T / TRADING_DAYS
    run_max = np.maximum.accumulate(nav_series)
    rets_a = np.array(rets) if rets else np.array([0.0])
    return {
        "dip": dip,
        "ann": float(nav_series[-1] ** (1 / yrs) - 1),
        "max_dd": float((nav_series / run_max - 1).min()),
        "trades": len(rets),
        "win_rate": float((rets_a > 0).mean()) if rets else None,
        "avg_hold": float(np.mean(holds)) if holds else None,
        "avg_exposure": expo_sum / T,
    }


def rotation_run(series_map: dict, tickers: list, p: dict, variant: str,
                 max_pos: int = 5) -> dict:
    """全部回檔深度的輪動回測 + 等權持有基準。"""
    cols = {t: series_map[t] for t in tickers if t in series_map}
    if len(cols) < 2:
        return {"available": False, "reason": "可用標的不足兩檔。"}
    df = pd.DataFrame(cols).sort_index()
    if len(df) < TRADING_DAYS * 2:
        return {"available": False, "reason": "共同歷史不足兩年。"}

    cash_d = (1 + p["cash_rate"]) ** (1 / TRADING_DAYS) - 1
    bh_nav = _equal_weight_hold(df.to_numpy(dtype=float), cash_d)
    yrs = len(df) / TRADING_DAYS
    bh_run = np.maximum.accumulate(bh_nav)

    rows = [rotation_backtest(df, dip, p, variant, max_pos) for dip in DIP_LEVELS]
    return {
        "available": True,
        "n_tickers": len(cols),
        "years": round(yrs, 1),
        "from": df.index[0].strftime("%Y-%m-%d"),
        "to": df.index[-1].strftime("%Y-%m-%d"),
        "variant": variant,
        "max_pos": max_pos,
        "rows": rows,
        "bh": {"ann": float(bh_nav[-1] ** (1 / yrs) - 1),
               "max_dd": float((bh_nav / bh_run - 1).min())},
    }
