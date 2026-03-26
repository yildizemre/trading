import itertools
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

SYMBOL = "BZ=F"
INITIAL_BALANCE = 1000.0
OUTPUT_PARAMS = Path("best_params.json")
OUTPUT_REPORT = Path("backtest_report.json")


@dataclass
class BacktestResult:
    final_balance: float
    pnl: float
    pnl_pct: float
    trades: int
    win_rate: float
    max_drawdown_pct: float


def fetch_history(symbol: str = SYMBOL, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError("Price history is empty.")
    return df[["Close"]].dropna().copy()


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.astype("float64").fillna(50.0)


def run_backtest(
    prices: pd.Series,
    short_ema: int,
    long_ema: int,
    buy_threshold: float,
    sell_threshold: float,
    trailing_stop_pct: float,
    max_loss_pct: float,
    rsi_window: int,
) -> BacktestResult:
    df = pd.DataFrame({"close": prices})
    df["ema_s"] = df["close"].ewm(span=short_ema, adjust=False).mean()
    df["ema_l"] = df["close"].ewm(span=long_ema, adjust=False).mean()
    df["ema_gap_pct"] = ((df["ema_s"] - df["ema_l"]) / df["ema_l"].replace(0, float("nan"))) * 100
    df["tech_score"] = (df["ema_gap_pct"] * 80).clip(-20, 20).fillna(0)
    df["rsi"] = compute_rsi(df["close"], rsi_window)

    balance = INITIAL_BALANCE
    qty = 0.0
    entry = 0.0
    peak_pnl = 0.0
    wins = 0
    losses = 0
    equity_curve: List[float] = []

    for row in df.itertuples():
        price = float(row.close)
        tech = float(row.tech_score)
        rsi = float(row.rsi)
        decision_score = tech  # backtest mode: pure technical proxy

        has_pos = qty > 0
        if has_pos:
            pnl_pct = ((price - entry) / entry) * 100
            peak_pnl = max(peak_pnl, pnl_pct)
        else:
            pnl_pct = 0.0
            peak_pnl = 0.0

        buy_signal = (
            not has_pos
            and row.ema_s > row.ema_l
            and rsi < 72
            and (decision_score >= buy_threshold or (tech >= 1.5 and row.ema_gap_pct >= 0.03))
        )
        sell_signal = (
            has_pos
            and (
                row.ema_s < row.ema_l
                or decision_score <= sell_threshold
                or pnl_pct <= max_loss_pct
                or (peak_pnl > 0.3 and pnl_pct < (peak_pnl - trailing_stop_pct))
                or (rsi >= 78 and pnl_pct > 0.35)
            )
        )

        if buy_signal:
            usd = balance
            qty = usd / price
            balance -= usd
            entry = price
        elif sell_signal:
            proceeds = qty * price
            pnl = proceeds - (qty * entry)
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
            balance += proceeds
            qty = 0.0
            entry = 0.0

        equity = balance + (qty * price)
        equity_curve.append(equity)

    if qty > 0:
        last_price = float(df["close"].iloc[-1])
        balance += qty * last_price
        pnl = (qty * last_price) - (qty * entry)
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

    final_balance = balance
    pnl = final_balance - INITIAL_BALANCE
    pnl_pct = (pnl / INITIAL_BALANCE) * 100
    trades = wins + losses
    win_rate = (wins / trades * 100) if trades else 0.0

    eq = pd.Series(equity_curve) if equity_curve else pd.Series([INITIAL_BALANCE])
    peak = eq.cummax()
    drawdown = (eq - peak) / peak.replace(0, float("nan"))
    max_drawdown_pct = abs(float(drawdown.min() * 100)) if not drawdown.empty else 0.0

    return BacktestResult(
        final_balance=round(final_balance, 2),
        pnl=round(pnl, 2),
        pnl_pct=round(pnl_pct, 2),
        trades=trades,
        win_rate=round(win_rate, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
    )


def score_result(result: BacktestResult) -> float:
    # Higher PnL and win rate, lower drawdown.
    return (result.pnl_pct * 1.8) + (result.win_rate * 0.3) - (result.max_drawdown_pct * 1.1)


def main(fast_mode: bool | None = None) -> None:
    if fast_mode is None:
        env_fast = str(os.getenv("BACKTEST_FAST_MODE", "1")).strip().lower()
        fast_mode = env_fast in {"1", "true", "yes", "on"}

    df = fetch_history()
    prices = df["Close"]

    if fast_mode:
        grid = {
            "short_ema": [5, 6],
            "long_ema": [20, 24],
            "buy_threshold": [1.0, 2.0],
            "sell_threshold": [-4.0, -5.0],
            "trailing_stop_pct": [0.5, 0.7],
            "max_loss_pct": [-3.0, -4.0],
            "rsi_window": [10, 14],
        }
    else:
        grid = {
            "short_ema": [4, 5, 6, 8],
            "long_ema": [16, 20, 24, 30],
            "buy_threshold": [1.0, 2.0, 3.0, 4.0],
            "sell_threshold": [-3.0, -4.0, -5.0, -6.0],
            "trailing_stop_pct": [0.4, 0.5, 0.7, 1.0],
            "max_loss_pct": [-2.5, -3.0, -4.0],
            "rsi_window": [10, 14, 18],
        }

    best_params: Dict[str, float] = {}
    best_result: BacktestResult | None = None
    best_score = float("-inf")
    all_results: List[Dict[str, object]] = []

    keys = list(grid.keys())
    total_combos = 1
    for k in keys:
        total_combos *= len(grid[k])
    print(f"Backtest mode: {'FAST' if fast_mode else 'FULL'} | raw combos: {total_combos}")

    checked = 0
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        if params["short_ema"] >= params["long_ema"]:
            continue

        checked += 1
        result = run_backtest(prices=prices, **params)
        s = score_result(result)
        row = {"params": params, "result": asdict(result), "score": round(s, 4)}
        all_results.append(row)
        if checked % 20 == 0:
            print(f"Backtest progress: checked={checked}")

        if s > best_score:
            best_score = s
            best_params = params
            best_result = result

    if not best_result:
        raise RuntimeError("No valid backtest result generated.")

    payload = {
        "symbol": SYMBOL,
        "period": "60d",
        "interval": "15m",
        "best_params": best_params,
        "best_result": asdict(best_result),
        "score": round(best_score, 4),
    }
    OUTPUT_PARAMS.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Best params saved to best_params.json")
    print(json.dumps(payload, indent=2))
    print(f"Evaluated combinations: {len(all_results)}")


if __name__ == "__main__":
    main()
