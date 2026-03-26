import asyncio
import json
import os
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from telegram import Bot

import backtest as bt
from news_fetcher import NewsFetcher, analyze_sentiment_detailed
from trading import PaperWallet

SYMBOL = "BZ=F"
BASE_CHECK_INTERVAL_SECONDS = 15
IN_POSITION_CHECK_INTERVAL_SECONDS = 5
LOOP_JITTER_SECONDS = 2
NEWS_CHECK_EVERY_N_LOOPS = 10
SHORT_EMA_WINDOW = 5
LONG_EMA_WINDOW = 20
TRAILING_STOP_PCT = 0.5
MAX_LOSS_PCT = -4.0
BUY_SCORE_THRESHOLD = 8.0
BUY_SCORE_THRESHOLD_NO_POSITION = 1.0
SELL_SCORE_THRESHOLD = -4.0
RSI_WINDOW = 14
COOLDOWN_LOOPS_AFTER_SELL = 0
VOL_HIGH_THRESHOLD = 0.22
VOL_LOW_THRESHOLD = 0.08
DEFAULT_TELEGRAM_BOT_TOKEN = "8364365943:AAEHGl8yWSI-60xwkuDdXUKnYxMQevzoQuM"
DEFAULT_TELEGRAM_CHAT_ID = "-1003780545165"
BEST_PARAMS_PATH = Path("best_params.json")


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {message}", flush=True)


def load_runtime_params() -> None:
    global SHORT_EMA_WINDOW
    global LONG_EMA_WINDOW
    global BUY_SCORE_THRESHOLD_NO_POSITION
    global SELL_SCORE_THRESHOLD
    global TRAILING_STOP_PCT
    global MAX_LOSS_PCT
    global RSI_WINDOW

    if not BEST_PARAMS_PATH.exists():
        log("best_params.json bulunamadi, varsayilan ayarlar kullaniliyor.")
        return

    try:
        data = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"best_params.json okunamadi: {exc}")
        return

    SHORT_EMA_WINDOW = int(data.get("short_ema", SHORT_EMA_WINDOW))
    LONG_EMA_WINDOW = int(data.get("long_ema", LONG_EMA_WINDOW))
    BUY_SCORE_THRESHOLD_NO_POSITION = float(data.get("buy_threshold", BUY_SCORE_THRESHOLD_NO_POSITION))
    SELL_SCORE_THRESHOLD = float(data.get("sell_threshold", SELL_SCORE_THRESHOLD))
    TRAILING_STOP_PCT = float(data.get("trailing_stop_pct", TRAILING_STOP_PCT))
    MAX_LOSS_PCT = float(data.get("max_loss_pct", MAX_LOSS_PCT))
    RSI_WINDOW = int(data.get("rsi_window", RSI_WINDOW))
    log(
        "Optimize parametreleri yuklendi: "
        f"short={SHORT_EMA_WINDOW}, long={LONG_EMA_WINDOW}, buy={BUY_SCORE_THRESHOLD_NO_POSITION}, "
        f"sell={SELL_SCORE_THRESHOLD}, trail={TRAILING_STOP_PCT}, max_loss={MAX_LOSS_PCT}, rsi={RSI_WINDOW}"
    )


def maybe_run_startup_backtest() -> None:
    run_flag = os.getenv("RUN_BACKTEST_ON_START", "1").strip().lower()
    if run_flag not in {"1", "true", "yes", "on"}:
        log("Startup backtest kapali (RUN_BACKTEST_ON_START=0).")
        return

    log("Startup backtest basliyor...")
    try:
        bt.main()
        log("Startup backtest tamamlandi, best_params.json guncellendi.")
    except Exception as exc:
        log(f"Startup backtest hatasi: {exc}. Varsayilan/onceki parametrelerle devam ediliyor.")


def get_current_price(symbol: str = SYMBOL) -> Optional[float]:
    ticker = yf.Ticker(symbol)

    # Use recent intraday data; fall back to latest daily close.
    intraday = ticker.history(period="1d", interval="1m")
    if not intraday.empty:
        return float(intraday["Close"].dropna().iloc[-1])

    daily = ticker.history(period="5d", interval="1d")
    if not daily.empty:
        return float(daily["Close"].dropna().iloc[-1])

    return None


async def send_telegram(bot: Optional[Bot], chat_id: Optional[str], message: str) -> None:
    if not bot or not chat_id:
        return
    await bot.send_message(chat_id=chat_id, text=message)


def format_end_of_day_report(wallet: PaperWallet) -> str:
    position = wallet.get_position()
    position_text = "Yok"
    if position:
        position_text = (
            f"Açık Pozisyon: {position.symbol} | Adet: {position.quantity:.6f} | "
            f"Maliyet: {position.avg_price:.4f}"
        )

    return (
        "Gun Sonu Raporu\n"
        f"Balance: ${wallet.get_balance():.2f}\n"
        f"Realized PnL: ${float(wallet.state.get('realized_pnl', 0.0)):.2f}\n"
        f"{position_text}"
    )


def build_trade_summary(wallet: PaperWallet, price: float) -> str:
    realized = float(wallet.state.get("realized_pnl", 0.0))
    unrealized_pct = wallet.current_unrealized_pnl_percent(price)
    position = wallet.get_position()
    if not position:
        return (
            f"Guncel bakiye: ${wallet.get_balance():.2f}\n"
            f"Toplam gerceklesen kar/zarar: ${realized:.2f}\n"
            "Acik pozisyon: yok"
        )
    return (
        f"Guncel bakiye: ${wallet.get_balance():.2f}\n"
        f"Toplam gerceklesen kar/zarar: ${realized:.2f}\n"
        f"Acik pozisyon: {position.symbol} | adet={position.quantity:.6f} | "
        f"maliyet=${position.avg_price:.4f} | anlik PnL={unrealized_pct:.2f}%"
    )


def compute_rsi(prices: pd.Series, window: int = RSI_WINDOW) -> float:
    if len(prices) < window + 1:
        return 50.0
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean().iloc[-1]
    avg_loss = loss.rolling(window=window).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_volatility_pct(prices: pd.Series, lookback: int = 20) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    returns = prices.pct_change().dropna().tail(lookback)
    return float(returns.std() * 100.0)


def classify_regime(short_ema: Optional[float], long_ema: Optional[float], vol_pct: float) -> str:
    if short_ema is None or long_ema is None:
        return "UNDEFINED"
    if vol_pct >= VOL_HIGH_THRESHOLD:
        return "HIGH_VOL"
    if short_ema > long_ema:
        return "UPTREND"
    if short_ema < long_ema:
        return "DOWNTREND"
    return "RANGE"


async def trading_loop() -> None:
    maybe_run_startup_backtest()
    load_runtime_params()
    token = os.getenv("TELEGRAM_BOT_TOKEN", DEFAULT_TELEGRAM_BOT_TOKEN)
    chat_id = os.getenv("TELEGRAM_CHAT_ID", DEFAULT_TELEGRAM_CHAT_ID)
    bot = Bot(token=token) if token else None

    wallet = PaperWallet(state_file="wallet_state.json", initial_balance=1000.0)
    news_fetcher = NewsFetcher()
    last_report_date = None
    sentiment_action = "TUT"
    sentiment_score = 0.0
    sentiment_conf = 0
    sentiment_meta = "init"
    loop_count = 0
    price_buffer = deque(maxlen=LONG_EMA_WINDOW + 10)
    peak_pnl_percent = None
    consecutive_price_failures = 0
    cooldown_until_loop = 0
    trade_loops: deque[int] = deque(maxlen=64)
    log("Bot baslatildi, trading loop aktif.")

    await send_telegram(
        bot,
        chat_id,
        "AI Trading Simulator basladi. Ticker: BZ=F, hizli izleme aktif, bakiye: "
        f"${wallet.get_balance():.2f}",
    )

    while True:
        loop_count += 1
        log(f"Yeni dongu basladi. loop={loop_count}")
        now_utc = datetime.now(timezone.utc)
        price = get_current_price(SYMBOL)
        if price is None:
            consecutive_price_failures += 1
            fail_sleep = min(120, BASE_CHECK_INTERVAL_SECONDS * (2 ** min(consecutive_price_failures, 3)))
            log("Fiyat cekilemedi, sonraki denemeye geciliyor.")
            await send_telegram(bot, chat_id, "Fiyat verisi alinamadi. Kisa sure sonra tekrar denenecek.")
            await asyncio.sleep(fail_sleep)
            continue
        consecutive_price_failures = 0

        log(f"Fiyat cekildi: {SYMBOL} = ${price:.4f}")
        price_buffer.append(price)

        if loop_count == 1 or loop_count % NEWS_CHECK_EVERY_N_LOOPS == 0:
            log("Haberler okunuyor ve sentiment analiz ediliyor...")
            news_items = news_fetcher.fetch_latest_news(max_items_per_source=2)
            sentiment_result = analyze_sentiment_detailed(news_items)
            sentiment_action = sentiment_result.action
            sentiment_score = sentiment_result.score
            sentiment_conf = sentiment_result.confidence
            sentiment_meta = (
                f"themes={','.join(sentiment_result.matched_themes[:3]) or 'none'} | "
                f"sources={sentiment_result.source_count} | {sentiment_result.summary}"
            )
            log(
                f"Haber analizi tamam: action={sentiment_action}, score={sentiment_score:.1f}, "
                f"conf={sentiment_conf}, news_count={len(news_items)}"
            )
            await send_telegram(
                bot,
                chat_id,
                f"Haber analizi: {sentiment_action} | skor={sentiment_score:.1f} | "
                f"conf={sentiment_conf}\nFiyat: ${price:.4f} | Haber adedi: {len(news_items)}\n"
                f"{sentiment_meta}",
            )

        has_position = wallet.has_open_position()
        pnl_percent = wallet.current_unrealized_pnl_percent(price)
        if has_position:
            peak_pnl_percent = pnl_percent if peak_pnl_percent is None else max(peak_pnl_percent, pnl_percent)
        else:
            peak_pnl_percent = None

        short_ema = None
        long_ema = None
        prices = None
        rsi = 50.0
        vol_pct = 0.0
        if len(price_buffer) >= LONG_EMA_WINDOW:
            prices = pd.Series(list(price_buffer), dtype=float)
            short_ema = float(prices.ewm(span=SHORT_EMA_WINDOW, adjust=False).mean().iloc[-1])
            long_ema = float(prices.ewm(span=LONG_EMA_WINDOW, adjust=False).mean().iloc[-1])
            rsi = compute_rsi(prices, RSI_WINDOW)
            vol_pct = compute_volatility_pct(prices, LONG_EMA_WINDOW)

        technical_score = 0.0
        ema_gap_pct = 0.0
        if short_ema is not None and long_ema is not None and long_ema != 0:
            ema_gap_pct = ((short_ema - long_ema) / long_ema) * 100.0
            technical_score = max(-20.0, min(20.0, ema_gap_pct * 80.0))

        # News-first model: sentiment is dominant, technical only for timing.
        decision_score = (sentiment_score * 0.75) + (technical_score * 0.25)
        regime = classify_regime(short_ema, long_ema, vol_pct)
        if regime == "UPTREND" and 45 <= rsi <= 65:
            decision_score += 2.0
        if regime == "DOWNTREND" and rsi < 35:
            decision_score -= 2.0
        if regime == "HIGH_VOL":
            decision_score -= 1.2
        log(
            f"Skorlar: news={sentiment_score:.1f}, tech={technical_score:.1f}, "
            f"total={decision_score:.1f}, rsi={rsi:.1f}, vol={vol_pct:.3f}, regime={regime}"
        )

        buy_threshold = BUY_SCORE_THRESHOLD_NO_POSITION if not has_position else BUY_SCORE_THRESHOLD
        sentiment_block = sentiment_action == "SAT" and sentiment_conf >= 55
        cooldown_active = loop_count < cooldown_until_loop
        technical_aggressive_entry = (
            short_ema is not None
            and long_ema is not None
            and short_ema > long_ema
            and technical_score >= 4.0
            and ema_gap_pct >= 0.03
        )
        ultra_aggressive_bootstrap_entry = (
            not has_position
            and short_ema is not None
            and long_ema is not None
            and short_ema > long_ema
            and technical_score >= 1.5
            and rsi < 72
        )
        pullback_reentry = (
            not has_position
            and short_ema is not None
            and long_ema is not None
            and short_ema > long_ema
            and 40 <= rsi <= 55
            and technical_score >= 1.0
        )
        buy_signal = (
            not sentiment_block
            and not cooldown_active
            and (
                decision_score >= buy_threshold
                or (not has_position and technical_aggressive_entry)
                or ultra_aggressive_bootstrap_entry
                or pullback_reentry
            )
            and short_ema is not None
            and long_ema is not None
            and short_ema > long_ema
        )

        sell_reasons = []
        if has_position and sentiment_action == "SAT":
            sell_reasons.append("negatif haber")
        if has_position and short_ema is not None and long_ema is not None and short_ema < long_ema:
            sell_reasons.append("trend asagi dondu")
        if has_position and decision_score <= SELL_SCORE_THRESHOLD:
            sell_reasons.append("toplam skor bozuldu")
        if has_position and rsi >= 78 and pnl_percent > 0.35:
            sell_reasons.append("rsi asiri alim")
        if has_position and pnl_percent <= MAX_LOSS_PCT:
            sell_reasons.append("maksimum zarar limiti")
        if has_position and peak_pnl_percent is not None:
            trailing_threshold = peak_pnl_percent - TRAILING_STOP_PCT
            if peak_pnl_percent > 0.3 and pnl_percent < trailing_threshold:
                sell_reasons.append("trailing stop")

        if buy_signal and not has_position and wallet.get_balance() > 1:
            # Dynamic sizing: higher confidence => higher position size.
            confidence_ratio = max(0.35, min(0.90, sentiment_conf / 100.0))
            if sentiment_conf < 35:
                confidence_ratio = max(confidence_ratio, 0.85)
            usd_to_use = wallet.get_balance() * confidence_ratio
            if regime == "HIGH_VOL":
                usd_to_use *= 0.65
            if regime == "UPTREND" and 45 <= rsi <= 62 and decision_score > 5:
                usd_to_use *= 1.08
            usd_to_use = max(50.0, min(wallet.get_balance(), usd_to_use))
            buy_result = wallet.buy(symbol=SYMBOL, price=price, usd_amount=usd_to_use)
            if buy_result["success"]:
                trade_loops.append(loop_count)
                log(
                    f"ALIM yapildi: price=${price:.4f}, usd={usd_to_use:.2f}, "
                    f"balance=${wallet.get_balance():.2f}"
                )
                buy_reasons = [
                    (
                        f"toplam skor yuksek ({decision_score:.1f} >= {buy_threshold:.1f})"
                        if decision_score >= buy_threshold
                        else "agresif teknik giris (haber notru olsa da trend guclu)"
                    ),
                    (
                        "ultra agresif bootstrap girisi"
                        if ultra_aggressive_bootstrap_entry
                        else "standart agresif giris"
                    ),
                    f"rejim={regime}, rsi={rsi:.1f}, vol={vol_pct:.3f}",
                    f"haber {sentiment_action} (skor={sentiment_score:.1f}, conf={sentiment_conf})",
                    f"trend pozitif (EMA{SHORT_EMA_WINDOW} > EMA{LONG_EMA_WINDOW})",
                ]
                await send_telegram(
                    bot,
                    chat_id,
                    f"ALIM yapildi: {SYMBOL} @ ${price:.4f}\n"
                    f"Alim nedenleri: {' | '.join(buy_reasons)}\n"
                    f"EMA({SHORT_EMA_WINDOW})={short_ema:.4f}, EMA({LONG_EMA_WINDOW})={long_ema:.4f}\n"
                    f"Skorlar -> news={sentiment_score:.1f}, tech={technical_score:.1f}, total={decision_score:.1f}\n"
                    f"{build_trade_summary(wallet, price)}",
                )
            else:
                log(f"ALIM denenemedi: {buy_result.get('reason', 'bilinmeyen hata')}")

        elif has_position and sell_reasons:
            reason = ", ".join(sorted(set(sell_reasons)))
            sell_result = wallet.sell(price=price)
            if sell_result["success"]:
                trade_loops.append(loop_count)
                tx = sell_result["transaction"]
                if tx.get("pnl", 0.0) < 0:
                    cooldown_until_loop = loop_count + COOLDOWN_LOOPS_AFTER_SELL
                log(
                    f"SATIS yapildi: price=${price:.4f}, reason={reason}, "
                    f"balance=${wallet.get_balance():.2f}"
                )
                await send_telegram(
                    bot,
                    chat_id,
                    f"SATIS yapildi ({reason}): {SYMBOL} @ ${price:.4f}\n"
                    f"Satis nedenleri: {reason}\n"
                    f"Islem PnL: ${tx['pnl']:.2f} ({tx['pnl_percent']:.2f}%)\n"
                    f"Skorlar -> news={sentiment_score:.1f}, tech={technical_score:.1f}, total={decision_score:.1f}, "
                    f"rsi={rsi:.1f}, vol={vol_pct:.3f}\n"
                    f"{build_trade_summary(wallet, price)}",
                )
            else:
                log(f"SATIS denenemedi: {sell_result.get('reason', 'bilinmeyen hata')}")
        else:
            block_note = ""
            if cooldown_active:
                block_note = f" (cooldown aktif, {cooldown_until_loop - loop_count} loop kaldi)"
            log(f"Islem yok: bekleme modunda.{block_note}")

        # End-of-day report at first loop pass after UTC day changes.
        current_date = now_utc.date().isoformat()
        if last_report_date is None:
            last_report_date = current_date
        elif current_date != last_report_date:
            log("Gun sonu raporu gonderiliyor.")
            await send_telegram(bot, chat_id, format_end_of_day_report(wallet))
            last_report_date = current_date

        has_position_after = wallet.has_open_position()
        base_sleep = IN_POSITION_CHECK_INTERVAL_SECONDS if has_position_after else BASE_CHECK_INTERVAL_SECONDS
        jitter = random.uniform(-LOOP_JITTER_SECONDS, LOOP_JITTER_SECONDS)
        sleep_seconds = max(5.0, base_sleep + jitter)
        log(f"Uykuya geciliyor: {sleep_seconds:.1f} sn")
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    asyncio.run(trading_loop())

