
import os
import json
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# Yahoo Finance COMEX Gold Futures
TICKER = "GC=F"
DATA_SOURCE = "Yahoo Finance GC=F (Gold Futures)"

STATE_FILE = Path("last_alert.json")


def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )
    response.raise_for_status()

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the message.")


def load_candles() -> pd.DataFrame:

    data = yf.download(
        TICKER,
        period="5d",
        interval="5m",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"No 5-minute data returned for {TICKER}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).lower() for c in data.columns]

    required = {"open", "high", "low", "close"}

    if not required.issubset(data.columns):
        raise RuntimeError(
            f"Unexpected data columns: {list(data.columns)}"
        )

    data = data.dropna(
        subset=["open", "high", "low", "close"]
    ).copy()

    if len(data) < 80:
        raise RuntimeError("Not enough 5-minute candles.")

    # Analyze completed candles only
    data = data.iloc[:-1].copy()

    close = data["close"]

    # EMA
    data["ema20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    data["ema50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, float("nan"))

    data["rsi"] = 100 - 100 / (1 + rs)

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["atr"] = tr.ewm(
        alpha=1/14, adjust=False
    ).mean()

    return data.dropna()


def get_signal(data: pd.DataFrame):

    last = data.iloc[-1]
    previous = data.iloc[-2]

    buy = (
        last["ema20"] > last["ema50"]
        and last["close"] > last["ema20"]
        and 52 <= last["rsi"] <= 68
        and last["close"] > previous["high"]
    )

    sell = (
        last["ema20"] < last["ema50"]
        and last["close"] < last["ema20"]
        and 32 <= last["rsi"] <= 48
        and last["close"] < previous["low"]
    )

    if not buy and not sell:
        return None

    entry = float(last["close"])
    atr = float(last["atr"])

    if atr <= 0:
        return None

    side = "BUY" if buy else "SELL"

    if side == "BUY":
        sl = entry - 1.5 * atr
        tp1 = entry + 1.5 * atr
        tp2 = entry + 2 * atr
    else:
        sl = entry + 1.5 * atr
        tp1 = entry - 1.5 * atr
        tp2 = entry - 2 * atr

    candle_time = last.name

    if candle_time.tzinfo is None:
        candle_time = candle_time.tz_localize("UTC")
    else:
        candle_time = candle_time.tz_convert("UTC")

    return {
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": float(last["rsi"]),
        "atr": atr,
        "candle": candle_time.isoformat(),
    }


def main():

    print(f"Downloading Gold data: {TICKER}")

    data = load_candles()

    print(f"Loaded {len(data)} candles successfully.")

    signal = get_signal(data)

    if signal is None:
        print("No qualifying BUY or SELL setup.")
        return

    candle_id = signal["candle"]

    old_state = {}

    if STATE_FILE.exists():
        try:
            old_state = json.loads(
                STATE_FILE.read_text()
            )
        except (json.JSONDecodeError, OSError):
            old_state = {}

    if old_state.get("last_alert_candle") == candle_id:
        print("Duplicate signal skipped.")
        return

    message = (
        f"🟡 GOLD SIGNAL | M5 | {signal['side']}\n\n"
        f"Data: {DATA_SOURCE}\n"
        f"Entry reference: {signal['entry']:.2f}\n"
        f"Stop Loss: {signal['sl']:.2f}\n"
        f"Take Profit 1: {signal['tp1']:.2f}\n"
        f"Take Profit 2: {signal['tp2']:.2f}\n\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"ATR: {signal['atr']:.2f}\n"
        f"Closed candle: {candle_id}\n\n"
        "⚠️ Alert only. No trades are placed.\n"
        "This is a rule-based test signal, not financial advice.\n"
        "GC=F is Gold Futures, not PU Prime XAUUSD.s.\n"
        "Prices, spread and execution may differ."
    )

    send_telegram(message)

    STATE_FILE.write_text(
        json.dumps(
            {"last_alert_candle": candle_id},
            indent=2,
        )
    )

    print(
        f"Telegram alert sent: {signal['side']} "
        f"at {candle_id}"
    )


if __name__ == "__main__":
    send_telegram(
        "✅ Telegram Connection Test Successful!\n\n"
        "Your Gold Trading Bot is connected.\n"
        "Bot: Bilal Gold Alerts"
    )
    print("Telegram test message sent successfully.")


