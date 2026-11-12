"""
Revisa XAUUSD una vez y envía la señal a Telegram si hay cruce.
Pensado para correr cada 15 min en GitHub Actions (gratis).
"""
import os
import requests
import pandas as pd
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
TEST = os.environ.get("TEST") == "1"

SYMBOL = "GC=F"
INTERVAL = "15m"
ATR_SL, ATR_TP = 1.5, 3.0


def send(text: str):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
        timeout=20,
    )


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def main():
    df = yf.download(SYMBOL, period="5d", interval=INTERVAL,
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 35:
        if TEST:
            send("No pude obtener datos del mercado (¿mercado cerrado?).")
        return

    df = df.iloc[:-1]  # quitar la vela en formación; usar solo velas cerradas
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    r = float(rsi(close).iloc[-1])
    atr = float((high - low).rolling(14).mean().iloc[-1])
    price = float(close.iloc[-1])

    up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
    dn = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

    side = None
    if up and r < 70:
        side, sl, tp = "BUY", price - ATR_SL * atr, price + ATR_TP * atr
    elif dn and r > 30:
        side, sl, tp = "SELL", price + ATR_SL * atr, price - ATR_TP * atr

    if side:
        icon = "🟢" if side == "BUY" else "🔴"
        send(f"{icon} *{side} XAUUSD*\nEntrada: {price:.2f}\nSL: {sl:.2f}\n"
             f"TP: {tp:.2f}\nRSI: {r:.1f} | TF: {INTERVAL}\n\n"
             f"⚠️ No es asesoría financiera.")
    elif TEST:
        send(f"✅ Bot funcionando. Sin señal ahora.\nPrecio: {price:.2f} | RSI: {r:.1f}")


if __name__ == "__main__":
    main()
