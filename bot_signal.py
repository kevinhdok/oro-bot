"""
Revisa XAUUSD una vez y envía la señal a Telegram si hay cruce.
Formato estilo canal: entrada, 5 TPs y SL.
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

# Ajustes de la señal (en múltiplos del ATR, la volatilidad promedio)
TP_STEP = 0.5    # distancia entre cada TP
N_TPS = 5        # cantidad de TPs
SL_MULT = 1.5    # distancia del stop loss
ZONE_MULT = 0.5  # ancho de la zona de entrada


def send(text: str):
    requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def build_message(side: str, price: float, atr: float) -> str:
    sign = 1 if side == "BUY" else -1
    zone = price - sign * ZONE_MULT * atr
    tps = [price + sign * TP_STEP * atr * k for k in range(1, N_TPS + 1)]
    sl = price - sign * SL_MULT * atr
    lines = [f"⭐ GOLD {side} NOW {round(price)}// {round(zone)}", ""]
    lines += [f"✅ TP {round(tp)}" for tp in tps]
    lines += ["", f"👀 SL {round(sl)}", "", "⚠️ No es asesoría financiera."]
    return "\n".join(lines)


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

    if up and r < 70:
        send(build_message("BUY", price, atr))
    elif dn and r > 30:
        send(build_message("SELL", price, atr))
    elif TEST:
        send(f"✅ Bot funcionando. Sin señal ahora.\nPrecio: {price:.2f} | RSI: {r:.1f}")
        send("Así se verá una señal de ejemplo:\n\n" + build_message("BUY", price, atr))


if __name__ == "__main__":
    main()
