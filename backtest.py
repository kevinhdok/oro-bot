"""
Backtest de la estrategia del bot XAUUSD (EMA 9/21 + RSI + ATR).
Usa los últimos ~60 días de velas de 15 min y envía el resultado a Telegram.
Se ejecuta manualmente desde GitHub Actions.
"""
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

SYMBOL = "GC=F"
INTERVAL = "15m"
PERIOD = "60d"       # máximo que da Yahoo para velas de 15 min

# Mismos ajustes que el bot
TP_STEP = 0.5        # distancia entre TPs (en ATR)
N_TPS = 5
SL_MULT = 1.5        # stop loss (en ATR)

SPREAD = 0.4         # costo supuesto por operación, en puntos de precio
MAX_BARS = 96        # una operación se cierra a las 24 h si no toca TP ni SL

COMBOS = [(1, 1), (1, 2), (1.5, 1.5), (1.5, 3), (2, 2), (2, 4), (3, 3)]  # (SL, TP) en ATR


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


def exit_price(side, sl, tp, highs, lows, last_close):
    """Recorre las velas siguientes. Si en una vela toca SL y TP, asume SL (caso conservador)."""
    for h, l in zip(highs, lows):
        if side == 1:
            if l <= sl:
                return sl
            if h >= tp:
                return tp
        else:
            if h >= sl:
                return sl
            if l <= tp:
                return tp
    return last_close


def stats(pnls):
    p = np.array(pnls, dtype=float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return (p > 0).mean() * 100, p.sum(), pf


def fmt_pf(pf):
    return "∞" if pf == float("inf") else f"{pf:.2f}"


def main():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 200:
        send("Backtest: no pude obtener suficientes datos de Yahoo Finance.")
        return
    df = df.iloc[:-1]  # quitar vela en formación

    close_s = df["Close"].squeeze()
    close = close_s.to_numpy(dtype=float)
    high = df["High"].squeeze().to_numpy(dtype=float)
    low = df["Low"].squeeze().to_numpy(dtype=float)

    fast = close_s.ewm(span=9, adjust=False).mean().to_numpy()
    slow = close_s.ewm(span=21, adjust=False).mean().to_numpy()
    r = rsi(close_s).to_numpy()
    atr = (df["High"].squeeze() - df["Low"].squeeze()).rolling(14).mean().to_numpy()

    signals = []
    for i in range(30, len(df) - MAX_BARS):
        up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
        dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]
        if up and r[i] < 70:
            signals.append((i, 1))
        elif dn and r[i] > 30:
            signals.append((i, -1))

    if not signals:
        send("Backtest: la estrategia no generó señales en el período.")
        return

    def run(sl_mult, tp_mult):
        pnls = []
        for i, side in signals:
            entry, a = close[i], atr[i]
            sl = entry - side * sl_mult * a
            tp = entry + side * tp_mult * a
            w = slice(i + 1, i + 1 + MAX_BARS)
            ex = exit_price(side, sl, tp, high[w], low[w], close[i + MAX_BARS])
            pnls.append(side * (ex - entry) - SPREAD)
        return pnls

    n_buy = sum(1 for _, s in signals if s == 1)
    days = (df.index[-1] - df.index[0]).days

    lines = [
        "📊 BACKTEST XAUUSD (oro, velas 15m)",
        f"Período: últimos {days} días | Señales: {len(signals)} ({n_buy} BUY / {len(signals) - n_buy} SELL)",
        f"Costo supuesto por operación: {SPREAD} pts. 1 pt = $1 con 0.01 lote.",
        "",
        f"FORMATO ACTUAL DEL BOT (SL {SL_MULT}×ATR, TPs cada {TP_STEP}×ATR)",
        "Resultado si cierras todo en cada TP:",
    ]
    legs = []
    for k in range(1, N_TPS + 1):
        p = run(SL_MULT, TP_STEP * k)
        legs.append(p)
        wr, tot, pf = stats(p)
        lines.append(f"TP{k}: aciertan {wr:.0f}% | total {tot:+.0f} pts | PF {fmt_pf(pf)}")

    split = np.mean(np.array(legs), axis=0)
    wr, tot, pf = stats(split)
    lines += [
        "",
        f"Repartiendo en {N_TPS} partes (una por TP): total {tot:+.0f} pts | PF {fmt_pf(pf)}",
        f"Promedio por señal: {split.mean():+.2f} pts",
        "",
        "OTRAS COMBINACIONES (una sola salida)",
    ]
    for sl_m, tp_m in COMBOS:
        wr, tot, pf = stats(run(sl_m, tp_m))
        lines.append(f"SL {sl_m}× / TP {tp_m}×: aciertan {wr:.0f}% | total {tot:+.0f} pts | PF {fmt_pf(pf)}")

    lines += [
        "",
        "PF (factor de beneficio) mayor a 1 = ganó dinero; menor a 1 = perdió.",
        "Ojo: son solo ~60 días, muestra pequeña. No garantiza resultados futuros.",
    ]
    send("\n".join(lines))


if __name__ == "__main__":
    main()
