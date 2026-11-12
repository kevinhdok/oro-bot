"""
Búsqueda de estrategia para XAUUSD (oro), velas de 1 hora, ~2 años de datos.
Prueba varias estrategias con una regla estricta:
  - Se elige la mejor configuración con los datos de ENTRENAMIENTO (primer 60%).
  - Se evalúa en datos que nunca vio (último 40%).
Solo una estrategia que gane en ambos tramos merece usarse.
Se ejecuta manualmente desde GitHub Actions y envía el resultado a Telegram.
"""
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

SYMBOL = "GC=F"
INTERVAL = "1h"
PERIOD = "730d"        # máximo que da Yahoo para velas de 1 hora
COST = 0.5             # spread + deslizamiento por operación, en puntos de precio
MAX_BARS = 48          # una operación se cierra a las 48 h si no toca TP ni SL
TRAIN_FRAC = 0.6
MIN_TRAIN = 30         # mínimo de operaciones para considerar una configuración
MIN_TEST = 20
PF_OK = 1.15           # factor de beneficio mínimo para considerarla válida
WARM = 250             # velas iniciales descartadas (para que existan los indicadores)

EXITS = [(1.5, 1.5), (1.5, 3), (2, 2), (2, 4), (3, 3), (3, 6)]   # (SL, TP) en múltiplos de ATR


def send(text: str):
    for i in range(0, len(text), 3900):
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text[i:i + 3900]},
            timeout=20,
        )


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)


def shift(a, k=1):
    out = np.full_like(a, np.nan, dtype=float)
    out[k:] = a[:-k]
    return out


def edge(x):
    """Solo la primera vela en que se cumple la condición."""
    prev = np.r_[False, x[:-1]]
    return x & ~prev


# ---------- Estrategias: devuelven +1 (compra), -1 (venta) o 0 por vela ----------
def sig_pullback(I, lvl):
    up = (I["c"] > I["ema200"]) & (I["ema50"] > I["ema200"])
    dn = (I["c"] < I["ema200"]) & (I["ema50"] < I["ema200"])
    r, rp = I["rsi"], shift(I["rsi"])
    buy = up & (rp > lvl) & (r <= lvl)
    sell = dn & (rp < 100 - lvl) & (r >= 100 - lvl)
    return buy.astype(int) - sell.astype(int)


def sig_donchian(I, n):
    hh = shift(pd.Series(I["h"]).rolling(n).max().to_numpy())
    ll = shift(pd.Series(I["l"]).rolling(n).min().to_numpy())
    buy = edge((I["c"] > hh) & (I["c"] > I["ema200"]))
    sell = edge((I["c"] < ll) & (I["c"] < I["ema200"]))
    return buy.astype(int) - sell.astype(int)


def sig_cross(I, p):
    f, s = p
    cs = pd.Series(I["c"])
    ef = cs.ewm(span=f, adjust=False).mean().to_numpy()
    es = cs.ewm(span=s, adjust=False).mean().to_numpy()
    up = (ef > es) & (shift(ef) <= shift(es)) & (I["c"] > I["ema200"])
    dn = (ef < es) & (shift(ef) >= shift(es)) & (I["c"] < I["ema200"])
    return up.astype(int) - dn.astype(int)


def sig_boll(I, k):
    cs = pd.Series(I["c"])
    m = cs.rolling(20).mean().to_numpy()
    sd = cs.rolling(20).std().to_numpy()
    buy = edge(I["c"] < m - k * sd)
    sell = edge(I["c"] > m + k * sd)
    return buy.astype(int) - sell.astype(int)


STRATS = [
    ("Tendencia + retroceso RSI", sig_pullback, [30, 35, 40], lambda p: f"RSI {p}"),
    ("Ruptura de rango + tendencia", sig_donchian, [20, 40, 55], lambda p: f"{p} velas"),
    ("Cruce de EMAs + tendencia", sig_cross, [(9, 21), (12, 26), (20, 50)], lambda p: f"EMA {p[0]}/{p[1]}"),
    ("Reversión Bollinger", sig_boll, [2, 2.5], lambda p: f"{p} desv."),
]


# ---------- Simulación ----------
def simulate(I, sig, sl_m, tp_m, lo, hi):
    """Una operación a la vez. Entra en la apertura de la vela siguiente a la señal.
    Si en una vela toca SL y TP, asume SL (caso conservador)."""
    n = len(I["c"])
    trades, busy = [], -1
    for i in range(lo, hi):
        s = sig[i]
        if s == 0 or i < busy or i + 1 >= n:
            continue
        a = I["atr"][i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = I["o"][i + 1]
        sl = entry - s * sl_m * a
        tp = entry + s * tp_m * a
        end = min(n, i + 1 + MAX_BARS)
        ex, k_exit = None, end - 1
        for k in range(i + 1, end):
            hk, lk = I["h"][k], I["l"][k]
            if s == 1:
                if lk <= sl:
                    ex, k_exit = sl, k
                    break
                if hk >= tp:
                    ex, k_exit = tp, k
                    break
            else:
                if hk >= sl:
                    ex, k_exit = sl, k
                    break
                if lk <= tp:
                    ex, k_exit = tp, k
                    break
        if ex is None:
            ex = I["c"][end - 1]
        trades.append((s, s * (ex - entry) - COST))
        busy = k_exit + 1
    return trades


def stats(trades):
    if not trades:
        return dict(n=0, pf=0.0, total=0.0, wr=0.0, buy=0.0, sell=0.0)
    p = np.array([t[1] for t in trades])
    sd = np.array([t[0] for t in trades])
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    pf = wins / losses if losses > 0 else 99.0
    return dict(n=len(p), pf=pf, total=p.sum(), wr=(p > 0).mean() * 100,
                buy=p[sd == 1].sum(), sell=p[sd == -1].sum())


def main():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    if df.empty or len(df) < 3000:
        send("Búsqueda de estrategia: no pude obtener suficientes datos de Yahoo Finance.")
        return
    df = df.iloc[:-1]

    o = df["Open"].squeeze().to_numpy(dtype=float)
    h = df["High"].squeeze().to_numpy(dtype=float)
    l = df["Low"].squeeze().to_numpy(dtype=float)
    cser = df["Close"].squeeze().astype(float)
    c = cser.to_numpy()
    pc = cser.shift(1)
    tr = pd.concat([df["High"].squeeze() - df["Low"].squeeze(),
                    (df["High"].squeeze() - pc).abs(),
                    (df["Low"].squeeze() - pc).abs()], axis=1).max(axis=1)
    I = dict(
        o=o, h=h, l=l, c=c,
        atr=tr.rolling(14).mean().to_numpy(),
        ema50=cser.ewm(span=50, adjust=False).mean().to_numpy(),
        ema200=cser.ewm(span=200, adjust=False).mean().to_numpy(),
        rsi=rsi(cser).to_numpy(),
    )

    n = len(c)
    split = WARM + int((n - WARM) * TRAIN_FRAC)
    years = (df.index[-1] - df.index[0]).days / 365
    move = (c[-1] / c[split] - 1) * 100

    lines = [
        f"🔬 BÚSQUEDA DE ESTRATEGIA XAUUSD (velas 1h, {years:.1f} años)",
        "Se elige la mejor configuración con datos de entrenamiento y se evalúa en datos nuevos que nunca vio.",
        f"Costo supuesto por operación: {COST} pts. 1 pt = $1 con 0.01 lote.",
        f"En el tramo de prueba el oro cambió {move:+.0f}%: si solo ganan las compras, puede ser la tendencia y no la estrategia.",
        "",
    ]

    valid = []
    with np.errstate(invalid="ignore"):
        for name, fn, params, label in STRATS:
            best = None
            for p in params:
                sig = fn(I, p)
                for sl_m, tp_m in EXITS:
                    tr_t = stats(simulate(I, sig, sl_m, tp_m, WARM, split))
                    if tr_t["n"] < MIN_TRAIN or tr_t["total"] <= 0:
                        continue
                    if best is None or tr_t["pf"] > best["train"]["pf"]:
                        best = dict(p=p, sl=sl_m, tp=tp_m, train=tr_t, sig=sig)
            if best is None:
                lines += [f"{name}: ninguna configuración ganó ni en entrenamiento.", ""]
                continue
            te = stats(simulate(I, best["sig"], best["sl"], best["tp"], split, n - 1))
            ok = (te["n"] >= MIN_TEST and te["pf"] >= PF_OK
                  and best["train"]["pf"] >= PF_OK and te["total"] > 0)
            if ok:
                valid.append((name, label(best["p"]), best["sl"], best["tp"], te))
            tr_t = best["train"]
            lines += [
                f"{'✅' if ok else '❌'} {name} ({label(best['p'])}, SL {best['sl']}×ATR / TP {best['tp']}×ATR)",
                f"  Entrenamiento: {tr_t['n']} ops | aciertan {tr_t['wr']:.0f}% | {tr_t['total']:+.0f} pts | PF {tr_t['pf']:.2f}",
                f"  Prueba (datos nuevos): {te['n']} ops | aciertan {te['wr']:.0f}% | {te['total']:+.0f} pts | PF {te['pf']:.2f}",
                f"  Prueba por lado: BUY {te['buy']:+.0f} pts / SELL {te['sell']:+.0f} pts",
                "",
            ]

    if valid:
        lines.append(f"VEREDICTO: {len(valid)} estrategia(s) ganaron en entrenamiento Y en datos nuevos (PF ≥ {PF_OK}).")
        lines.append("Ojo: al probar tantas combinaciones, alguna puede pasar solo por suerte. 2 años es muestra limitada: primero demo.")
    else:
        lines.append("VEREDICTO: ninguna estrategia ganó de forma consistente en datos nuevos. No conviene operar dinero real con estas reglas.")
    lines.append("PF = ganado ÷ perdido. Mayor a 1 gana; menor a 1 pierde.")
    send("\n".join(lines))


if __name__ == "__main__":
    main()
