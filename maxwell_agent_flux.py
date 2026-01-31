# Copyright (c) 2025, Elias, Martha
# License: Apache 2.0 (see LICENSE)
# Terms, concepts and note on disclosure in README.md
# Elias, M. (2025). Applied Mathematics: Signal Geometric Framework for Finance and Agent-Based (Deterministic) Modeling. 
# Id be happy if you like my work: https://buymeacoffee.com/marthafay
# Disclaimer: research only
# I want to get hired! Contact marthaelias[at]protonmail.com

"""
maxwell_agent_flux.py — Maxwell-Inspired Features + Simple Agent
- E ~ Preis-Gradient (Momentum)
- B ~ "Krümmung" (Beschleunigung des Momentums)
- S ~ Poynting-ähnlicher Fluss  (hier skalar: E*B)
- u ~ Energiedichte            (0.5*(E^2+B^2))
Signal:
  S_z > +k*σ(S)  -> 1 (Buy)
  S_z < -k*σ(S)  -> 0 (Sell)
  else          -> 0.5 (Hold)
  
"""

from __future__ import annotations
import argparse, numpy as np, pandas as pd

def em_features(close: pd.Series, dt: float = 1.0, ma: int = 10) -> pd.DataFrame:
    p = close.astype(float).ffill()
    # Glätten für stabilere Ableitungen
    m = p.rolling(ma, min_periods=1).mean()
    # "Feldgrößen"
    E = np.gradient(m, dt)                # ~ Geschwindigkeit / Momentum
    B = np.gradient(E, dt)                # ~ Beschleunigung / Krümmung
    # Fluss & Energie
    S = E * B                             # in 2D-Proxy: S_z ~ E×B -> E*B
    u = 0.5 * (E**2 + B**2)               # Energiedichte
    # zusätzliche Stabilisierung
    Sz = pd.Series(S, index=p.index).ewm(span=ma, adjust=False).mean()
    u  = pd.Series(u, index=p.index).ewm(span=ma, adjust=False).mean()
    E  = pd.Series(E, index=p.index)
    B  = pd.Series(B, index=p.index)
    # “Spannungstensor”-Skizze: Diagonale als Druck/Spannung-Proxys
    Txx = 0.5*(E**2 - u)                  # reine Heuristik für Diversität
    Tyy = 0.5*(B**2 - u)
    return pd.DataFrame({"E":E, "B":B, "S":Sz, "u":u, "Txx":Txx, "Tyy":Tyy})

def maxwell_agent(close: pd.Series, k: float = 1.0, vol_win: int = 20) -> pd.Series:
    feat = em_features(close)
    S = feat["S"]
    sig = pd.Series(0.5, index=S.index, dtype=float)
    # adaptiver Schwellenwert: k * roll. Std des Flusses
    sigm = S.rolling(vol_win, min_periods=5).std().replace(0, np.nan)
    th = k * sigm
    sig[(S >  th)] = 1.0   # Buy
    sig[(S < -th)] = 0.0   # Sell
    # Hold beibehaltung: 0.5 = kein Umschalten
    # in finale Position übersetzen: 1/0 halten bis neues hartes Signal kommt
    pos = np.zeros(len(sig))
    last = 0.5  # start neutral
    for i, s in enumerate(sig):
        if s == 0.5:
            pos[i] = last
        else:
            last = s
            pos[i] = last
    return pd.Series(pos, index=close.index, name="mx_pos")

def backtest(close: pd.Series, pos: pd.Series, fee_bps=1.0, slip_bps=2.0) -> pd.DataFrame:
    # diskrete Log-Returns
    r = pd.Series(np.r_[0.0, np.diff(np.log(close.astype(float) + 1e-12))], index=close.index)
    w = pos.shift(1).bfill()  # 1-bar Delay (zukunftssicher)


    # P&L: Gewicht * Return
    ret_gross = w * r
    # Kosten ∝ Turnover der gewichteten 1/0-Position
    turnover = (w - w.shift(1).fillna(0.5)).abs()
    cost = turnover * (fee_bps + slip_bps) / 1e4
    ret_net = ret_gross - cost
    eq = (1.0 + ret_net).cumprod()
    out = pd.DataFrame({"ret_net":ret_net, "equity":eq, "turnover":turnover})
    return out

def metrics(ret_net: pd.Series) -> dict:
    r = ret_net.dropna().to_numpy(float)
    if r.size < 10: return {k: np.nan for k in ["CAGR","Sharpe","MaxDD","Hit"]}
    eq = np.cumprod(1+r)
    cagr = eq[-1]**(252/len(r)) - 1
    mu = r.mean()*252
    sigma = r.std(ddof=1)*np.sqrt(252)
    sharpe = mu/(sigma+1e-12)
    rollmax = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq/rollmax - 1.0))
    hit = float((r>0).mean())
    return {"CAGR":cagr, "Sharpe":sharpe, "MaxDD":maxdd, "Hit":hit}

def demo_series(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.bdate_range("2019-01-01", periods=n, freq="C")
    # kleines Regime-Spiel: Abschnitte mit unterschiedl. Drift/Vol
    blocks = [(0.08,0.18),(0.00,0.25),(0.15,0.12),(-0.05,0.28)]
    per = n//len(blocks)
    rr = []
    for mu_ann, vol_ann in blocks:
        mu = mu_ann/252; vol = vol_ann/np.sqrt(252)
        rr.append(mu + vol*rng.standard_normal(per))
    r = np.concatenate(rr)[:n]
    price = 100*np.cumprod(1+r)
    return pd.Series(price, index=ts, name="close")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--price-col", type=str, default="close")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--vol-win", type=int, default=20)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.demo:
        close = demo_series()
    elif args.csv:
        df = pd.read_csv(args.csv)
        if "timestamp" in df.columns:
            df["timestamp"]=pd.to_datetime(df["timestamp"]); df=df.set_index("timestamp")
        close = pd.Series(df[args.price_col].values, index=pd.to_datetime(df.index), name="close").dropna()
    else:
        raise SystemExit("Nutze --demo oder --csv pfad.csv")

    pos = maxwell_agent(close, k=args.k, vol_win=args.vol_win)
    bt  = backtest(close, pos, fee_bps=args.fee_bps, slip_bps=args.slip_bps)
    M   = metrics(bt["ret_net"])
    print("== Maxwell-Flux Agent ==")
    for k,v in M.items(): print(f"{k:>7}: {v: .4f}")
    print(f" FinalEq: {bt['equity'].iloc[-1]:.4f}  Bars: {len(bt)}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            feat = em_features(close)
            fig, ax = plt.subplots(3,1, figsize=(11,8), sharex=True)
            ax[0].plot(close.index, close.values); ax[0].set_title("Price")
            ax[1].plot(feat.index, feat["S"], label="S (flux)"); ax[1].legend(); ax[1].axhline(0,color="k",lw=0.7)
            ax[2].plot(bt.index, bt["equity"], label="Equity (net)"); ax[2].legend(); ax[2].axhline(1,color="k",lw=0.7)
            plt.tight_layout(); plt.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()
