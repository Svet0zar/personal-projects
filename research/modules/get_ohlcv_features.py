import numpy as np
import pandas as pd # type: ignore

def make_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create OHLCV-derived features, aligned so that row t uses info only up to t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['Open','High','Low','Close','Volume'].
    h : int
        Horizon (not used directly here, but useful if you want horizon-specific tuning).

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed like df, shifted to avoid leakage.
    """
    out = pd.DataFrame(index=df.index)

    # --- Parkinson volatility (log-range based) ---
    hl2 = np.log(df["High"] / df["Low"]).pow(2)
    out["parkinson_20"] = hl2.rolling(20, min_periods=10).mean().pow(0.5)

    # --- ATR (normalized by Close) ---
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = (tr.rolling(14, min_periods=7).mean() / df["Close"]).replace([np.inf, -np.inf], np.nan)

    # --- Volume surprise (z-score over 20d window) ---
    vol = df["Volume"].astype(float)
    vol = vol.replace(0, np.nan).ffill(limit=3)  # smooth out short zero runs
    vmean = vol.rolling(20, min_periods=10).mean()
    vstd  = vol.rolling(20, min_periods=10).std().replace(0, np.nan)
    out["vol_z20"] = ((vol - vmean) / vstd).fillna(0.0)

    # --- Candlestick body-to-range ratio ---
    body  = (df["Close"] - df["Open"]).abs()
    rng   = (df["High"] - df["Low"]).abs()
    out["body_to_range"] = (body / rng.replace(0, np.nan)).clip(0, 5).fillna(0.0)

    # Align: shift all features so row t uses info up to t-1
    out = out.shift(1)

    return out.dropna()


def _sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # prices must be > 0
    for c in ["Open","High","Low","Close"]:
        df.loc[df[c] <= 0, c] = np.nan
    # drop/sanitize rows where High < Low
    bad = df["High"] < df["Low"]
    df.loc[bad, ["High","Low"]] = np.nan
    return df

def make_ohlcv_features_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    df = _sanitize_ohlc(df)

    out = pd.DataFrame(index=df.index)

    # returns
    r_cc1 = np.log(df["Close"] / df["Close"].shift(1))
    out["r_cc1"] = r_cc1
    out["r_oo1"] = np.log(df["Open"]  / df["Close"].shift(1))
    out["r_oc1"] = np.log(df["Close"] / df["Open"])

    # Parkinson (clip before sqrt)
    hl2 = np.log(df["High"] / df["Low"])**2
    var_pk = hl2.rolling(20, min_periods=10).mean() / (4*np.log(2))
    out["parkinson_20"] = np.sqrt(var_pk.clip(lower=0))

    # Garman–Klass (clip before sqrt)
    co2 = np.log(df["Close"] / df["Open"])**2
    var_gk = (0.5*hl2 - (2*np.log(2)-1)*co2).rolling(20, min_periods=10).mean()
    out["gk_20"] = np.sqrt(var_gk.clip(lower=0))

    # Optional: Rogers–Satchell (variance always >= 0 for valid OHLC)
    u = np.log(df["High"]/df["Open"])
    d = np.log(df["Low"]/df["Open"])
    var_rs = (u*(u - np.log(df["Open"]/df["Close"])) + d*(d - np.log(df["Open"]/df["Close"]))).rolling(20, min_periods=10).mean()
    out["rs_20"] = np.sqrt(var_rs.clip(lower=0))

    # ATR (normalized)
    pc = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs()
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=7).mean() / df["Close"]

    # Volume surprise (robust)
    lv = np.log1p(df["Volume"])
    med20 = lv.rolling(20, min_periods=10).median()
    mad20 = (lv - med20).abs().rolling(20, min_periods=10).median()
    out["vol_surp20"] = ((lv - med20) / (1.4826*mad20)).clip(-5, 5)

    # Candle shape/position
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    out["body_to_range"] = ((df["Close"] - df["Open"]).abs() / rng).clip(0, 5).fillna(0.0)
    out["clv"] = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng).fillna(0.0)

    # Momentum & historical vol
    for w in [5, 10, 20]:
        out[f"mom_{w}"] = np.log(df["Close"] / df["Close"].shift(w))
        out[f"hv_{w}"]  = r_cc1.rolling(w, min_periods=int(0.8*w)).std()

    # Shift once to avoid look-ahead
    return out.shift(1).dropna()
