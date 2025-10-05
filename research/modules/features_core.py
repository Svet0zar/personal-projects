import numpy as np
import pandas as pd

def make_horizon_target(price: pd.Series, 
                        horizon: int = 5,
                        kind: str = "log") -> pd.Series:
    """
    Compute forward H-horizon return aligned at time t.

    Parameters
    ----------
    price : pd.Series
        Price series.
    horizon : int
        Forward horizon in periods (e.g. trading days).
    kind : {'log','pct'}
        Return type:
        - 'log' : log(P_{t+h} / P_t)
        - 'pct' : (P_{t+h} - P_t) / P_t

    Returns
    -------
    pd.Series
        Forward return series, named 'ret_h{horizon}'.
    """
    if kind == "log":
        r = np.log(price.shift(-horizon) / price)
    elif kind == "pct":
        r = price.shift(-horizon).div(price).sub(1.0)
    else:
        raise ValueError("kind must be 'log' or 'pct'")
    return r.rename(f"ret_h{horizon}")


def make_supervised_df_horizon(df: pd.DataFrame,
                               price_col: str = "Matif_Prices",
                               horizon: int = 5,
                               return_kind: str = "log") -> pd.DataFrame:
    """
    Build a supervised learning DataFrame with target and price/macro features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain price_col and optionally macro columns.
    price_col : str
        Column name for the price series.
    horizon : int
        Horizon for forward return target.
    return_kind : {'log','pct'}
        Return type for target.

    Returns
    -------
    pd.DataFrame
        Supervised dataset with target, price-based features, macro deltas, and calendar dummies.
    """
    out = pd.DataFrame(index=df.index)

    # --- target ---
    price = df[price_col].astype(float)
    out[f"ret_h{horizon}"] = make_horizon_target(price, horizon=horizon, kind=return_kind)

    # --- daily log returns (base series for features) ---
    ret_1d = np.log(price / price.shift(1))

    # --- lagged returns ---
    for L in [1, 2, 3, 5, 10, 20]:
        out[f"ret_lag_{L}"] = ret_1d.shift(L)

    # --- rolling stats of daily returns ---
    for W in [5, 10, 20, 60]:
        out[f"ret_rollmean_{W}"] = ret_1d.shift(1).rolling(W, min_periods=W).mean()
        out[f"ret_rollstd_{W}"]  = ret_1d.shift(1).rolling(W, min_periods=W).std(ddof=0)

    # --- momentum (moving average gap) ---
    ma_fast = price.shift(1).rolling(10, min_periods=10).mean()
    ma_slow = price.shift(1).rolling(30, min_periods=30).mean()
    out["ma_gap_10_30"] = (ma_fast - ma_slow) / ma_slow

    # --- macro deltas ---
    macro_cols = [c for c in df.columns if c not in {price_col}]
    for c in macro_cols:
        s = df[c]
        out[f"{c}_d1"] = s.diff(1).shift(1)
        out[f"{c}_d5"] = s.diff(5).shift(1)

    # --- calendar features ---
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["quarter"] = out.index.quarter
    out = pd.get_dummies(out, columns=["dow", "month", "quarter"], drop_first=True)

    return out.dropna()
