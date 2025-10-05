import numpy as np
import pandas as pd # type: ignore
from arch import arch_model # type: ignore

def label_regimes(price: pd.Series,
                  method: str = "realized",
                  window: int = 20,
                  low_q: float = 0.5,
                  high_q: float = 0.7,
                  dist: str = "t",
                  annualize: bool = True,
                  return_vol: bool = False) -> pd.Series:
    """
    Label calm/volatile regimes based on realized or conditional volatility.

    Parameters
    ----------
    price : pd.Series
        Price series (datetime index).
    method : {"realized","garch","egarch","gjr-garch"}
        Volatility estimation method.
    window : int
        Window length for realized volatility (ignored if GARCH).
    low_q, high_q : float
        Quantile thresholds for hysteresis (CALM<low, VOL>high).
    dist : str
        Distribution for ARCH model errors ("normal","t","skewt").
    annualize : bool
        Whether to scale daily vols to annualized (sqrt(252)).

    Returns
    -------
    regimes : pd.Series
        CALM/VOL regime labels, indexed like price.
    """

    # Compute returns (% for arch)
    ret_d = np.log(price / price.shift(1)).dropna() * 100

    if method == "realized":
        # rolling realized vol
        vol = ret_d.rolling(window).std()
    else:
        # pick model
        method = method.lower()
        if method == "garch":
            am = arch_model(ret_d, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
        elif method == "egarch":
            am = arch_model(ret_d, mean="Zero", vol="EGARCH", p=1, q=1, dist=dist)
        elif method == "gjr-garch":
            am = arch_model(ret_d, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
        else:
            raise ValueError(f"Unknown method: {method}")

        res = am.fit(disp="off")
        vol = res.conditional_volatility
        vol = pd.Series(vol, index=ret_d.index)

    # Convert percent back to decimal
    vol = vol / 100

    # Annualize if desired
    if annualize:
        vol = vol * np.sqrt(252)

    # shift(1): regime at t decided by vol up to t-1
    vol_ref = vol.shift(1).reindex(price.index)

    # Thresholds
    low_thr  = vol_ref.quantile(low_q)
    high_thr = vol_ref.quantile(high_q)

    # Hysteresis regime labeling
    # regimes = pd.Series(index=price.index, dtype="object")
    regimes = pd.qcut(vol_ref, q=2, labels=["CALM","VOL"])  # 2-state regime
    state = "CALM"
    
    # print(regime)
    for t, v in vol_ref.items():
        if pd.isna(v):
            regimes.loc[t] = np.nan
            continue
        if state == "CALM" and v >= high_thr:
            state = "VOL"
        elif state == "VOL" and v <= low_thr:
            state = "CALM"
        regimes.loc[t] = state

    regimes.name = f"regime_{method}"

    return (regimes, vol_ref) if return_vol else regimes


def label_regimes_persistent(
    price: pd.Series,
    method: str = "realized",          # "realized","garch","egarch","gjr-garch"
    window: int = 20,                   # realized vol window
    dist: str = "t",
    annualize: bool = True,

    # --- persistence controls ---
    ewma_halflife: int | None = 20,     # None to disable smoothing
    roll_q_win: int | None = 252*3,     # rolling window (e.g., 3y) for adaptive thresholds; None = global
    low_q: float = 0.40,                # hysteresis lower quantile
    high_q: float = 0.80,               # hysteresis upper quantile
    consec_k: int = 5,                  # require k consecutive days beyond threshold to switch
    min_duration: int = 20,             # minimum regime length in days
    use_garch_forecast: bool = True,    # for GARCH: use 1-step-ahead forecast var (preferred)
    return_vol: bool = False
):
    """
    Returns a regime series ('CALM'/'VOL') with persistence controls and the vol measure (optional).
    Regime at t is based only on info available up to t-1 (no lookahead).
    """
    price = price.dropna().astype(float)
    idx = price.index

    # Daily log returns in percent for arch
    ret_pct = np.log(price / price.shift(1)) * 100.0
    ret_pct = ret_pct.dropna()

    # --- Vol measure ---
    m = method.lower()
    if m == "realized":
        vol = ret_pct.rolling(window).std()
    else:
        if m == "garch":
            am = arch_model(ret_pct, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
        elif m == "egarch":
            am = arch_model(ret_pct, mean="Zero", vol="EGARCH", p=1, q=1, dist=dist)
        elif m in ("gjr", "gjr-garch"):
            am = arch_model(ret_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=dist)
        else:
            raise ValueError(f"Unknown method: {method}")

        res = am.fit(disp="off")
        if use_garch_forecast:
            H = 10  # your prediction horizon

            # IMPORTANT: start=0 to get in-sample rolling forecasts at every t
            fvar = res.forecast(horizon=H, start=0, reindex=False).variance
            # variance of the SUM of next H daily returns: r_{t+1}+...+r_{t+H}
            var_H = fvar.cumsum(axis=1).iloc[:, H-1]     # a Series, index = ret_pct.index

            vol = np.sqrt(var_H) / 100.0                # %^2 -> decimal H-day vol
        else:
            vol = res.conditional_volatility / 100.0    # 1-day vol (decimal)
            vol.index = ret_pct.index

    # Align to price index, shift so decision at t uses info up to t-1
    vol = vol.reindex(idx).shift(1)

    # If you annualize, remember this is H-day vol; annualization depends on how you’ll use it.
    # For features/regimes, you can skip annualization or scale as needed.
    # convert percent to decimal & annualize
    vol = (vol / 100.0)
    if annualize:
        vol = vol * np.sqrt(252)

    # align to price index and ensure decision uses info up to t-1
    vol = vol.reindex(idx)
    vol_ref = vol.shift(1)

    # --- smoothing (optional) ---
    if ewma_halflife is not None and ewma_halflife > 0:
        vol_ref = vol_ref.ewm(halflife=ewma_halflife, adjust=False, min_periods=1).mean()
    # alternative/additional: rolling median for outlier-robust smoothing
    # vol_ref = vol_ref.rolling(5, min_periods=1, center=False).median()

    # --- thresholds (global or rolling/adaptive) ---
    if roll_q_win:
        low_thr  = vol_ref.rolling(roll_q_win, min_periods=20).quantile(low_q)
        high_thr = vol_ref.rolling(roll_q_win, min_periods=20).quantile(high_q)
    else:
        low_thr = pd.Series(vol_ref.quantile(low_q), index=vol_ref.index)
        high_thr = pd.Series(vol_ref.quantile(high_q), index=vol_ref.index)

    # --- regime labeling with hysteresis + consecutive-days + min-duration ---
    regimes = pd.Series(index=idx, dtype="object")
    state = "CALM"
    last_switch_pos = 0
    streak = 0

    for i, t in enumerate(idx):
        v = vol_ref.iloc[i]
        lo = low_thr.iloc[i]
        hi = high_thr.iloc[i]

        if pd.isna(v) or pd.isna(lo) or pd.isna(hi):
            regimes.iloc[i] = np.nan
            continue

        # desired state from thresholds
        desired = state
        if state == "CALM":
            if v >= hi:
                streak += 1
                if streak >= consec_k and (i - last_switch_pos) >= min_duration:
                    desired = "VOL"
            else:
                streak = 0
        else:  # state == "VOL"
            if v <= lo:
                streak += 1
                if streak >= consec_k and (i - last_switch_pos) >= min_duration:
                    desired = "CALM"
            else:
                streak = 0

        if desired != state:
            state = desired
            last_switch_pos = i
            streak = 0

        regimes.iloc[i] = state

    regimes.name = f"regime_{method.lower()}"
    return (regimes, vol_ref) if return_vol else regimes


import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def fit_mrs(price: pd.Series, k_regimes: int = 3,
            trend: str = "c", switching_variance: bool = True):
    """
    Fit a Markov Regime Switching model with k_regimes states.
    
    Parameters
    ----------
    price : pd.Series
        Price series with datetime index.
    k_regimes : int
        Number of regimes (default 3).
    trend : str
        Trend in the regression ('c' = constant).
    switching_variance : bool
        Whether variances differ by regime.
    
    Returns
    -------
    res : fitted MarkovRegression result
    smoothed_probs : pd.DataFrame of smoothed regime probabilities
    """
    # compute daily returns
    ret = np.log(price / price.shift(1)).dropna()
    
    # fit MRS
    mod = MarkovRegression(ret, k_regimes=k_regimes,
                           trend=trend, switching_variance=switching_variance)
    res = mod.fit(disp=False)
    
    # smoothed regime probabilities
    smoothed_probs = res.smoothed_marginal_probabilities
    smoothed_probs.index = ret.index
    
    return res, smoothed_probs


import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def fit_mrs_weekly(price: pd.Series,
                   k_regimes: int = 3,
                   week_close: str = "W-FRI",     # weekly bar
                   trend: str = "c",
                   switching_variance: bool = True):
    """
    Fit a k-regime Markov switching model on WEEKLY log returns.
    Returns: (res, smoothed_probs (weekly), weekly_returns)
    """
    price_w = price.resample(week_close).last().dropna()
    ret_w = np.log(price_w).diff().dropna()           # weekly log returns

    mod = MarkovRegression(ret_w, k_regimes=k_regimes,
                           trend=trend, switching_variance=switching_variance)
    res = mod.fit(disp=False)

    probs_w = res.smoothed_marginal_probabilities
    probs_w.index = ret_w.index
    return res, probs_w, ret_w


def enforce_min_duration(reg: pd.Series, min_bars: int = 4) -> pd.Series:
    """Merge runs shorter than min_bars into the previous regime."""
    r = reg.copy()
    start = 0
    for i in range(1, len(r)):
        if r.iloc[i] != r.iloc[i-1]:
            if (i - start) < min_bars:
                r.iloc[start:i] = r.iloc[start]
            start = i
    return r

def weekly_regimes(price: pd.Series, p_thresh: float = 0.7, min_weeks: int = 4):
    res, probs_w, ret_w = fit_mrs_weekly(price)

    # hard labels with optional probability filter
    hard = probs_w.idxmax(axis=1)
    maxp = probs_w.max(axis=1)
    hard[maxp < p_thresh] = np.nan
    hard = hard.ffill().dropna()

    # persistence on weekly scale
    hard = enforce_min_duration(hard, min_bars=min_weeks)

    # map regimes by realized variance to get (0=calm, 1=normal, 2=crisis)
    tmp = pd.DataFrame({"r": ret_w.reindex(hard.index), "reg": hard})
    order = tmp.groupby("reg")["r"].var().sort_values().index.tolist()
    mapping = {old:new for new, old in enumerate(order)}
    reg_weekly = hard.map(mapping)
    reg_weekly.name = "regime_mrs_weekly"

    return res, probs_w, reg_weekly

def to_daily_regime(reg_weekly: pd.Series, price: pd.Series, shift_one: bool = True):
    reg_daily = reg_weekly.reindex(price.resample("D").last().index).ffill()
    reg_daily = reg_daily.reindex(price.index).ffill()
    if shift_one:
        reg_daily = reg_daily.shift(1).ffill()
    reg_daily.name = "regime_mrs_daily"
    return reg_daily
