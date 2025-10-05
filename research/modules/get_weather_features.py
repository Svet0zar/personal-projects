import pandas as pd # type: ignore
import numpy as np

def get_weather_features(wx_all: pd.DataFrame, use_eod_cutoff: bool = True) -> pd.DataFrame:
    """
    Create daily and weekly weather features for crop modelling.
    
    Parameters
    ----------
    wx_all : pd.DataFrame
        Raw weather dataframe with columns including ['date','tmin','tmax','tmean','precip_mm'].
    use_eod_cutoff : bool
        If True, features at day t may include day-t values (EOD availability).
        If False, features are shifted by 1 to ensure only past info is used.
    
    Returns
    -------
    df_weather : daily weather features
    wk : weekly resampled features with lags
    """

    df_weather = wx_all.copy()
    df_weather = df_weather.drop(columns=[c for c in ["number","step","surface"] if c in df_weather.columns])

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_weather = df_weather.set_index("date").sort_index()

    shift = (lambda s: s) if use_eod_cutoff else (lambda s: s.shift(1))

    # Daily flags
    df_weather["rainday"]  = (df_weather["precip_mm"] >= 1).astype(int)
    df_weather["dryday"]   = 1 - df_weather["rainday"]
    df_weather["heat32"]   = (df_weather["tmax"] >= 32).astype(int)
    df_weather["heat35"]   = (df_weather["tmax"] >= 35).astype(int)
    df_weather["frost0"]   = (df_weather["tmin"] < 0).astype(int)
    df_weather["frostm5"]  = (df_weather["tmin"] <= -5).astype(int)
    df_weather["gdd5"]     = (df_weather["tmean"] - 5).clip(lower=0)

    # Rolling windows (14/30/60)
    for w in [14, 30, 60]:
        df_weather[f"r{w}_precip"]   = shift(df_weather["precip_mm"]).rolling(w, min_periods=w).sum()
        df_weather[f"r{w}_tmean"]    = shift(df_weather["tmean"]).rolling(w, min_periods=w).mean()
        df_weather[f"heat32_{w}"]    = shift(df_weather["heat32"]).rolling(w, min_periods=w).sum()
        df_weather[f"frost0_{w}"]    = shift(df_weather["frost0"]).rolling(w, min_periods=w).sum()
        df_weather[f"frostm5_{w}"]   = shift(df_weather["frostm5"]).rolling(w, min_periods=w).sum()
        df_weather[f"raindays_{w}"]  = shift(df_weather["rainday"]).rolling(w, min_periods=w).sum()
        df_weather[f"drydays_{w}"]   = shift(df_weather["dryday"]).rolling(w, min_periods=w).sum()

    # Current dry spell length
    is_dry = df_weather["dryday"]
    dry_run = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum() * is_dry
    df_weather["dry_spell_current"] = shift(dry_run)

    # Crop-year accumulators
    crop_year_oct = np.where(df_weather.index.month >= 10, df_weather.index.year + 1, df_weather.index.year)
    df_weather["precip_since_Oct1"] = shift(df_weather["precip_mm"]).groupby(crop_year_oct).cumsum()

    year_mar = np.where(df_weather.index.month >= 3, df_weather.index.year, df_weather.index.year - 1)
    df_weather["gdd5_since_Mar1"]   = shift(df_weather["gdd5"]).groupby(year_mar).cumsum()

    # Anomalies (vs. monthly climatology)
    clim_tmean = df_weather.groupby(df_weather.index.month)["tmean"].transform("mean")
    df_weather["tmean_month_anom"] = shift(df_weather["tmean"] - clim_tmean)
    clim_r30 = df_weather.groupby(df_weather.index.month)["r30_precip"].transform("mean")
    df_weather["r30_precip_anom"]  = shift(df_weather["r30_precip"] - clim_r30)

    # # Weekly resample
    # wk = (df_weather
    #       .resample("W-MON")
    #       .agg({
    #           "tmean":"mean","tmin":"mean","tmax":"mean",
    #           "precip_mm":"sum",
    #           "r14_precip":"last","r30_precip":"last","r60_precip":"last",
    #           "r14_tmean":"last","r30_tmean":"last",
    #           "gdd5_since_Mar1":"last","precip_since_Oct1":"last",
    #           "heat32_30":"last","frost0_30":"last",
    #           "dry_spell_current":"last",
    #           "tmean_month_anom":"last","r30_precip_anom":"last",
    #       }))
    # # Shift weekly values so they are only known at the start of the following week
    # wk = wk.shift(1)

    # # Weekly lags
    # for k in [1, 2, 4]:
    #     wk[f"r30_precip_lag{k}"] = wk["r30_precip"].shift(k)
    #     wk[f"r30_tmean_lag{k}"]  = wk["r30_tmean"].shift(k)

    return df_weather#, wk


def get_weather_features_lgbm(wx_all: pd.DataFrame,
                         use_eod_cutoff: bool = True,
                         baseline_years: tuple[int,int] | None = (1991, 2020)
                         ) -> pd.DataFrame:
    """
    Build daily weather features with proper timing.
    If use_eod_cutoff=False, every feature at day t only uses data <= t-1.
    """

    df = wx_all.copy()
    df = df.drop(columns=[c for c in ["number","step","surface"] if c in df.columns and c in df])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Hygiene
    for c in ["tmin","tmax","tmean","precip_mm"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["precip_mm"] = df["precip_mm"].clip(lower=0)

    # Decide timing: build features from base that is known at close of t
    # If not EOD, shift base by 1 day so all downstream calcs inherit timing.
    base = df.copy()
    if not use_eod_cutoff:
        base = base.shift(1)

    # Daily flags (now inherit timing consistently)
    base["rainday"] = (base["precip_mm"] >= 1).astype(int)
    base["dryday"]  = 1 - base["rainday"]
    base["heat32"]  = (base["tmax"] >= 32).astype(int)
    base["heat35"]  = (base["tmax"] >= 35).astype(int)
    base["frost0"]  = (base["tmin"] < 0).astype(int)
    base["frostm5"] = (base["tmin"] <= -5).astype(int)
    base["gdd5"]    = (base["tmean"] - 5).clip(lower=0)

    # Rolling windows (14/30/60) — min_periods keeps NaNs early (good)
    out = base.copy()
    for w in [14, 30, 60]:
        out[f"r{w}_precip"]  = base["precip_mm"].rolling(w, min_periods=w).sum()
        out[f"r{w}_tmean"]   = base["tmean"].rolling(w, min_periods=w).mean()
        out[f"heat32_{w}"]   = base["heat32"].rolling(w, min_periods=w).sum()
        out[f"frost0_{w}"]   = base["frost0"].rolling(w, min_periods=w).sum()
        out[f"frostm5_{w}"]  = base["frostm5"].rolling(w, min_periods=w).sum()
        out[f"raindays_{w}"] = base["rainday"].rolling(w, min_periods=w).sum()
        out[f"drydays_{w}"]  = base["dryday"].rolling(w, min_periods=w).sum()

    # Current dry spell length (known at t)
    is_dry = base["dryday"].fillna(0)
    dry_run = is_dry.groupby((is_dry.ne(is_dry.shift())).cumsum()).cumsum() * is_dry
    out["dry_spell_current"] = dry_run

    # Crop-year accumulators (Oct→Sep, and Mar→Feb)
    crop_year_oct = np.where(out.index.month >= 10, out.index.year + 1, out.index.year)
    out["precip_since_Oct1"] = base["precip_mm"].groupby(crop_year_oct).cumsum()

    year_mar = np.where(out.index.month >= 3, out.index.year, out.index.year - 1)
    out["gdd5_since_Mar1"] = base["gdd5"].groupby(year_mar).cumsum()

    # Climatology anomalies WITHOUT future info
    if baseline_years is not None:
        y0, y1 = baseline_years
        ref = df.loc[(df.index.year >= y0) & (df.index.year <= y1)].copy()
        # month means & std from baseline only
        m_mean = ref.groupby(ref.index.month)["tmean"].mean()
        m_std  = ref.groupby(ref.index.month)["tmean"].std(ddof=0)
        p_mean = ref.groupby(ref.index.month)["precip_mm"].rolling(30, min_periods=30).sum().reset_index(0,drop=True).groupby(ref.index.month).mean()
        p_std  = ref.groupby(ref.index.month)["precip_mm"].rolling(30, min_periods=30).sum().reset_index(0,drop=True).groupby(ref.index.month).std(ddof=0)

        out["tmean_month_anom"] = base["tmean"] - base.index.month.map(m_mean)
        out["tmean_month_z"]    = (base["tmean"] - base.index.month.map(m_mean)) / base.index.month.map(m_std)

        # r30 precip anomaly/z
        r30 = out["r30_precip"]
        out["r30_precip_anom"] = r30 - out.index.month.map(p_mean)
        out["r30_precip_z"]    = (r30 - out.index.month.map(p_mean)) / out.index.month.map(p_std)
    else:
        # Expanding-window alternative (leakage-safe, adapts to warming)
        # Month-by-month expanding mean/std up to t-1
        mm = base.groupby(base.index.month)["tmean"].expanding().mean().droplevel(0)
        ms = base.groupby(base.index.month)["tmean"].expanding().std(ddof=0).droplevel(0)
        out["tmean_month_anom"] = base["tmean"] - mm
        out["tmean_month_z"]    = (base["tmean"] - mm) / ms

        r30 = out["r30_precip"]
        r30_exp_mean = r30.groupby(r30.index.month).expanding().mean().droplevel(0)
        r30_exp_std  = r30.groupby(r30.index.month).expanding().std(ddof=0).droplevel(0)
        out["r30_precip_anom"] = r30 - r30_exp_mean
        out["r30_precip_z"]    = (r30 - r30_exp_mean) / r30_exp_std

    return out
