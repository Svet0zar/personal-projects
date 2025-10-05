import pandas as pd # type: ignore

def build_full_supervised(supervised_core: pd.DataFrame,
                          regime: pd.Series = None,
                          ohlcv_feats: pd.DataFrame = None,
                          weather_feats: pd.DataFrame = None,
                          dropna_target: bool = True,
                          target_col: str = None) -> pd.DataFrame:
    """
    Combine target/core features with optional regime, OHLCV, and weather features.

    Parameters
    ----------
    supervised_core : pd.DataFrame
        Core supervised dataset (must include target + basic price/macro features).
    regime : pd.Series, optional
        Regime labels (indexed by date).
    ohlcv_feats : pd.DataFrame, optional
        OHLCV-derived features (indexed by date).
    weather_feats : pd.DataFrame, optional
        Weather-derived features (daily or weekly, indexed by date).
    dropna_target : bool, default True
        Drop rows with missing target values.
    target_col : str, optional
        Explicit target column name (if not the default 'ret_h*').

    Returns
    -------
    pd.DataFrame
        Final aligned supervised dataset with all features merged.
    """
    out = supervised_core.copy()

    # --- join regime ---
    if regime is not None:
        out = out.join(regime.rename("regime"), how="inner")

    # --- join OHLCV features ---
    if ohlcv_feats is not None:
        out = out.join(ohlcv_feats, how="left")

    # --- join weather features ---
    if weather_feats is not None:
        out = out.join(weather_feats, how="left")

    # --- drop missing target if requested ---
    if dropna_target:
        if target_col is None:
            target_cols = [c for c in out.columns if c.startswith("ret_h")]
            if not target_cols:
                raise ValueError("No target column found in supervised_core. Please set target_col.")
            target_col = target_cols[0]
        out = out.dropna(subset=[target_col])

    return out
