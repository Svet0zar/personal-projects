import pandas as pd # type: ignore
import exchange_calendars as xcals # type: ignore

def preprocess_prices_lgbm(
    df: pd.DataFrame,
    date_col: str = "Period",
    target_col: str = "Matif_Prices",
    dailyize: bool = True,
    calendar: str | None = None,  # e.g., "XPAR" (Euronext Paris) via exchange_calendars
    ffill_limit: int | None = None,  # e.g., 10 for 2 weeks; None = unlimited
    release_lags: dict[str, int] | None = None,  # e.g., {"GDP_deflator": 30, "Oil_Average_Prices": 1}
) -> tuple[pd.DataFrame, dict]:
    """
    - Parse dates, sort, drop dup timestamps (keep last).
    - Coerce all non-date columns to numeric.
    - Optionally shift columns by publication lag (days) to prevent look-ahead.
    - Reindex to trading days (if calendar given) or generic business days.
    - Forward-fill exogenous features; DO NOT fill the target.
    - Return diagnostics.
    """

    out = {}

    # 1) Basic index prep
    df = df.rename(columns={date_col: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # 2) De-dup
    dup_count = df.index.duplicated(keep=False).sum()
    out["duplicate_timestamps"] = int(dup_count)
    df = df[~df.index.duplicated(keep="last")]

    # 3) Numeric coercion (preserve column order)
    non_date_cols = [c for c in df.columns if c != "date"]
    df[non_date_cols] = df[non_date_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    # Optionally drop all-NaN cols
    all_na = [c for c in df.columns if df[c].isna().all()]
    out["all_na_columns"] = all_na
    if all_na:
        df = df.drop(columns=all_na)

    # 4) Apply release lags (days) to *features only*
    if release_lags:
        for col, lag in release_lags.items():
            if col in df.columns and col != target_col:
                df[col] = df[col].shift(lag)

    # 5) Build the target separately (no ffill on target)
    y = None
    if target_col in df.columns:
        y = df[[target_col]].copy()

    # 6) Choose output index
    if dailyize:
        if calendar:
            # Use exchange calendar if available
            try:
                cal = xcals.get_calendar(calendar)
                # trading sessions only within data span
                start, end = df.index.min().date(), df.index.max().date()
                sessions = cal.sessions_in_range(str(start), str(end))
                idx = pd.DatetimeIndex(sessions.tz_localize(None))
            except Exception:
                # Fallback to generic business days
                idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
        else:
            idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    else:
        # Keep native frequency
        idx = df.index

    # 7) Reindex and controlled forward-fill for features (not target)
    feat_cols = [c for c in df.columns if c != target_col]
    feats = df[feat_cols].reindex(idx)
    # Create fill masks BEFORE filling
    fill_masks = {c: feats[c].isna() for c in feat_cols}
    feats = feats.ffill(limit=ffill_limit)

    # 8) Recombine target on the same index (no ffill)
    if y is not None:
        y = y.reindex(idx)

    # 9) Optionally drop rows with missing target (common at start)
    df_out = pd.concat([feats, y], axis=1)

    # 10) Diagnostics
    out["rows"] = int(len(df_out))
    out["start"] = df_out.index.min()
    out["end"] = df_out.index.max()
    out["missing_after_fill"] = {
        c: int(df_out[c].isna().sum()) for c in df_out.columns
    }
    out["ffill_limit"] = ffill_limit
    out["calendar_used"] = calendar if dailyize else "native"

    return df_out, out


def preprocess_prices(df: pd.DataFrame, date_col: str = "Period") -> pd.DataFrame:
    """
    Clean raw price dataframe:
    - Rename date column and set as index
    - Ensure datetime index, sorted ascending
    - Drop duplicate timestamps (keep last)
    - Reindex to business-day frequency
    - Forward-fill missing values
    """

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Check for duplicates
    dup_mask = df.index.duplicated(keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        print(f"Duplicate timestamps found: {dup_count}. Keeping last occurrence.")

    df = df[~df.index.duplicated(keep="last")]

    # Business day frequency, forward-fill
    df = df.asfreq("B").ffill()

    return df

def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and clean raw OHLCV CSV (e.g. from Investing.com):
    - Rename columns: 'Price'→'Close', 'Vol.'→'Volume', 'Date'→'date'
    - Parse 'date' as datetime index (ascending)
    - Convert 'Volume' from strings with suffixes (e.g. '12.5K') to floats
    """
    # Rename columns consistently
    df = df.rename(columns={
        "Price": "Close",
        "Vol.": "Volume",
        "Date": "date"
    })

    # Date index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Volume conversion
    def convert_volume(x):
        if pd.isna(x) or x == "":
            return 0.0
        x = str(x).replace(",", "").strip().upper()
        if x.endswith("K"):
            return float(x[:-1]) * 1_000
        if x.endswith("M"):
            return float(x[:-1]) * 1_000_000
        return float(x)

    df["Volume"] = df["Volume"].apply(convert_volume)

    return df
