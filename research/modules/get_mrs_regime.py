import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ---------- utilities ----------
def _hday_log_returns(price: pd.Series, h: int) -> pd.Series:
    """y_t = log(P_{t+h}) - log(P_t), indexed at t (predict at t for t→t+h)."""
    p = np.log(price).dropna()
    y = p.shift(-h) - p
    return y.dropna()

def enforce_min_duration(reg: pd.Series, min_bars: int = 4) -> pd.Series:
    r = reg.copy()
    start = 0
    for i in range(1, len(r)):
        if r.iloc[i] != r.iloc[i-1]:
            if (i - start) < min_bars:
                r.iloc[start:i] = r.iloc[start]
            start = i
    return r

def _hard_labels_from_probs(probs: pd.DataFrame, p_thresh: float) -> pd.Series:
    hard = probs.idxmax(axis=1)
    maxp = probs.max(axis=1)
    hard[maxp < p_thresh] = np.nan
    return hard.ffill().dropna()

def _variance_order_mapping(y: pd.Series, hard: pd.Series) -> Dict[int, int]:
    tmp = pd.DataFrame({"y": y.reindex(hard.index), "reg": hard})
    order = tmp.groupby("reg")["y"].var().sort_values().index.tolist()
    return {old: new for new, old in enumerate(order)}

def shift_for_live_use(reg: pd.Series, shift_one: bool = True) -> pd.Series:
    """Make a daily 'live' label (e.g., to trade at close of t using info up to t-1)."""
    if shift_one:
        reg = reg.shift(1).ffill()
    reg.name = (reg.name or "regime") + "_live"
    return reg

# ---------- config & output ----------
@dataclass
class MRSHorizonConfig:
    k_regimes: int = 3
    trend: str = "c"
    switching_variance: bool = True
    p_thresh: float = 0.7
    min_bars: int = 4          # persistence measured in model observation steps (daily in this setup)

@dataclass
class MRSHorizonOutput:
    # train
    res_train: object
    mapping_old_to_new: Dict[int, int]
    y_train: pd.Series
    probs_train: pd.DataFrame
    reg_train: pd.Series
    reg_train_live: pd.Series
    # test (predicted, real-time)
    y_test: Optional[pd.Series]
    probs_test_pred: Optional[pd.DataFrame]
    reg_test_pred: Optional[pd.Series]
    reg_test_pred_live: Optional[pd.Series]

# ---------- main pipeline ----------
def train_test_regimes_horizon(price: pd.Series,
                               train_end: pd.Timestamp,
                               horizon_days: int,
                               cfg: MRSHorizonConfig = MRSHorizonConfig()
                               ) -> MRSHorizonOutput:
    """
    1) Build h-day returns y_t = log P_{t+h} - log P_t.
    2) Fit MarkovRegression on TRAIN y_t.
    3) TRAIN regimes: smoothed -> hard labels -> variance-ordered -> persistence.
    4) TEST regimes: run filter with TRAIN params; use one-step-ahead predicted probs; label as above.
    Notes:
      - y_t is indexed at t: it's the future h-day return you want to predict at time t.
      - 'predicted' probs at t use info up to t-1, so they're live-safe for forecasting y_t.
    """
    price = price.sort_index()
    y_all = _hday_log_returns(price, horizon_days).sort_index()

    # split by date on the y-series index (prediction time t)
    y_train = y_all.loc[:train_end]
    y_test  = y_all.loc[train_end + pd.Timedelta(days=1):]

    # --- Fit on TRAIN ---
    mod = MarkovRegression(y_train, k_regimes=cfg.k_regimes,
                           trend=cfg.trend, switching_variance=cfg.switching_variance)
    res_train = mod.fit(disp=False)

    # TRAIN probs (smoothed for clean in-sample classification)
    probs_train = res_train.smoothed_marginal_probabilities
    probs_train.index = y_train.index

    # Hard labels + persistence
    hard_train = _hard_labels_from_probs(probs_train, cfg.p_thresh)
    hard_train = enforce_min_duration(hard_train, cfg.min_bars)

    # Map regimes by variance (0=calm ... 2=crisis), locked from TRAIN
    mapping = _variance_order_mapping(y_train, hard_train)
    reg_train = hard_train.map(mapping).astype("Int64")
    reg_train.name = f"regime_mrs_h{horizon_days}"

    # Live-shifted train label (optional convenience)
    reg_train_live = shift_for_live_use(reg_train, shift_one=True)

    # --- TEST (optional) with fixed params, predicted probs (t uses info up to t-1) ---
    probs_test_pred = reg_test_pred = reg_test_pred_live = None
    if len(y_test) > 0:
        mod_test = MarkovRegression(y_test,
                                    k_regimes=res_train.model.k_regimes,
                                    trend=res_train.model.trend,
                                    switching_variance=res_train.model.switching_variance)
        res_test = mod_test.filter(res_train.params)

        probs_test_pred = res_test.predicted_marginal_probabilities
        probs_test_pred.index = y_test.index

        hard_test = _hard_labels_from_probs(probs_test_pred, cfg.p_thresh)
        hard_test = enforce_min_duration(hard_test, cfg.min_bars)
        reg_test_pred = hard_test.map(mapping).astype("Int64")
        reg_test_pred.name = f"regime_mrs_h{horizon_days}_pred"

        reg_test_pred_live = shift_for_live_use(reg_test_pred, shift_one=True)

    return MRSHorizonOutput(
        res_train=res_train,
        mapping_old_to_new=mapping,
        y_train=y_train,
        probs_train=pd.DataFrame(probs_train),
        reg_train=reg_train,
        reg_train_live=reg_train_live,
        y_test=y_test if len(y_test) > 0 else None,
        probs_test_pred=pd.DataFrame(probs_test_pred) if probs_test_pred is not None else None,
        reg_test_pred=reg_test_pred,
        reg_test_pred_live=reg_test_pred_live
    )

# ---------- optional: pure forward projection (no future data) ----------
def project_regimes_forward(last_probs, transition_matrix, steps: int):
    import numpy as np
    import pandas as pd

    p = np.asarray(last_probs, dtype=float).reshape(-1)          # (k,)
    T = np.asarray(transition_matrix, dtype=float)               # (k,k)

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"T must be square, got {T.shape}")
    k = T.shape[0]
    if p.size != k:
        raise ValueError(f"last_probs length {p.size} != T dimension {k}")

    probs = []
    for _ in range(steps):
        p = p @ T                                               # row vec × left-stochastic T
        probs.append(p.copy())

    return pd.DataFrame(probs, index=pd.RangeIndex(1, steps+1, name="h"))


