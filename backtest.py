import pandas as pd
import numpy as np
import utils
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("./processed_data.parquet")
print(df)

import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import utils


# -----------------------------
# Helpers: interactions & pivots
# -----------------------------
def add_interactions(df, feature_cols, interactions, inter_names):
    """
    Add interaction columns a*b to df inplace.
    Returns updated DataFrame with main effects + interactions.
    """
    X_df = df[feature_cols].copy()
    for (a, b), name in zip(interactions, inter_names):
        if a in X_df.columns and b in X_df.columns:
            X_df[name] = X_df[a] * X_df[b]
    return X_df


def pivot_features(df, feature_cols, target_col):
    """Pivot long-form df into wide format dict of feature DataFrames + target DataFrame."""
    Xs = {f: df.pivot(index="datetime", columns="symbol", values=f) for f in feature_cols}
    Y = df.pivot(index="datetime", columns="symbol", values=target_col)
    return Xs, Y


def build_design_matrix(train, feature_cols, interactions, inter_names):
    """Build design matrix with intercept, main effects, and interactions."""
    X_df = add_interactions(train, feature_cols, interactions, inter_names)
    X_df = X_df.copy()
    X_df.insert(0, 'intercept', 1.0)
    return X_df.values, X_df.columns.tolist()


def fit_ols(Xmat, yvec):
    """Fit plain OLS via lstsq."""
    return np.linalg.lstsq(Xmat, yvec, rcond=None)[0]


def build_today_features(Xs, di, cols, feature_cols, interactions, inter_names, beta_cols):
    """Build today's design matrix for prediction (no standardization applied here)."""
    X_today_df = pd.DataFrame(index=cols, dtype=float)
    for f in feature_cols:
        X_today_df[f] = Xs[f].iloc[di].reindex(cols).values
    X_today_df = add_interactions(X_today_df, feature_cols, interactions, inter_names)
    X_today_df.insert(0, 'intercept', 1.0)
    return X_today_df[beta_cols].values


# -----------------------------
# Standardization utilities
# -----------------------------
def standardize_features(df, feature_cols, interactions, inter_names):
    """
    Standardize predictors in df.
    Returns standardized design matrix (with intercept), column names,
    and the (mean, std) dictionaries used for transformation.
    """
    X_df = add_interactions(df, feature_cols, interactions, inter_names)
    mu = X_df.mean()
    sigma = X_df.std(ddof=0).replace(0, 1.0)  # avoid div by 0
    X_std = (X_df - mu) / sigma
    X_std.insert(0, "intercept", 1.0)
    return X_std.values, ["intercept"] + list(X_df.columns), mu, sigma


def apply_standardization(X_df, mu, sigma, beta_cols):
    """
    Apply a given mean/std transform to new data (for prediction).
    """
    X_std = (X_df - mu) / sigma
    X_std.insert(0, "intercept", 1.0)
    return X_std[beta_cols].values


# -----------------------------
# Exponential weights for panels
# -----------------------------
def per_row_expo_weights_by_date(index, decay):
    """
    Given a MultiIndex (datetime, symbol) sorted ascending by datetime,
    return a 1D array of per-row weights where *dates* get exponential
    downweighting and all rows on the same date share the same weight.
    Newest date has weight 1.0, previous date has 'decay', etc.
    """
    if isinstance(index, pd.MultiIndex):
        dates = index.get_level_values(0)
    else:
        dates = index  # assume DatetimeIndex
    unique_dates = pd.Index(dates.unique())
    # map date -> position 0..n-1
    pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    # weight for a date d = decay ** (n-1 - pos[d]) so newest date -> 1.0
    ages = (len(unique_dates) - 1) - pos.reindex(dates).values
    w = np.power(decay, ages.astype(float))
    return w


# -----------------------------
# CV for decay selection
# -----------------------------
def _cv_train_test_split_dates(dates, n_splits, horizon):
    """
    Forward-chaining CV over unique dates with purging/embargo by 'horizon' steps.
    Yields (train_dates, test_dates) for each fold.
    - dates: pd.Index of unique sorted datetimes
    - n_splits: number of folds (blocks)
    - horizon: embargo in *steps* (dates)
    """
    if len(dates) < n_splits + 1:
        return  # yields nothing

    blocks = np.array_split(dates, n_splits)
    for b in range(1, n_splits):  # require at least one block before test
        test_dates = blocks[b]
        if len(test_dates) == 0:
            continue
        test_start = test_dates[0]
        before = dates[dates < test_start]
        embargo_n = min(horizon, len(before))
        train_dates = before[:-embargo_n] if embargo_n > 0 else before
        if len(train_dates) == 0:
            continue
        yield train_dates, test_dates


def _cv_score_for_decay(train_long_df,
                        feature_cols,
                        target_col,
                        interactions,
                        inter_names,
                        horizon,
                        n_splits_cv,
                        decay):
    """
    Compute average OOS R^2 over forward-chaining purged CV folds
    for a fixed 'decay'. Uses per-date exponential weights on TRAIN only.
    """
    if train_long_df.empty:
        return -np.inf

    # Ensure sorted by (datetime, symbol)
    if isinstance(train_long_df.index, pd.MultiIndex):
        train_long_df = train_long_df.sort_index(level=[0, 1])
    else:
        train_long_df = train_long_df.sort_index()

    dates = train_long_df.index.get_level_values(0).unique().sort_values()
    total_r2 = 0.0
    total_n = 0

    for tr_dates, te_dates in _cv_train_test_split_dates(dates, n_splits_cv, horizon):
        tr = train_long_df.loc[train_long_df.index.get_level_values(0).isin(tr_dates)]
        te = train_long_df.loc[train_long_df.index.get_level_values(0).isin(te_dates)]
        if tr.empty or te.empty:
            continue

        # Standardize on TRAIN only
        Xmat_tr, Xcols, mu, sigma = standardize_features(
            tr[feature_cols], feature_cols, interactions, inter_names
        )
        y_tr = tr[target_col].values

        # Per-date exponential weights on TRAIN
        w_tr = per_row_expo_weights_by_date(tr.index, decay)
        W = np.sqrt(w_tr)[:, None]
        Xw = Xmat_tr * W
        yw = y_tr * np.sqrt(w_tr)

        beta = fit_ols(Xw, yw)

        # Predict on TEST (unweighted, but standardized using TRAIN mu/sigma)
        X_te_df = add_interactions(te[feature_cols], feature_cols, interactions, inter_names)
        X_te = apply_standardization(X_te_df, mu, sigma, Xcols)
        y_te = te[target_col].values
        yhat = X_te @ beta

        if len(y_te) >= 2 and np.isfinite(y_te).all() and np.isfinite(yhat).all():
            r2 = r2_score(y_te, yhat)
            # Weight each fold by number of test observations
            total_r2 += r2 * len(y_te)
            total_n += len(y_te)

    if total_n == 0:
        return -np.inf
    return total_r2 / total_n


def _select_decay_via_cv(train_long_df,
                         feature_cols,
                         target_col,
                         interactions,
                         inter_names,
                         horizon,
                         n_splits_cv,
                         decay_grid):
    """
    Evaluate each decay in 'decay_grid' via CV and pick the argmax R^2.
    Returns (best_decay, best_score, scores_dict)
    """
    best_decay = None
    best_score = -np.inf
    scores = {}
    for d in decay_grid:
        score = _cv_score_for_decay(train_long_df, feature_cols, target_col,
                                    interactions, inter_names, horizon,
                                    n_splits_cv, d)
        scores[d] = score
        if score > best_score:
            best_score = score
            best_decay = d
    return best_decay, best_score, scores


# -----------------------------
# Main walk-forward function
# -----------------------------
def walkforward_cs_ols(df,
                       feature_cols,
                       lookback,
                       horizon,
                       target_col="target",
                       interactions=None,
                       window_type="rolling",   # "rolling" | "expanding" | "exponential"
                       decay=0.94,              # float | "cv"
                       decay_grid=None,         # iterable of floats in (0,1), used if decay == "cv" or provided
                       n_splits_cv=5,           # CV folds for decay selection
                       verbose=False,
                       refit_interval=10):      # New parameter for refit interval
    """
    Walk-forward cross-sectional OLS with proper feature standardization.
    If window_type == "exponential" and (decay == "cv" or decay_grid is provided),
    select the exponential decay via purged, forward-chaining CV on each date's
    training window.

    Returns
    -------
    F : pd.DataFrame
        Forecasts (index=datetime, columns=symbol).
    B : pd.DataFrame
        Betas per date (index=datetime, columns=['intercept', *features, *interactions]).
    CV_R2 : pd.Series
        If exponential with CV: best CV score per forecast date; else NaN.
    V : pd.DataFrame
        Per-date VIFs for predictors (main + interactions), NaN if not computable.
    ChosenDecay : pd.Series
        Chosen decay per forecast date (NaN if not exponential or not CV).
    """
    if interactions is None:
        interactions = []

    # --- Precompute pivots
    Xs, Y = pivot_features(df, feature_cols, target_col)
    idx, cols = Y.index, Y.columns

    inter_names = [f"{a}*{b}" for (a, b) in interactions]
    beta_cols = ["intercept"] + list(feature_cols) + inter_names
    vif_cols = list(feature_cols) + inter_names

    # outputs
    F = pd.DataFrame(index=idx, columns=cols, dtype=float)
    B = pd.DataFrame(index=idx, columns=beta_cols, dtype=float)
    CV_R2 = pd.Series(index=idx, dtype=float)
    V = pd.DataFrame(index=idx, columns=vif_cols, dtype=float)
    ChosenDecay = pd.Series(index=idx, dtype=float)

    # default decay grid if needed
    if window_type == "exponential" and (decay == "cv" or decay_grid is not None):
        if decay_grid is None:
            # Reasonable wide grid; adjust as needed
            decay_grid = np.concatenate([
                np.linspace(0.90, 0.99, 5, endpoint=True)
            ])
            print(decay_grid)

    # --- Walk forward
    for di in range(max(lookback, horizon), len(idx)):
        # Only refit the model every `refit_interval` days
        if di % refit_interval != 0:
            # Skip refitting and reuse the previous beta coefficients for prediction
            if di > 0:
                F.iloc[di] = F.iloc[di - 1]
                B.iloc[di] = B.iloc[di - 1]
                CV_R2.iloc[di] = CV_R2.iloc[di - 1]
                ChosenDecay.iloc[di] = ChosenDecay.iloc[di - 1]
                continue

        # Define training window
        if window_type == "rolling":
            start, end = di - lookback, di - horizon + 1
        elif window_type in ("expanding", "exponential"):
            start, end = 0, di - horizon + 1
        else:
            raise ValueError("window_type must be 'rolling', 'expanding', or 'exponential'")

        if end <= start:
            continue

        train_dates = idx[start:end]

        # Stack long-form training panel
        stacked_feats = [Xs[f].iloc[start:end].stack().rename(f) for f in feature_cols]
        ys = Y.iloc[start:end].stack().rename(target_col)
        train = pd.concat(stacked_feats + [ys], axis=1).dropna()
        if train.empty:
            continue

        # Ensure MultiIndex (datetime, symbol) sorted
        if isinstance(train.index, pd.MultiIndex):
            train = train.sort_index(level=[0, 1])
        else:
            train = train.sort_index()

        if verbose:
            print(f"Forecast date {idx[di].date()} "
                  f"→ training window {train_dates[0].date()} to {train_dates[-1].date()} "
                  f"(n={len(train)})")

        # -------------------------
        # Choose decay (if applicable)
        # -------------------------
        chosen_decay = None
        best_cv_score = np.nan

        if window_type == "exponential":
            if decay == "cv" or decay_grid is not None:
                chosen_decay, best_cv_score, _ = _select_decay_via_cv(
                    train_long_df=train,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    interactions=interactions,
                    inter_names=inter_names,
                    horizon=horizon,
                    n_splits_cv=n_splits_cv,
                    decay_grid=decay_grid
                )
                print(f"  Chosen decay: {chosen_decay} with CV R2: {best_cv_score}")
                # Fallback if CV failed (e.g., not enough folds)
                if chosen_decay is None:
                    chosen_decay = 0.94 if decay == "cv" else (decay_grid[-1] if len(decay_grid) else 0.94)
            else:
                chosen_decay = float(decay)

            ChosenDecay.iloc[di] = chosen_decay
            CV_R2.iloc[di] = best_cv_score
        else:
            CV_R2.iloc[di] = np.nan
            ChosenDecay.iloc[di] = np.nan

        # -------------------------
        # Standardize predictors (TRAIN)
        # -------------------------
        Xmat, Xcols, mu, sigma = standardize_features(
            train[feature_cols], feature_cols, interactions, inter_names
        )
        yvec = train[target_col].values

        # VIFs (unweighted) — safe guard if utils.calculate_vif signature differs
        try:
            # If your calculate_vif expects a DataFrame without intercept:
            V.loc[idx[di], vif_cols] = utils.calculate_vif(
                pd.DataFrame(Xmat[:, 1:], columns=vif_cols)
            )
        except Exception:
            # fallback to NaNs if VIF calculation unavailable
            V.loc[idx[di], vif_cols] = np.nan

        # -------------------------
        # Fit OLS (with optional exponential weights)
        # -------------------------
        if window_type == "exponential":
            w = per_row_expo_weights_by_date(train.index, chosen_decay)
            W = np.sqrt(w)[:, None]
            Xw = Xmat * W
            yw = yvec * np.sqrt(w)
            beta = fit_ols(Xw, yw)
        else:
            beta = fit_ols(Xmat, yvec)

        B.loc[idx[di], Xcols] = beta

        # -------------------------
        # Predict today's cross-section
        # -------------------------
        X_today_df = pd.DataFrame(index=cols, dtype=float)
        for f in feature_cols:
            X_today_df[f] = Xs[f].iloc[di].reindex(cols).values
        X_today_df = add_interactions(X_today_df, feature_cols, interactions, inter_names)

        X_today = apply_standardization(X_today_df, mu, sigma, Xcols)
        F.iloc[di] = X_today @ beta

    return F, B, CV_R2, V, ChosenDecay


# -----------------------------
# Example usage
# -----------------------------
F, B, CV_R2, V, ChosenDecay = walkforward_cs_ols(
    df,
    feature_cols=["bolmom", "dv_decile", "buy_volume_ratio"],
    interactions=[("bolmom", "dv_decile"), ("buy_volume_ratio", "dv_decile")],
    target_col="target",
    lookback=45,
    horizon=5,
    verbose=True,
    window_type="exponential",
    decay="cv",               # turn on per-date CV for decay
    # Optionally provide a custom grid:
    # decay_grid=np.linspace(0.93, 0.995, 14),
    n_splits_cv=5
)
