# ── Parameters ──────────────────────────────────────────────────────────────
DATA_PATH = "C:/Users/itataurov/Desktop/ccxt_data/data_parsed/1d/technical/binance/joined/binance_perps_joined.parquet"
HORIZON = 10
UNIVERSE_MIN_DOLLAR_VOL = 1e6
UNIVERSE_MIN_LISTING_DAYS = 90
VOL_EXPANDING_MIN_PERIODS = 30
VOL_EWM_SPAN = 30
VOL_EWM_MIN_PERIODS = 30
VOL_COMBINE_EXPANDING_WEIGHT = 0.25
VOL_COMBINE_EWM_WEIGHT = 0.75
DOLLAR_VOL_EWM_SPAN = 60
DOLLAR_VOL_EWM_MIN_PERIODS = 30
MCAP_EWM_SPAN = 60
MCAP_EWM_MIN_PERIODS = 30
DV_DECILE_BINS = 10
MC_DECILE_BINS = 10
TARGET_LOOKBACK = 45
TARGET_HORIZON = HORIZON
CV_N_SPLITS = 5
CV_MIN_TRAIN_DAYS = 10
CV_EMBARGO_BY = "steps"
CV_INTERACTIONS = None
FEATURE_COLS = ["bolmom"]
TRADING_COST_RATE = 0
WEIGHTS_LAG = True
WEIGHTS_INERTIA = 0.25
WEIGHTS_L2_SHRINK = 0.0
VOL_TARGET_ANN = 0.2
VOL_TARGET_LOOKBACK = 180
VOL_TARGET_MIN_OBS = 90
PLOT_HIGHLIGHT_DATE = "2025-01-01"

# Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf   # falls back to sample cov below

path = DATA_PATH

df = pd.read_parquet(path)
df = df.sort_values(by=['datetime'])
first_idx = df['fundingRate'].first_valid_index() # important to see when stuff began trading
df = df.loc[first_idx:]
df = df[df.datetime>'2021-01-01']
df = df[df.datetime<'2023-01-01']

HORIZON = 10

import signals

# Signal generation
def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
	df = signals._compute_buy_volume_ratio(df, 'buy_volume_ratio', 8)
	df = signals._compute_bolmom(df, 'bolmom', 128)
	df = signals._compute_funding_volatility(df, 'funding_volatility', 2)
	df = signals._compute_ls_ratio(df, 'ls_ratio', 8)

	return df

# ── Universe filtering ──────────────────────────────────────────────────────
def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Filter out illiquids and newly listed coins.
	"""

	df = df[df.dollar_volume_slowed >= UNIVERSE_MIN_DOLLAR_VOL] # at least $1M daily volume 
	df = df[df.days_since_listing >= UNIVERSE_MIN_LISTING_DAYS] # at least 90 days since listing (need data for risk estimate)
	return df

# Data preparation
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
	df = df.sort_values(['datetime', 'symbol'])
	df['return'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change())

	# 1) Calculate volatility
	df['vol_expanding_window'] = (
		df.groupby('symbol')['return']
		  .transform(lambda x: x
					  .expanding(min_periods=VOL_EXPANDING_MIN_PERIODS)
					  .std())
	)
	df['vol_ewm'] = (
		df.groupby('symbol')['return']
		  .transform(lambda x: x
					  .ewm(span=VOL_EWM_SPAN, adjust=False, min_periods=VOL_EWM_MIN_PERIODS)
					  .std())
	)
	df['volatility'] = (
		df['vol_expanding_window'] * VOL_COMBINE_EXPANDING_WEIGHT +
		df['vol_ewm'] * VOL_COMBINE_EWM_WEIGHT
	)

	# 2) filter universe
	df['dollar_volume'] = df['close'] * df['volume']
	df['dollar_volume_slowed'] = (
		df.groupby('symbol')['dollar_volume']
		  .transform(lambda x: x.ewm(span=DOLLAR_VOL_EWM_SPAN, min_periods=DOLLAR_VOL_EWM_MIN_PERIODS, adjust=False).mean())
	)
	df['dv_rank'] = (
		df.groupby('datetime')['dollar_volume_slowed']
		  .rank(method='first', ascending=False)
	)
	df['dv_decile'] = df.groupby('datetime')['dollar_volume_slowed'].transform(lambda x: pd.qcut(x, DV_DECILE_BINS, labels=False, duplicates='drop')) + 1

	df['market_cap_slowed'] = (
	df.groupby('symbol')['mkt_cap']
		.transform(lambda x: x.ewm(span=MCAP_EWM_SPAN, min_periods=MCAP_EWM_MIN_PERIODS, adjust=False).mean())
	)
	df['mc_rank'] = (
		df.groupby('datetime')['market_cap_slowed']
			.rank(method='first', ascending=False)
	)
	df['mc_decile'] = df.groupby('datetime')['market_cap_slowed'].transform(lambda x: pd.qcut(x, MC_DECILE_BINS, labels=False, duplicates='drop')) + 1

	# 1) how many observations each symbol has accumulated up to *and incl.* the row
	df['days_since_listing'] = df.groupby('symbol').cumcount() + 1

	return df

# TODO: Add funding returns to target 
def add_prediction_target(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df = df.sort_values(['datetime'])
	# demean within each date
	df['target'] = df.groupby('datetime')['fwd_return'].transform(lambda x: x - x.mean())
	return df.drop(columns=['forward_price', 'fwd_return'])

# Caclculate before uni filtering
df['forward_price'] = df.groupby('symbol')['close'].shift(-HORIZON)
df['fwd_return'] = (df['forward_price'] - df['close']) / df['close']

print("Preparing data...")
df = prepare_data(df)

print("Computing signal...")
df = compute_signal(df)

print("Filtering universe...")
df = filter_universe(df)

print("Adding prediction target...")
df = add_prediction_target(df)

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import utils


def purged_blocked_cv_r2(
    train_df,                      # MultiIndex (datetime, symbol)
    feature_cols,                  # main-effect feature names
    target_col,
    horizon,
    interactions=None,             # list of tuples [('a','b'), ...]
    n_splits=5,
    min_train_days=10,
    embargo_by="steps"             # "steps" (bars) or "days"
):
    """
    Forward-chaining, purged CV that matches your production spec:
    - Standardize main effects on the fold's train split only
    - Build interactions as z(a)*z(b) on train and test
    - Embargo 'horizon' (by steps or days) before the test block
    - Fit OLS with intercept; score R^2 out-of-sample
    """
    if interactions is None:
        interactions = []
    inter_names = [f"{a}*{b}" for (a, b) in interactions]

    dates = train_df.index.get_level_values(0).unique().sort_values()
    if len(dates) < n_splits + 1:
        return float("nan")

    blocks = np.array_split(dates, n_splits)
    r2s = []

    for b in range(1, n_splits):  # need past to train on
        test_dates = blocks[b]
        test_start = test_dates[0]

        # Embargo: remove last 'horizon' worth of training dates
        if embargo_by == "days":
            cutoff = test_start - pd.Timedelta(days=horizon)
            train_dates = dates[dates < cutoff]
        else:  # "steps" == bars
            # all dates strictly before test_start
            before = dates[dates < test_start]
            embargo_n = min(horizon, len(before))
            train_dates = before[:-embargo_n] if embargo_n > 0 else before

        if len(train_dates) < min_train_days:
            continue

        tr = train_df.loc[train_dates]
        te = train_df.loc[test_dates]

        # --- Standardize main effects on TRAIN only ---
        mu = tr[feature_cols].mean()
        sig = tr[feature_cols].std().replace(0, np.nan)
        valid = sig.notna()

        Ztr = (tr[feature_cols] - mu) / sig
        Zte = (te[feature_cols] - mu) / sig
        Ztr = Ztr.loc[:, valid]
        Zte = Zte.loc[:, valid]

        # --- Build interactions as z(a)*z(b) ---
        if interactions:
            ZItr = pd.DataFrame(index=Ztr.index, dtype=float)
            ZIte = pd.DataFrame(index=Zte.index, dtype=float)
            kept_inters = []
            for (a, b), name in zip(interactions, inter_names):
                if a in Ztr.columns and b in Ztr.columns:
                    ZItr[name] = Ztr[a] * Ztr[b]
                    ZIte[name] = Zte[a] * Zte[b]
                    kept_inters.append(name)
            # Concatenate features actually available in this fold
            Xtr_df = pd.concat([Ztr, ZItr[kept_inters] if kept_inters else pd.DataFrame(index=Ztr.index)], axis=1)
            Xte_df = pd.concat([Zte, ZIte[kept_inters] if kept_inters else pd.DataFrame(index=Zte.index)], axis=1)
        else:
            Xtr_df, Xte_df = Ztr, Zte

        # Drop rows with any NaN in X or y
        tr_join = pd.concat([Xtr_df, tr[[target_col]]], axis=1).dropna()
        te_join = pd.concat([Xte_df, te[[target_col]]], axis=1).dropna()
        if tr_join.empty or te_join.empty:
            continue

        Xtr = np.column_stack([np.ones(len(tr_join)), tr_join[Xtr_df.columns].values])
        Xte = np.column_stack([np.ones(len(te_join)), te_join[Xte_df.columns].values])
        ytr = tr_join[target_col].values
        yte = te_join[target_col].values

        # Fit & score
        beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        yhat = Xte @ beta
        r2s.append(r2_score(yte, yhat))

    return float(np.nanmean(r2s)) if r2s else float("nan")


def pivot_features(df, feature_cols, target_col):
    """Pivot long-form df into wide format dict of feature DataFrames + target DataFrame."""
    Xs = {f: df.pivot(index="datetime", columns="symbol", values=f) for f in feature_cols}
    Y = df.pivot(index="datetime", columns="symbol", values=target_col)
    return Xs, Y


def build_design_matrix(train, feature_cols, interactions, inter_names):
    """Build design matrix with intercept, main effects, and interactions."""
    X_train_df = pd.DataFrame(index=train.index, dtype=float)
    X_train_df['intercept'] = 1.0

    # main effects
    for f in feature_cols:
        X_train_df[f] = train[f].values

    # interactions
    for (a, b), name in zip(interactions, inter_names):
        X_train_df[name] = train[a] * train[b]

    return X_train_df.values


def fit_ols(Xmat, yvec):
    """Fit plain OLS via lstsq."""
    return np.linalg.lstsq(Xmat, yvec, rcond=None)[0]


def build_today_features(Xs, di, cols, feature_cols, interactions, inter_names, beta_cols):
    """Build today's design matrix for prediction."""
    X_today_df = pd.DataFrame(index=cols, dtype=float)
    X_today_df['intercept'] = 1.0

    # main effects
    for f in feature_cols:
        X_today_df[f] = Xs[f].iloc[di].reindex(cols).values

    # interactions
    for (a, b), name in zip(interactions, inter_names):
        X_today_df[name] = X_today_df[a] * X_today_df[b]

    return X_today_df[beta_cols].values


def walkforward_cs_ols(df,
                       feature_cols,
                       lookback,
                       horizon,
                       target_col="target",
                       interactions=None,
                       verbose=False):
    """
    Walk-forward cross-sectional OLS (NO STANDARDIZATION).
    """
    if interactions is None:
        interactions = []

    # --- Precompute pivots
    Xs, Y = pivot_features(df, feature_cols, target_col)
    idx, cols = Y.index, Y.columns

    inter_names = [f"{a}*{b}" for (a, b) in interactions]
    beta_cols = ['intercept'] + list(feature_cols) + inter_names
    vif_cols = list(feature_cols) + inter_names

    # outputs
    F = pd.DataFrame(index=idx, columns=cols, dtype=float)
    B = pd.DataFrame(index=idx, columns=beta_cols, dtype=float)
    CV_R2 = pd.Series(index=idx, dtype=float)
    V = pd.DataFrame(index=idx, columns=vif_cols, dtype=float)

    # --- Walk forward
    for di in range(lookback, len(idx)):
        start, end = di - lookback, di - horizon + 1
        if end <= start:
            continue

        train_dates = idx[start:end]
        stacked_feats = [Xs[f].iloc[start:end].stack().rename(f) for f in feature_cols]
        ys = Y.iloc[start:end].stack().rename(target_col)
        train = pd.concat(stacked_feats + [ys], axis=1).dropna()
        if train.empty:
            continue

        if verbose:
            print(f"Forecast date {idx[di].date()} "
                  f"→ training window {train_dates[0].date()} to {train_dates[-1].date()} "
                  f"(n={len(train)})")

        # Build design matrix
        Xmat = build_design_matrix(train[feature_cols], feature_cols, interactions, inter_names)
        yvec = train[target_col].values

        # CV metric
        CV_R2.iloc[di] = purged_blocked_cv_r2(
            train,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=horizon,
            interactions=interactions,
            n_splits=5,
            min_train_days=10,
            embargo_by="steps"
        )

        # VIF
        try:
            V.loc[idx[di], vif_cols] = utils.calculate_vif(Xmat, vif_cols)
        except Exception:
            pass

        # Fit
        beta = fit_ols(Xmat, yvec)
        B.loc[idx[di], beta_cols] = beta

        # Predict
        X_today = build_today_features(Xs, di, cols, feature_cols, interactions, inter_names, beta_cols)
        F.iloc[di] = X_today @ beta

    return F, B, CV_R2, V


# Run walk-forward OLS
forecast, betas, cv_r2, vif_df = walkforward_cs_ols(
    df,
    feature_cols=FEATURE_COLS,
    # interactions = [("bolmom", "dv_decile")],
    target_col="target",
    lookback=TARGET_LOOKBACK,
    horizon=TARGET_HORIZON,
    verbose=True
)

# # Join forecasts back to original df
# df = df.merge(forecast.stack().rename('forecast'), left_on=['datetime', 'symbol'], right_index=True)

# Drop existing 'forecast' column from df to avoid conflicts
if 'forecast' in df.columns:
    df = df.drop(columns='forecast')

# Join forecasts back to original df
df = df.merge(forecast.stack().rename('forecast'), 
              left_on=['datetime', 'symbol'], 
              right_index=True, 
              how='left')


# Pivot for backtest inputs
def pivot_data(df: pd.DataFrame):
	forecast = df.pivot(index='datetime', columns='symbol', values='forecast')
	returns = df.pivot(index='datetime', columns='symbol', values='return')
	funding = df.pivot(index='datetime', columns='symbol', values='fundingRate')
	volatility = df.pivot(index='datetime', columns='symbol', values='volatility')

	return forecast, returns, funding, volatility

forecast, returns, funding, volatility = pivot_data(df)

def compute_weights(alpha_forecast: pd.DataFrame,
                    lag: bool = True,
                    inertia: float = 0.0,   # 0 ≤ inertia < 1
                    l2_shrink: float = 0.0  # ≥ 0, strength of shrinkage
                   ) -> pd.DataFrame:
    """
    Convert a cross-sectional alpha signal into dollar-neutral weights,
    with optional inertia and L2 shrinkage.

    Parameters
    ----------
    alpha_forecast : DataFrame  (index = timestamp, columns = symbol)
    lag            : bool       shift weights by one bar to avoid look-ahead
    inertia        : float      fraction of yesterday’s book to keep
                               0 → no decay (default)
                               0.6 → keep 60 % of prev. weights
    l2_shrink      : float      λ ≥ 0, shrinkage strength
                               0 → no shrinkage
                               larger → more pull toward 0
    """
    # 1) Demean signal
    adj = alpha_forecast.sub(alpha_forecast.mean(axis=1), axis=0)

    # 2) Dollar-neutral, |w| = 1
    weights_raw = adj.div(adj.abs().sum(axis=1), axis=0)

    # 3) Optional inertia
    if inertia:
        prev_w  = weights_raw.shift(1).fillna(0.0)
        weights = inertia * prev_w + (1 - inertia) * weights_raw
    else:
        weights = weights_raw

    # 4) Optional shrinkage: rescale toward 0
    if l2_shrink > 0:
        # Divide by (1 + λ) ⇒ simple ridge-like shrinkage
        weights = weights / (1.0 + l2_shrink)

    # 5) Optional lag
    if lag:
        weights = weights.shift(1)

    return weights.fillna(0.0)


weights = compute_weights(forecast, lag=WEIGHTS_LAG, inertia=WEIGHTS_INERTIA, l2_shrink=WEIGHTS_L2_SHRINK)
print(f"Turnover with inertia {WEIGHTS_INERTIA}: {weights.diff().abs().sum(axis=1).mean():.4f} per day")

weights_scaled = vol_target_weights(weights, returns,
								vol_target_ann=VOL_TARGET_ANN,
								lookback=VOL_TARGET_LOOKBACK,
								min_obs=VOL_TARGET_MIN_OBS)

# PnL and turnover calculation
def backtest(weights: pd.DataFrame,
			 returns: pd.DataFrame,
			 funding: pd.DataFrame,
			 cost_rate: float):
	weights = weights.fillna(0)
	return_pnl = (weights * returns).sum(axis=1)
	funding_pnl = -(weights * funding).sum(axis=1)
	total = return_pnl + funding_pnl
	turnover = weights.fillna(0).diff().abs().sum(axis=1)
	total_after_cost = total - cost_rate * turnover
	return return_pnl, funding_pnl, total_after_cost, turnover

trading_cost_rate = 0

rtn_pnl, fnd_pnl, tot_pnl_post_cost, turn = backtest(weights_scaled, returns, funding, TRADING_COST_RATE)

def plot_results(return_pnl, funding_pnl, total_after_cost, *, highlight_date=PLOT_HIGHLIGHT_DATE):
    # Build cumulative equity series (constant GMV, start = 1.0)
    gross_eq   = 1 + (return_pnl + funding_pnl).cumsum()
    total_eq   = 1 + total_after_cost.cumsum()
    return_eq  = 1 + return_pnl.cumsum()
    funding_eq = 1 + funding_pnl.cumsum()

    gross_eq.index   = pd.to_datetime(gross_eq.index)
    total_eq.index   = pd.to_datetime(total_eq.index)
    return_eq.index  = pd.to_datetime(return_eq.index)
    funding_eq.index = pd.to_datetime(funding_eq.index)

    # Fresh figure; constrained layout but no bbox-tight cropping
    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140, constrained_layout=True)

    # Colors (Okabe–Ito-ish)
    palette = {
        "gross":   "#E15759",
        "net":     "#4E4E4E",
        "return":  "#3B8EE5",
        "funding": "#47B39C",
    }
    lw = 2.2

    gross_eq.plot(ax=ax, label="Gross Equity (no costs)",   lw=lw, ls="--", color=palette["gross"])
    total_eq.plot(ax=ax, label="Net Equity (constant GMV)", lw=lw,            color=palette["net"])
    return_eq.plot(ax=ax, label="Price-Return Equity",       lw=lw,            color=palette["return"])
    funding_eq.plot(ax=ax, label="Funding Equity",           lw=lw,            color=palette["funding"])

    # Optional vertical reference line (+ label kept INSIDE the axes)
    if highlight_date is not None:
        d = pd.to_datetime(highlight_date)
        ax.axvline(d, lw=1.6, ls=":", color="#6F6F6F", alpha=0.85, zorder=0)
        ax.annotate(
            d.strftime("%Y-%m-%d"),
            xy=(d, 1.0), xycoords=('data', 'axes fraction'),
            xytext=(5, -5), textcoords='offset points',
            rotation=90, va='top', ha='left', fontsize='small',
            annotation_clip=False   # don't clip, but stays in axes bbox
        )

    ax.set_title("Equity Curves – Constant GMV", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ensure the full date range is used and give a little padding
    ax.set_xlim(min(gross_eq.index.min(), total_eq.index.min()),
                max(gross_eq.index.max(), total_eq.index.max()))
    ax.margins(x=0.01, y=0.05)

    plt.show()


# ── Expanded performance report ──────────────────────────────────────────────
def report_metrics(total_after_cost, return_pnl, turnover, cost_rate):
	"""
	Prints an extended performance summary.
	"""
	import numpy as np
	import pandas as pd

	# --- core statistics -----------------------------------------------------
	mu     = total_after_cost.mean()
	sigma  = return_pnl.std()
	sharpe = mu / sigma * np.sqrt(365)

	# 95 % CI on annualised Sharpe (Jobson & Korkie, 1981)
	n   = len(total_after_cost)
	se  = np.sqrt((1 + 0.5 * sharpe**2) * 365 / n)
	ci  = sharpe - 1.96 * se, sharpe + 1.96 * se

	# --- risk-adjusted extras -----------------------------------------------
	# annualised volatility
	vol_ann = sigma * np.sqrt(365) * 100

	# CAGR of the equity curve (constant-GMV arithmetic cum-P&L)
	equity_curve = 1 + total_after_cost.cumsum()
	cagr = equity_curve.iloc[-1]**(365 / n) - 1

	# max drawdown of cumulative P&L (not compounded)
	running_max = equity_curve.cummax()
	drawdown    = equity_curve - running_max
	max_dd      = drawdown.min()      # negative number

	# turnover & cost
	avg_turnover   = turnover.mean()
	avg_hold       = 1 / avg_turnover if avg_turnover else np.inf
	cost_pnl       = -cost_rate * turnover
	cost_drag_bps  = cost_pnl.mean() * 1e4* 365   # annual cost in bps

	# --- print nicely --------------------------------------------------------
	print("──────── Performance Summary ────────")
	print(f"CAGR:                          {cagr:7.2%}")
	print(f"Annualised volatility:         {vol_ann:7.2f}%")
	print(f"Sharpe (ann.):                 {sharpe:7.2f}   95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
	print(f"Max drawdown:                 {max_dd:8.2%}")
	print("──────── Execution / Cost ───────────")
	print(f"Avg daily turnover:            {avg_turnover:7.2%}")
	print(f"Avg holding period:            {avg_hold:7.2f} days")
	print(f"Average annual cost drag:       {cost_drag_bps:7.2f} bp")


report_metrics(tot_pnl_post_cost, rtn_pnl, turn, TRADING_COST_RATE)