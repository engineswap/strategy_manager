import pandas as pd
import numpy as np

# Note: we compute raw signals here; 
# group dependent transformations (eg. z scoring) should happen after uni filtering 

def _compute_ls_ratio(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing long–short ratio signal...")
	df[signal_name] = df.groupby('symbol')['ls_ratio'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	return df

def _compute_longs_pct(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing long percentage signal...")
	df[signal_name] = df.groupby('symbol')['longs_pct'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	return df

def _compute_bolmom(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing Bollinger Band momentum signal...")
	# df = df.sort_values(['datetime', 'symbol'])

	# compute daily returns and EWM volatility/bands
	df['vol_ewm']   = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=lookback).std())
	df['mid_band']  = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=lookback).mean())
	df['boll_dist'] = (df['close'] - df['mid_band']) / (2 * df['vol_ewm'])

	# final signal clipped
	df[signal_name] = df['boll_dist'].clip(-1, 1)

	# drop intermediates
	return df.drop(columns=['vol_ewm','mid_band','boll_dist'])

def _compute_carry(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing funding rate momentum signal...")
	# EWM carry and adjust by vol
	df['carry_ewm'] = df.groupby('symbol')['fundingRate'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())

	df['carry_adj'] = df['carry_ewm'] / (df['volatility'])

	df[signal_name] = df['carry_adj']
	return df.drop(columns=['carry_ewm','carry_adj'])


def _compute_buy_volume_ratio(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing buy volume ratio signal...")
	df['buy_volume_perp_slowed'] = df.groupby('symbol')['buy_volume_perp'].transform(
		lambda x: x.ewm(span=lookback).mean()
	)
	df['total_volume_perp_slowed'] = df.groupby('symbol')['total_volume_perp'].transform(
		lambda x: x.ewm(span=lookback).mean()
	)
	df[signal_name] = df['buy_volume_perp_slowed'] / df['total_volume_perp_slowed']

	return df.drop(columns=['buy_volume_perp_slowed','total_volume_perp_slowed'])


def _compute_liquidation_imbalance(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing liquidation imbalance signal...")
	# EWM spans set to 20 periods
	df['long_liq_ewm']  = df.groupby('symbol')['long_liquidation_volume'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	df['short_liq_ewm'] = df.groupby('symbol')['short_liquidation_volume'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	df['mcap_ewm']      = df.groupby('symbol')['mkt_cap'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())

	# imbalance = (long - short) / mcap; negative to buy short-liquidations, sell long-liquidations
	df[signal_name] = -((df['long_liq_ewm'] - df['short_liq_ewm']) / df['mcap_ewm'])

	# replace inf with nan
	df[signal_name] = df[signal_name].replace([np.inf, -np.inf], np.nan)
	

	# drop intermediates
	return df.drop(columns=['long_liq_ewm','short_liq_ewm','mcap_ewm'])

def _compute_funding_volatility(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing funding rate volatility signal...")
	df[signal_name] = df.groupby('symbol')['fundingRate'].transform(lambda x: x.ewm(span=lookback, adjust=False).std())
	
	return df

def _compute_turnover(df: pd.DataFrame, signal_name, lookback) -> pd.DataFrame:
	print("Computing turnover signal...")
	df['oi_ewm']     = df.groupby('symbol')['oi'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	df['volume_ewm'] = df.groupby('symbol')['total_volume_perp'].transform(lambda x: x.ewm(span=lookback, adjust=False).mean())
	df[signal_name]  = np.log(df['volume_ewm'] / df['oi_ewm'])

	return df
