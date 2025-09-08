~ 20 mins to download data 
~ 10 minutes to rebalance

How to add a new strategy:
- add its parameters to rebal_params.json
- Add its signal to signals.py

Ideas
- LS ratio might work try it again
- finish the other signal suggested by chatgpt too
- switch to aggregated coinalyze data, might be cleaner signal

Todo
- Project: More sophisticated stat-arb
  - Model to forecast returns
  - Walk forward backtest
- Think about vol scaled signals, especially if the signal has small magnitude, sizing will have an outsized influence by volatility then
- Speed up the few slowest cells
- Show the ic/backtest by market caps 1-10, 10-20, ...
- Add security master and show loading on sector
- Show exposure to volatility, market cap, and momentum


Production changes 
- May 20th 
  - No longer XS z scoring during signal computation (since the universe isnt filtered yet)
  - The backtest changes seem small


Observations
- See how well we're matching the backtest (may 20, 2025)
  - Some are better than others, hard to say why exactly, perhaps capital is too low or coin blacklist or implementation changes.