import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# 1. Define tickers
tickers = ['NVDA','AMD','INTC','CSCO','AVGO','GLW','DELL','STX','WDC','VRT','CAT','ETN']

# 2. Download max historical data
data = yf.download(tickers, start="2000-01-01")['Close']

# 3. Trim to the earliest date where all tickers have data
start_date = data.dropna(how='any').index[0]
data = data.loc[start_date:]

# 4. Stock returns (%)
returns = data.pct_change() * 100

# 5. Market caps & weights
market_caps = {t: yf.Ticker(t).info.get('marketCap', np.nan) for t in tickers}
market_caps = {t: cap for t,cap in market_caps.items() if not np.isnan(cap)}
valid = [t for t in tickers if t in market_caps]
caps = pd.Series({t: market_caps[t] for t in valid})
weights = caps / caps.sum()

# 6. Portfolio returns & cumulative “price”
port_ret = (returns[valid].multiply(weights, axis=1)).sum(axis=1)
port_price = (1 + port_ret/100).cumprod()

# 7. Max drawdown
def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    dd = (prices - peak) / peak
    return dd.min() * 100  # as percentage

stock_drawdowns = {t: max_drawdown(data[t]) for t in valid}
port_drawdown = max_drawdown(port_price)

# 8. Beta calculation
benchmark = yf.download('SPY', start=start_date)['Close']
bm_ret = benchmark.pct_change() * 100

def calculate_beta(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    df = pd.concat([asset_ret, bench_ret], axis=1, join='inner').dropna()
    x = df.iloc[:,1]  # benchmark
    y = df.iloc[:,0]  # asset
    slope, _, _, _, _ = linregress(x, y)
    return slope

betas = {t: calculate_beta(returns[t], bm_ret) for t in valid}
port_beta = calculate_beta(port_ret, bm_ret)

# 9. Plotting (2 rows × 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Stock Prices
for t in valid:
    axs[0,0].plot(data.index, data[t], label=t)
axs[0,0].set_title("Stock Prices Over Time")
axs[0,0].legend(loc='upper left', fontsize='small', ncol=2)
axs[0,0].grid(True)

# (0,1) Portfolio Performance (cumulative %)
axs[0,1].plot(port_price.index, (port_price - 1)*100, color='black')
axs[0,1].set_title("Portfolio Cumulative Return (%)")
axs[0,1].set_ylabel("Cumulative Return (%)")
axs[0,1].grid(True)

# (0,2) Risk vs Return
mean_ret = returns[valid].mean()
std_ret  = returns[valid].std()
axs[0,2].scatter(std_ret, mean_ret, color='blue')
for t in valid:
    axs[0,2].annotate(t, (std_ret[t], mean_ret[t]), fontsize=8)
axs[0,2].scatter(port_ret.std(), port_ret.mean(), color='red', marker='*', s=150, label='Portfolio')
axs[0,2].set_title("Mean vs Std Dev (%)")
axs[0,2].set_xlabel("Std Dev (%)")
axs[0,2].set_ylabel("Mean Return (%)")
axs[0,2].legend()
axs[0,2].grid(True)

# (1,0) Market Caps
axs[1,0].bar(valid, caps[valid]/1e9, color='teal')
axs[1,0].set_title("Market Cap (USD billions)")
axs[1,0].set_ylabel("Billions USD")
axs[1,0].tick_params(axis='x', rotation=90)
axs[1,0].grid(True)

# (1,1) Max Drawdown
dd_vals = {**stock_drawdowns, 'Portfolio': port_drawdown}
axs[1,1].bar(dd_vals.keys(), dd_vals.values(), color='orange')
axs[1,1].set_title("Max Drawdown (%)")
axs[1,1].tick_params(axis='x', rotation=90)
axs[1,1].grid(True)

# (1,2) Beta
beta_vals = {**betas, 'Portfolio': port_beta}
axs[1,2].bar(beta_vals.keys(), beta_vals.values(), color='green')
axs[1,2].set_title("Beta vs SPY")
axs[1,2].tick_params(axis='x', rotation=90)
axs[1,2].grid(True)

plt.tight_layout()
plt.show()
