import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Ask user how many stocks
num_stocks = int(input("How many stocks do you want to analyze? "))
# Input tickers
print(f"Enter {num_stocks} stock ticker symbols (e.g., AAPL, TSLA, INFY.NS):")
stock_symbols = [input(f"Stock {i+1}: ").strip().upper() for i in range(num_stocks)]

# Initialize containers
closing_prices_all = pd.DataFrame()
returns_stats = {}

# Download & calculate metrics
for symbol in stock_symbols:
    try:
        data = yf.download(symbol, period="max", interval="1d")['Close']
        data.dropna(inplace=True)
        closing_prices_all[symbol] = data

        daily_returns = data.pct_change().dropna()
        annualized_return = daily_returns.mean() * 252
        annualized_std_dev = daily_returns.std() * (252 ** 0.5)

        # Extract scalar values for calculations
        ann_return_val = annualized_return.item()
        ann_std_val = annualized_std_dev.item()
        risk_reward_ratio = ann_std_val / ann_return_val if ann_return_val != 0 else float('inf')

        returns_stats[symbol] = {
            'mean': ann_return_val,
            'std': ann_std_val,
            'risk_reward': risk_reward_ratio
        }

    except Exception as e:
        print(f"Failed to process {symbol}: {e}")

# Create a figure with 3 plots side by side
fig, axs = plt.subplots(1, 3, figsize=(21, 6))

# Plot 1: Daily Closing Prices
for symbol in closing_prices_all.columns:
    axs[0].plot(closing_prices_all.index, closing_prices_all[symbol], label=symbol)
axs[0].set_title("Daily Closing Prices")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Price")
axs[0].legend()
axs[0].grid(True)

# Plot 2: Risk vs Return
for symbol, stats in returns_stats.items():
    axs[1].scatter(stats['std'], stats['mean'], label=symbol, s=100)
    axs[1].text(stats['std'] + 0.002, stats['mean'], symbol)
axs[1].set_title("Risk vs Return (Annualized)")
axs[1].set_xlabel("Standard Deviation (Risk)")
axs[1].set_ylabel("Expected Return")
axs[1].grid(True)

# Plot 3: Risk/Reward Ratio Bar Chart
symbols = list(returns_stats.keys())
ratios = [returns_stats[s]['risk_reward'] for s in symbols]
axs[2].bar(symbols, ratios, color='skyblue')
axs[2].set_title("Risk/Reward Ratio (Lower is Better)")
axs[2].set_xlabel("Stock")
axs[2].set_ylabel("Risk / Reward")
axs[2].grid(True, axis='y')

plt.tight_layout()
plt.show()
