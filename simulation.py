import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Title of the Streamlit app
st.title(" DESSI - Stock Price Simulation with GBM")

# Stock selection (user can choose the stock)
stock_symbol = st.selectbox(
    "Select a stock symbol",
    ('RACE.MI','GOOGL', 'AAPL', 'MSFT', 'META', 'NVDA')  # You can add more symbols if needed
)

# Fetch historical data for the selected stock
googl_hist = yf.download(stock_symbol, start='2023-01-01', end='2024-11-27')

# Calculate daily returns
googl_hist['Return'] = googl_hist['Close'].pct_change().dropna()

# Estimate drift (annualized) and volatility (annualized)
returns = googl_hist['Return'].dropna()
mu = returns.mean() * 252  # Annualize the mean
sigma = returns.std() * (252 ** 0.5)  # Annualize the standard deviation

# Set the initial stock price (last closing price)
S0 = googl_hist['Close'].iloc[-1]

# Define simulation parameters
T = (datetime.datetime(2025, 12, 31) - googl_hist.index[-1]).days / 365  # Total simulation time (in years)
N = int(T * 252)  # Number of time steps (252 trading days in a year)

# Allow the user to choose the number of simulated paths
num_paths = st.slider("Select number of simulation paths", min_value=1, max_value=100, value=5)

# Function to simulate multiple GBM paths
def simulate_gbm_multiple_paths(S0, mu, sigma, T, N, num_paths):
    dt = T / N
    t = np.linspace(0, T, N)
    paths = np.zeros((N, num_paths))
    for i in range(num_paths):
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion component
        paths[:, i] = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return paths

# Simulate GBM paths
simulated_paths = simulate_gbm_multiple_paths(S0, mu, sigma, T, N, num_paths)

# Actual final price (last known closing price)
actual_final_price = googl_hist['Close'].iloc[-1]
last_trading_date = googl_hist.index[-1]

# Create a date range for the simulation starting from the last trading date
future_dates = pd.date_range(last_trading_date, periods=N, freq='B')  # Business days for the simulated data

# Plot the actual price and all simulated prices
plt.figure(figsize=(10, 5))
plt.plot(googl_hist.index, googl_hist['Close'], color='blue', label='Actual Closing Prices')  # Actual closing prices

# Plot all simulated paths starting from the last actual price
for i in range(num_paths):
    plt.plot(future_dates, simulated_paths[:, i], alpha=0.7, label='Simulated Price' if i == 0 else "")

plt.axhline(y=actual_final_price, color='red', linestyle='--', label='Actual Price')  # Actual price line
plt.title(f"Simulated Stock Prices for {stock_symbol} (Starting from Last Trading Date)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
st.pyplot(plt)  # Display the plot in Streamlit
