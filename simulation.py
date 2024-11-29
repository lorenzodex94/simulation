import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error



# Get today's date
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Title of the Streamlit app
st.title(" DESSI - Stock Price Simulation with GBM")

# Stock selection (user can choose the stock)
stock_symbol = st.selectbox(
    "Select a stock symbol",
    ('RACE.MI','GOOGL', 'AAPL', 'MSFT', 'META', 'NVDA', 'SPY', 'TSLA', 'AMZN','XLC')  # You can add more symbols if needed
)

# Fetch historical data for the selected stock
googl_hist = yf.download(stock_symbol, start='2020-01-01', end=yesterday)

# Calculate daily returns
googl_hist['Return'] = googl_hist['Close'].pct_change().dropna()

# Estimate drift (annualized) and volatility (annualized)
returns = googl_hist['Return'].dropna()
mu = returns.mean() * 252  # Annualize the mean
sigma = returns.std() * (252 ** 0.5)  # Annualize the standard deviation

# Set the initial stock price (last closing price)
S0 = googl_hist['Close'].iloc[-1]

# Define simulation parameters
T = (datetime.datetime(2025, 6, 1) - googl_hist.index[-1]).days / 365  # Total simulation time (in years)
N = int(T * 252)  # Number of time steps (252 trading days in a year)

# Allow the user to choose the number of simulated paths
num_paths = st.slider("Select number of simulation paths", min_value=1, max_value=50, value=5)

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
plt.figure(figsize=(12, 6))
plt.plot(googl_hist.index, googl_hist['Close'], color='blue', label='Actual Closing Prices')  # Actual closing prices

# Plot all simulated paths starting from the last actual price
for i in range(num_paths):
    plt.plot(future_dates, simulated_paths[:, i], alpha=0.7, label='Simulated Price' if i == 0 else "")

plt.axhline(y=actual_final_price, color='red', linestyle='--', label='Actual Price')  # Actual price line
plt.title(f"Simulated Stock Prices for {stock_symbol} (Starting from Last Trading Date) - up {yesterday}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
st.pyplot(plt)  # Display the plot in Streamlit

#################################################################################################################################################
ticker = stock_symbol

# Function for fetching and cleaning stock data
def get_clean_financial_data(stock_symbol, start_date, end_date):
    # Download data
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Clean structure
    data.columns = data.columns.get_level_values(0)

    # Handle missing values
    data = data.ffill()

    # Standardize timezone
    data.index = data.index.tz_localize(None)

    return data

# Fetch historical stock data for DIA (Dow Jones Industrial Average ETF)
data = get_clean_financial_data(stock_symbol, '2020-01-01', yesterday)

# Use the 'Close' price as the target variable
data = data.reset_index()
data['Date_Ordinal'] = pd.to_numeric(data['Date'].map(pd.Timestamp.toordinal))

# Prepare features and target variable
X = data[['Date_Ordinal']].values
y = data['Close'].values

# Fit a Gaussian Mixture Model (GMM) to the data
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the latent values using the GMM
latent_features = gmm.predict_proba(X)

# Combine latent features with original features
X_latent = np.hstack([X, latent_features])

# Fit a polynomial regression model on the combined features
poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_reg.fit(X_latent, y)

# Predict and evaluate the model
y_pred = poly_reg.predict(X_latent)
mse = mean_squared_error(y, y_pred)

# Calculate the residuals and their standard deviation
residuals = y - y_pred
std_dev = np.std(residuals)

# Create upper and lower standard deviation lines
upper_bound = y_pred + 2 * std_dev
lower_bound = y_pred - 2 * std_dev

# Create buy and sell signals
data['Buy_Signal'] = np.where(y < lower_bound, 1, 0)   # Buy when price is below lower bound
data['Sell_Signal'] = np.where(y > upper_bound, 1, 0)  # Sell when price is above upper bound

# Plotting
plt.figure(figsize=(12, 6))
plt.title(f'Polynomial Regression {stock_symbol} Data with Buy and Sell Signals - update {yesterday}')

# Plot price data
plt.plot(data['Date'], y, color='blue', label='Actual Closing Price')
plt.plot(data['Date'], y_pred, color='red', linestyle='--', label='Fitted Values')
plt.plot(data['Date'], upper_bound, color='green', linestyle=':', label='Upper Bound (±2 Std Dev)')
plt.plot(data['Date'], lower_bound, color='green', linestyle=':', label='Lower Bound (±2 Std Dev)')
plt.fill_between(data['Date'], lower_bound, upper_bound, color='green', alpha=0.1)

# Plot Buy Signals
buy_signals = data[data['Buy_Signal'] == 1]
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='magenta', label='Buy Signal', s=100)

# Plot Sell Signals
sell_signals = data[data['Sell_Signal'] == 1]
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='orange', label='Sell Signal', s=100)

plt.ylabel('Close Price')
plt.xlabel('Date')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
st.pyplot(plt)



# Create a second plot for the distribution of daily returns
plt.figure(figsize=(10, 5))
plt.hist(returns, bins=30, color='orange', alpha=0.7, edgecolor='black')
plt.title(f"Distribution of Daily Returns for {stock_symbol} - update {yesterday}")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.grid()
st.pyplot(plt)  # Display the second plot in Streamlit















