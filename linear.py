import os
import requests
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# API Keys
POLYGON_API_KEY = "F9NEtqzR2Tkk60wQ2pvFPCjLkMZBz3n6"

# Global Variables
scaler = StandardScaler()
bought_stocks = {}

# Helper Functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_polygon_data(ticker, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []

def preprocess_data(data):
    if data:
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('t', inplace=True)
        df.rename(columns={
            'v': 'volume',
            'vw': 'vwap',
            'o': 'open',
            'c': 'close',
            'h': 'high',
            'l': 'low'
        }, inplace=True)
        return df.dropna()
    return pd.DataFrame()

# ... (Rest of your code, including data fetching and preprocessing)

def predict_volatility_linear_regression(data):
    # Calculate daily returns
    data['Return'] = data['close'].pct_change()

    # Calculate volatility (standard deviation of returns)
    data['Volatility'] = data['Return'].rolling(window=20).std()

    # Shift volatility to predict future volatility
    data['Target_Volatility'] = data['Volatility'].shift(-1)

    # Prepare data for training
    X = data[['Close', 'Volume', 'Moving Average', 'Volatility']]  # Adjust features as needed
    y = data['Target_Volatility']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    return model

# ... (Rest of your code, including model deployment and evaluation)
# Portfolio Analysis
def analyze_portfolio():
    results = []
    for ticker, info in bought_stocks.items():
        closing_price = fetch_polygon_data(ticker, info['Date'], info['Date'])
        if closing_price:
            try:
                market_price = closing_price[-1]['c']  # Use 'c' for closing price
                profit_loss = (market_price - info['Buy Price']) * info['Shares']

                results.append({
                    **info,
                    'Market Price': market_price,
                    'Profit/Loss': profit_loss,
                })
            except KeyError:
                print(f"Error: 'c' key missing for {ticker} data.")
        else:
            print(f"No data returned for {ticker} on {info['Date']}")
    return pd.DataFrame(results)

def make_trading_decision(predicted_volatility, current_price, buy_price, stop_loss, take_profit):
    if predicted_volatility > high_volatility_threshold:
        return "Sell"
    elif current_price >= take_profit:
        return "Sell (Profit Target)"
    elif current_price <= stop_loss:
        return "Sell (Stop Loss)"
    else:
        return "Hold"
def save_results(results):
    results.to_csv("portfolio_analysis.csv", index=False)
    print("Results saved to portfolio_analysis.csv")

def plot_portfolio(results):
    plt.bar(results['Ticker'], results['Profit/Loss'])
    plt.xlabel("Ticker")
    plt.ylabel("Profit/Loss")
    plt.title("Portfolio Analysis")
    plt.show()

# User Input and Main Menu
def get_user_input():
    ticker = input("Enter stock ticker: ").upper()
    date_str = input("Enter buy date (YYYY-MM-DD): ")
    shares = int(input("Enter shares bought: "))
    buy_price = float(input("Enter buy price: "))

    return {"Ticker": ticker, "Shares": shares, "Buy Price": buy_price, "Date": date_str}

def main():
    while True:
        choice = input("Enter 'a' to add, 'v' to view, 's' to save, 'p' to plot, 'q' to quit: ").lower()
        if choice == 'q':
            break
        elif choice == 'a':
            stock_info = get_user_input()
            bought_stocks[stock_info['Ticker']] = stock_info
        elif choice == 'v':
            results = analyze_portfolio()
            print(results)
        elif choice == 's':
            results = analyze_portfolio()
            save_results(results)
        elif choice == 'p':
            results = analyze_portfolio()
            plot_portfolio(results)

if __name__ == "__main__":
    main()
