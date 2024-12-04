# This script fetches stock data, calculates technical indicators (MA, RSI, Bollinger Bands, ATR), 
# trains a Random Forest Regressor to predict the next day's closing price, and evaluates the model using metrics like MSE, RMSE, and MAPE. 
# It also includes functionality to predict the next day's price and compare it to the actual price.
import os
import requests
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

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

def add_features(df):
    df['Moving Average'] = df['close'].rolling(window=5).mean()
    df['Volatility'] = df['close'].rolling(window=5).std()
    df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(4)
    df['RSI'] = calculate_rsi(df['close'], period=14)
    df.dropna(inplace=True)
    return df

# Training Function
def train_model(data):
    features = ['Moving Average', 'Volatility', 'EMA', 'Momentum', 'RSI']
    X = data[features]
    y = np.where(data['close'].shift(-1) > data['close'], 1, 0)

    X_scaled = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    print(f"Average Model Accuracy: {np.mean(accuracies) * 100:.2f}%")
    return model

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
