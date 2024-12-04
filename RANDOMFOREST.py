# This script is focusing on predicting the next day's closing price using a Random Forest Regressor. 
# It calculates technical indicators, trains the model, and evaluates its performance on a test dataset.
# It also prints the predicted and actual prices for a specific date.

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import datetime

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_data = _calculate_technical_indicators(stock_data)
    stock_data.dropna(inplace=True)
    return stock_data

def _calculate_technical_indicators(data):
    """
    Manually calculate technical indicators.
    """
    # Moving Averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)  # Keep positive changes
    loss = -delta.clip(upper=0)  # Keep negative changes and make positive

    # Align rolling windows correctly
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper_BB'] = rolling_mean + (2 * rolling_std)
    data['Lower_BB'] = rolling_mean - (2 * rolling_std)

    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14, min_periods=1).mean()

    # Target for Prediction
    data['Target'] = data['Close'].shift(-1)  # Predict next day's price

    return data

def prepare_features(data):
    """
    Prepare features for machine learning model.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'ATR']
    X = data[features]
    y = data['Target']
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def predict_next_day(model, scaler, latest_data):
    """
    Predict the next day's stock price.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'ATR']
    X_latest = latest_data[features].iloc[-1].values.reshape(1, -1)
    X_latest_scaled = scaler.transform(X_latest)
    predicted_price = model.predict(X_latest_scaled)[0]
    return predicted_price

if __name__ == "__main__":
    stock_symbol = "AAPL"
    end_date = datetime.date(2023, 12, 4)
    start_date = end_date - datetime.timedelta(days=730)
    test_size = 0.2

    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    X, y = prepare_features(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model, scaler = train_model(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    latest_data = stock_data.iloc[-1:]  # Get the latest day's data (December 2nd)
    predicted_price = predict_next_day(model, scaler, latest_data)

    actual_price = float(stock_data['Close'].iloc[-1])

    print(f"Predicted Price for {stock_symbol} on December 2nd, 2022: ${predicted_price:.2f}")
    print(f"Actual Price on December 2nd, 2022: ${actual_price:.2f}")

    # Evaluate the model's performance on the test set
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")