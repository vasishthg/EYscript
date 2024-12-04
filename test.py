import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import datetime

def fetch_stock_data(symbols, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    all_data = []
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if not stock_data.empty:
            stock_data['Symbol'] = symbol  # Add stock symbol for multi-stock data
            stock_data = _calculate_technical_indicators(stock_data)
            all_data.append(stock_data)

    return pd.concat(all_data)

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
    data.dropna(inplace=True)  # Drop all rows with NaN values
    return data

def prepare_features(data):
    """
    Prepare features for machine learning model.
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'ATR']
    X = data[features]
    y = data['Target']
    return X, y

def train_model(X, y):
    """
    Train a Random Forest Regressor model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    scores = cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=5)
    print(f"Cross-validated MSE: {-np.mean(scores):.4f}")
    
    model.fit(X_scaled, y)
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
    stock_symbol = ["MSFT", "AAPL", "GOOGL", "AMZN"] 
    end_date = datetime.date(2024, 5, 3)
    start_date = end_date - datetime.timedelta(days=730)
    test_size = 0.2

    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)

    X, y = prepare_features(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model, scaler = train_model(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    latest_data = stock_data.iloc[-1:] 
    predicted_price = predict_next_day(model, scaler, latest_data)

    stock_data['Close'] = stock_data['Close'].fillna(method = 'ffill')
    try:
        actual_price = float(stock_data['Close'].iloc[-1])
    except ValueError:
        print("Failed to convert 'Close' to float. Check if the last row contains valid data.")
        actual_price = None  # Handle the case where conversion fails

    if actual_price is not None:
        print(f"Actual Price on {stock_symbol}: ${actual_price:.2f}")
    else:

        print(f"Could not retrieve a valid actual price for {stock_symbol}.")
    print(f"Predicted Price for {stock_symbol} : ${predicted_price:.2f}")
    print(f"Actual Price on  ${actual_price:.2f}")

    # Evaluate the model's performance on the test set
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")