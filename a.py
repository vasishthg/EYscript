import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime

class StockPredictor:
    def __init__(self):
        pass

    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch historical stock data using yfinance.
        """
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data = self._calculate_technical_indicators(stock_data)
        stock_data.dropna(inplace=True)
        return stock_data

    def _calculate_technical_indicators(self, data):
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

    def prepare_features(self, data):
        """
        Prepare features for machine learning model.
        """
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'ATR']
        X = data[features]
        y = data['Target']
        return X, y

    def train_model(self, X, y):
        """
        Train a Random Forest Regressor model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        print(f"Model Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

        return model, scaler

    def predict_next_day(self, model, scaler, latest_data):
        """
        Predict the next day's stock price.
        """
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'RSI', 'ATR']
        X_latest = latest_data[features].iloc[-1].values.reshape(1, -1)
        X_latest_scaled = scaler.transform(X_latest)
        predicted_price = model.predict(X_latest_scaled)[0]
        return predicted_price

# Main Script
def main():
    predictor = StockPredictor()

    # Define the stocks to analyze
    stocks = input("Enter the stock symbols separated by commas: ").split(",")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)  # 2 years of data

    stock_predictions = []

    stock_predictions = []

    for symbol in stocks:
        print(f"Processing {symbol}...")
        stock_data = predictor.fetch_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            print(f"No data available for {symbol}. Skipping.")
            continue

        X, y = predictor.prepare_features(stock_data)
        model, scaler = predictor.train_model(X, y)

        predicted_price = predictor.predict_next_day(model, scaler, stock_data)
        current_price = float(stock_data['Close'].iloc[-1])

        
        try:
            potential_profit = float(predicted_price) - float(current_price)
            stock_predictions.append({
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "potential_profit": potential_profit
            })
        except ValueError as e:
            print(f"Error calculating potential profit for {symbol}: {e}")
            continue

    # Sort predictions by potential profit
    stock_predictions.sort(key=lambda x: x.get("potential_profit", float("-inf")), reverse=True)


    print("\nStock Recommendations:")
    for prediction in stock_predictions:
        print(f"{prediction['symbol']} - Current Price: ${prediction['current_price']:.2f}, "
              f"Predicted Price: ${prediction['predicted_price']:.2f}, "
              f"Potential Profit: ${prediction['potential_profit']:.2f}")

    best_stock = stock_predictions[0]
    print(f"\nBest Stock to Buy: {best_stock['symbol']} with potential profit of ${best_stock['potential_profit']:.2f}")

if __name__ == "__main__":
    main()
