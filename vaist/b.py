'''
This script is a simple stock predictor that uses historical stock data to predict future stock prices.
It shows historical stock data, calculates technical indicators, trains a Random Forest Regressor model, and predicts future stock prices.
It also calculates profit/loss based on the user's input for the purchase date and number of shares bought.
'''

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class StockPredictor:
    def __init__(self):
        pass

    def fetch_stock_data(self, symbol, start_date, end_date):
        try:
            extended_end_date = end_date + datetime.timedelta(days=30)
            stock_data = yf.download(symbol, start=start_date, end=extended_end_date)
            
            if stock_data.empty:
                print(f"No data available for {symbol}")
                return pd.DataFrame()

            stock_data = self._calculate_technical_indicators(stock_data)
            stock_data.dropna(inplace=True)
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, data):
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Upper_BB'] = rolling_mean + (2 * rolling_std)
        data['Lower_BB'] = rolling_mean - (2 * rolling_std)

        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14, min_periods=1).mean()

        data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['12_EMA'] - data['26_EMA']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        data['Pct_Change'] = data['Close'].pct_change()
        data['Target'] = data['Close'].shift(-1)

        return data

    def prepare_features(self, data):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA50', 'MA200', 'RSI', 'ATR', 
                    'MACD', 'Signal_Line', 'Pct_Change']
        X = data[features]
        y = data['Target']
        return X, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_scores)

        print("Model Performance Metrics:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2%}")
        print(f"R-squared Score: {r2:.4f}")
        print(f"Cross-Validation RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)

        return model, scaler

    def predict_next_day(self, model, scaler, latest_data):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA50', 'MA200', 'RSI', 'ATR', 
                    'MACD', 'Signal_Line', 'Pct_Change']
        X_latest = latest_data[features].iloc[-1].values.reshape(1, -1)
        X_latest_scaled = scaler.transform(X_latest)
        
        predicted_price = model.predict(X_latest_scaled)[0]
        return predicted_price

    def predict_future_prices(self, model, scaler, latest_data, days=5):
        """
        Predict stock prices for the next 'days' days along with dates.
        """
        future_predictions = []
        last_date = latest_data.index[-1]  # Get the last date in the stock data
        
        for i in range(1, days + 1):
            # Predict next day's price
            predicted_price = self.predict_next_day(model, scaler, latest_data)
            
            # Append predicted price with the corresponding date
            prediction_date = last_date + pd.Timedelta(days=i)
            future_predictions.append({
                'date': prediction_date,
                'predicted_price': predicted_price
            })
            
            # Update latest data with the new prediction for the next iteration
            latest_row = latest_data.iloc[-1].copy()
            latest_row['Close'] = predicted_price
            
            # Update other features based on the new predicted Close price
            latest_row['MA50'] = latest_data['Close'].rolling(window=50).mean().iloc[-1]  # Recalculate MA50
            latest_row['MA200'] = latest_data['Close'].rolling(window=200).mean().iloc[-1]  # Recalculate MA200
            
            # You may need to recalculate other indicators as needed
            latest_row['Pct_Change'] = (latest_row['Close'] - latest_data['Close'].iloc[-1]) / latest_data['Close'].iloc[-1]
            
            # Append the new row to the latest data
            latest_data = pd.concat([latest_data, latest_row.to_frame().T], ignore_index=True)

            # Recalculate all technical indicators for the new row
            latest_data = self._calculate_technical_indicators(latest_data)

        return future_predictions


    def calculate_profit_loss(self, stock_data, purchase_price, num_stocks, predicted_price):
        current_price = float(stock_data['Close'].iloc[-1]) 
        
        current_value = current_price * num_stocks
        purchase_value = purchase_price * num_stocks
        predicted_value = float(predicted_price) * num_stocks
        
        unrealized_gain_loss = current_value - purchase_value
        potential_gain_loss = predicted_value - purchase_value
        
        return {
            'current_price': current_price,
            'purchase_price': purchase_price,
            'predicted_price': float(predicted_price),
            'unrealized_gain_loss': unrealized_gain_loss,
            'potential_gain_loss': potential_gain_loss,
            'current_total_value': current_value,
            'purchase_total_value': purchase_value
        }

    def plot_stock_data(self, data, symbol):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Close Price', color='b', linewidth=1.5)
        plt.plot(data.index, data['MA50'], label='50-day MA', color='orange', linewidth=1)
        plt.plot(data.index, data['MA200'], label='200-day MA', color='g', linewidth=1)
        plt.fill_between(data.index, data['Upper_BB'], data['Lower_BB'], color='gray', alpha=0.1, label='Bollinger Bands')
        plt.title(f'{symbol} Stock Price and Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

def main():
    predictor = StockPredictor()

    stocks = input("Enter the stock symbols separated by commas: ").split(",")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=1095)  # 3 years of data

    stock_predictions = []

    for symbol in stocks:
        print(f"Processing {symbol}...")
        stock_data = predictor.fetch_stock_data(symbol.strip(), start_date, end_date)

        if stock_data.empty:
            print(f"No data available for {symbol}. Skipping.")
            continue

        predictor.plot_stock_data(stock_data, symbol)

        X, y = predictor.prepare_features(stock_data)
        model, scaler = predictor.train_model(X, y)

        predicted_price = predictor.predict_next_day(model, scaler, stock_data)
        future_predictions = predictor.predict_future_prices(model, scaler, stock_data)

        purchase_date = input(f"Enter the purchase date for {symbol} (YYYY-MM-DD): ")
        purchase_date = datetime.datetime.strptime(purchase_date, '%Y-%m-%d').date()
        num_stocks = int(input(f"Enter the number of shares of {symbol} you bought: "))

        purchase_price = float(stock_data.loc[stock_data.index.date == purchase_date, 'Close'].iloc[0]) if not stock_data.loc[stock_data.index.date == purchase_date, 'Close'].empty else None

        if purchase_price is not None:
            profit_loss_data = predictor.calculate_profit_loss(stock_data, purchase_price, num_stocks, predicted_price)

            stock_predictions.append({
                "symbol": symbol,
                "profit_loss_details": profit_loss_data,
                "future_predictions": future_predictions
            })

    print("\nDetailed Stock Analysis:")
    for prediction in stock_predictions:
        details = prediction['profit_loss_details']
        print(f"\n{prediction['symbol']} Analysis:")
        print(f"Current Price: ${details['current_price']:.2f}")
        print(f"Purchase Price: ${details['purchase_price']:.2f}")
        print(f"Predicted Price: ${details['predicted_price']:.2f}")
        print(f"Purchase Total Value: ${details['purchase_total_value']:.2f}")
        print(f"Current Total Value: ${details['current_total_value']:.2f}")
        print(f"Unrealized Gain/Loss: ${details['unrealized_gain_loss']:.2f}")
        print(f"Potential Gain/Loss: ${details['potential_gain_loss']:.2f}")
        
        print("\nFuture Price Predictions:")
        for prediction in prediction['future_predictions']:
            date = prediction['date'].strftime('%Y-%m-%d')
            predicted_price = prediction['predicted_price']
            
            print(f"Date: {date} - Predicted Price: ${predicted_price:.2f}")



if __name__ == '__main__':
    main()
