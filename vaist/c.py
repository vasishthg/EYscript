'''
This script is a simple stock price predictor that uses machine learning models to predict the next 5 days of stock prices for a given stock symbol. The script fetches historical stock data, calculates technical indicators, and trains Random Forest, XGBoost, and LSTM models to predict future stock prices. The script also provides an ensemble prediction using the three models.
'''

import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Machine Learning Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Additional ML Libraries
import xgboost as xgb

# Deep Learning Imports
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

class StockPredictor:
    def __init__(self):
        self.features = None  # Will be set during feature preparation

    def fetch_stock_data(self, symbol, start_date, end_date):
        try:
            # Extend end date to ensure we have enough recent data
            extended_end_date = end_date + datetime.timedelta(days=30)
            stock_data = yf.download(symbol, start=start_date, end=extended_end_date)
            
            if stock_data.empty:
                print(f"No data available for {symbol}")
                return pd.DataFrame()

            # Calculate technical indicators
            stock_data = self._calculate_technical_indicators(stock_data)
            stock_data.dropna(inplace=True)
            return stock_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, data):
        # Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp12 = data['Close'].ewm(span=12, adjust=False).mean()
        exp26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['%K'] = (data['Close'] - low_14) / (high_14 - low_14) * 100
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Upper_BB'] = rolling_mean + (2 * rolling_std)
        data['Lower_BB'] = rolling_mean - (2 * rolling_std)
        
        # Percentage Change and Lagged Features
        data['Pct_Change'] = data['Close'].pct_change()
        data['Target'] = data['Close'].shift(-1)
        
        # Add Lagged Features
        for lag in [1, 2, 3, 5, 10]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        
        return data

    def prepare_features(self, data):
        # Comprehensive Feature Selection
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'MA20', 'MA50', 'MA100', 'MA200', 
            'EMA20', 'EMA50',
            'RSI', 'MACD', 'MACD_Signal', 
            '%K', '%D', 
            'Upper_BB', 'Lower_BB',
            'Pct_Change'
        ]
        
        # Add Lagged Features
        lag_features = [col for col in data.columns if 'Lag' in col]
        features.extend(lag_features)
        
        # Store features for later use
        self.features = features
        
        X = data[features]
        y = data['Target']
        return X, y

    def train_random_forest(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Comprehensive Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf_grid = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring='neg_mean_squared_error'
        )

        rf_grid.fit(X_train_scaled, y_train)
        best_model = rf_grid.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Best Random Forest Parameters:", rf_grid.best_params_)
        print(f"Random Forest RMSE: {rmse:.4f}")
        print(f"Random Forest MAPE: {mape * 100:.2f}%")

        return best_model, scaler

    def train_xgboost(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # XGBoost with more parameters
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)

        y_pred = xgb_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f"XGBoost RMSE: {rmse:.4f}")
        print(f"XGBoost MAPE: {mape * 100:.2f}%")

        return xgb_model, scaler

    def train_lstm(self, data, lookback=60):
        # Prepare LSTM data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Close']])

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshape for LSTM input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Advanced LSTM Model
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.3),
            LSTM(units=100, return_sequences=True),
            Dropout(0.3),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=50, activation='relu'),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Callbacks for improved training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_lstm_model.keras', save_best_only=True)
        
        # Train the model
        model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=32, 
            validation_split=0.2, 
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f"LSTM RMSE: {rmse:.4f}")
        print(f"LSTM MAPE: {mape * 100:.2f}%")

        return model, scaler

    def predict_next_days(self, model, scaler, latest_data, method='random_forest', lookback=60, days=5):
        predictions = []
        last_data = latest_data.copy()

        for _ in range(days):
            if method == 'lstm':
                # LSTM prediction
                last_sequence = scaler.transform(last_data[['Close']].values[-lookback:].reshape(-1, 1))
                next_pred = model.predict(np.array([last_sequence.reshape(lookback, 1)]))
                next_value = scaler.inverse_transform(next_pred)[0][0]
            else:
                # Random Forest and XGBoost prediction
                latest_features = last_data[self.features].iloc[-1].values.reshape(1, -1)
                scaled_features = scaler.transform(latest_features)
                next_value = model.predict(scaled_features)[0]

            predictions.append(next_value)
            
            # Create new row and append to dataset
            new_row = last_data.iloc[-1].copy()
            new_row['Close'] = next_value
            last_data = pd.concat([last_data, new_row.to_frame().T], ignore_index=True)

        return predictions

    def ensemble_prediction(self, models, scalers, latest_data, days=5):
        ensemble_predictions = []
        
        for _ in range(days):
            model_predictions = []
            for model, scaler in zip(models, scalers):
                # Check if scaler is MinMaxScaler (for LSTM) or StandardScaler
                if isinstance(scaler, MinMaxScaler):
                    # For LSTM, use only Close price
                    last_sequence = scaler.transform(latest_data[['Close']].values[-60:].reshape(-1, 1))
                    next_pred = model.predict(np.array([last_sequence.reshape(60, 1)]))
                    next_value = scaler.inverse_transform(next_pred)[0][0]
                else:
                    # For Random Forest and XGBoost
                    latest_features = latest_data[self.features].iloc[-1].values.reshape(1, -1)
                    scaled_features = scaler.transform(latest_features)
                    next_value = model.predict(scaled_features)[0]

                model_predictions.append(next_value)
            
            # Take weighted average or median of predictions
            ensemble_pred = np.median(model_predictions)
            ensemble_predictions.append(ensemble_pred)
            
            # Update latest_data with new prediction
            new_row = latest_data.iloc[-1].copy()
            new_row['Close'] = ensemble_pred
            latest_data = pd.concat([latest_data, new_row.to_frame().T], ignore_index=True)
        
        return ensemble_predictions

    def plot_stock_data(self, data, symbol):
        plt.figure(figsize=(14, 7))

        plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.8)
        plt.plot(data.index, data['MA50'], label='50-day MA', color='orange', linestyle='--', alpha=0.8)
        plt.plot(data.index, data['MA200'], label='200-day MA', color='green', linestyle='--', alpha=0.8)

        plt.title(f'{symbol} Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

def main():
    predictor = StockPredictor()

    # Get user input
    symbol = input("Enter the stock symbol: ").strip().upper()
    
    # Set date range (3 years of historical data)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=2000)

    # Fetch stock data
    stock_data = predictor.fetch_stock_data(symbol, start_date, end_date)
    if stock_data.empty:
        print("No data available for the stock.")
        return

    # Visualize stock data
    predictor.plot_stock_data(stock_data, symbol)

    # Prepare features
    X, y = predictor.prepare_features(stock_data)

    # Train models
    print("\nTraining Random Forest...")
    rf_model, rf_scaler = predictor.train_random_forest(X, y)

    print("\nTraining XGBoost...")
    xgb_model, xgb_scaler = predictor.train_xgboost(X, y)

    print("\nTraining LSTM...")
    lstm_model, lstm_scaler = predictor.train_lstm(stock_data)

    # Predict next 5 days using individual models
    print("\nPredicting next 5 days' prices using each model:")
    dates = [end_date + datetime.timedelta(days=i + 1) for i in range(5)]
    
    rf_predictions = predictor.predict_next_days(rf_model, rf_scaler, stock_data, method='random_forest', days=5)
    xgb_predictions = predictor.predict_next_days(xgb_model, xgb_scaler, stock_data, method='xgboost', days=5)
    lstm_predictions = predictor.predict_next_days(lstm_model, lstm_scaler, stock_data, method='lstm', days=5)

    # Ensemble prediction
    print("\nEnsemble Prediction:")
    ensemble_models = [rf_model, xgb_model, lstm_model]
    ensemble_scalers = [rf_scaler, xgb_scaler, lstm_scaler]
    ensemble_predictions = predictor.ensemble_prediction(ensemble_models, ensemble_scalers, stock_data, days=5)

    # Print predictions
    for i in range(5):
        print(f"\nDate: {dates[i]}")
        print(f"  Random Forest Prediction: ${rf_predictions[i]:.2f}")
        print(f"  XGBoost Prediction: ${xgb_predictions[i]:.2f}")
        print(f"  LSTM Prediction: ${lstm_predictions[i]:.2f}")
        print(f"  Ensemble Prediction: ${ensemble_predictions[i]:.2f}")

if __name__ == '__main__':
    main()