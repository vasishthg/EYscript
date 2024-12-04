import yfinance as yf
import pandas as pd
import datetime

def fetch_and_save_stock_data(symbols, start_date, end_date, filename):
    """
    Fetch historical stock data for multiple symbols and save it to a CSV file.
    """
    all_data = []
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if not stock_data.empty:
            stock_data['Symbol'] = symbol  # Add stock symbol for multi-stock data
            all_data.append(stock_data)

    # Concatenate data for all symbols into a single DataFrame
    combined_data = pd.concat(all_data)
    
    # Save the data to a CSV file
    combined_data.to_csv(filename)
    print(f"Data saved to {filename}")

# Example usage
stock_symbols = ["MSFT", "AAPL"]
start_date = datetime.date(2024, 5, 1)  # Choose your preferred start date
end_date = datetime.date(2024, 5, 10)  # Choose your preferred end date

fetch_and_save_stock_data(stock_symbols, start_date, end_date, "stock_data.csv")
