import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define an empty dictionary to store buy information
bought_stocks = {}

def get_user_input():
  """
  Prompts user for stock information and returns a dictionary.
  """
  ticker = input("Enter stock ticker symbol: ")
  date_str = input("Enter buy date (YYYY-MM-DD): ")
  shares = int(input("Enter number of shares bought: "))
  buy_price = float(input("Enter buy price: "))
  return {"Ticker": ticker, "Shares": shares, "Buy Price": buy_price, "Date": date_str}

def update_bought_stocks(stock_info):
  """
  Updates the bought_stocks dictionary with user input.
  """
  bought_stocks[stock_info["Ticker"]] = stock_info

def analyze_stock(ticker, date_str):
  """
  Fetches data for a specific stock and buy date.
  """
  try:
    stock = yf.download(ticker, start=date_str, end=date_str)["Close"].iloc[0]
  except (KeyError, IndexError):
    print(f"No data found for {ticker} on {date_str}.")
    return None
  return stock

def analyze_portfolio():
  """
  Analyzes the bought stocks and returns a DataFrame.
  """
  results = []
  for ticker, info in bought_stocks.items():
    market_price = analyze_stock(ticker, info["Date"])
    if market_price is None:
      continue

    eps = yf.Ticker(ticker).info.get('trailingEps', None)
    pe_ratio = market_price / eps if eps and market_price else None
    profit_loss = (market_price - info["Buy Price"]) * info["Shares"]

    recommendation = "Hold"
    if pe_ratio and pe_ratio < 15:
      recommendation = "Buy - Undervalued"
    elif pe_ratio and pe_ratio > 30:
      recommendation = "Sell - Overvalued"

    results.append({
      **info,  # Include all info from bought_stocks
      "Market Price": market_price,
      "EPS": eps,
      "P/E Ratio": pe_ratio,
      "Profit/Loss": profit_loss,
      "Recommendation": recommendation
    })
  return pd.DataFrame(results)

# Rest of the code remains the same (save_results, plot_portfolio, main)

if __name__ == "__main__":
  while True:
    choice = input("Enter 'a' to add a purchase, 'v' to view analysis, or 'q' to quit: ")
    if choice.lower() == 'q':
      break
    elif choice.lower() == 'a':
      stock_info = get_user_input()
      update_bought_stocks(stock_info)
    elif choice.lower() == 'v':
      results = analyze_portfolio()
      if results is not None:
        print(results.to_string())  # Display analysis table
        # Rest of the functionality for saving and plotting (optional)