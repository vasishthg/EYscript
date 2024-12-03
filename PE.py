import requests
import pandas as pd
import matplotlib.pyplot as plt

# Define your Alpha Vantage API key here
ALPHA_VANTAGE_API_KEY = 'HTFN876ODMTYPN40'  # Replace with your actual API key

# Define an empty dictionary to store buy information
bought_stocks = {
    'MSFT': {
        'Ticker': 'MSFT', 
        'Shares': 10, 
        'Buy Price': 200, 
        'Date': '2024-12-02'
    }
}


def get_user_input():
    """
    Prompts user for stock information and returns a dictionary.
    """
    while True:
        ticker = input("Enter stock ticker symbol: ")
        if ticker.isalpha():
            break
        else:
            print("Invalid input. Please enter a valid ticker symbol.")
    
    while True:
        date_str = input("Enter buy date (YYYY-MM-DD): ")
        try:
            pd.to_datetime(date_str)
            break
        except ValueError:
            print("Invalid input. Please enter a valid date in YYYY-MM-DD format.")
    
    while True:
        try:
            shares = int(input("Enter number of shares bought: "))
            if shares <= 0:
                print("Invalid input. Please enter a positive number of shares.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid number of shares.")
    
    while True:
        try:
            buy_price = float(input("Enter buy price: "))
            if buy_price <= 0:
                print("Invalid input. Please enter a positive buy price.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid buy price.")
    
    return {"Ticker": ticker, "Shares": shares, "Buy Price": buy_price, "Date": date_str}

def update_bought_stocks(stock_info):
    """
    Updates the bought_stocks dictionary with user input.
    """
    if stock_info["Ticker"] in bought_stocks:
        print("Stock already exists in the portfolio. Updating the information.")
    bought_stocks[stock_info["Ticker"]] = stock_info

def fetch_stock_data(ticker, date_str):
    """
    Fetches data from Alpha Vantage for a specific stock and date.
    """
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'Time Series (Daily)' not in data:
            print(f"Error fetching data for {ticker} on {date_str}: {data.get('Note', 'No data available')}")
            return None
        
        time_series = data['Time Series (Daily)']
        if date_str in time_series:
            closing_price = float(time_series[date_str]['4. close'])
            return closing_price
        else:
            print(f"No data found for {ticker} on {date_str}.")
            return None
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date_str}: {e}")
        return None

def analyze_portfolio():
    """
    Analyzes the bought stocks and returns a DataFrame.
    """
    results = []
    for ticker, info in bought_stocks.items():
        market_price = fetch_stock_data(ticker, info["Date"])
        if market_price is None:
            continue

        # Alpha Vantage doesn't provide EPS directly, so we skip P/E calculation
        eps = None
        pe_ratio = None
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

def save_results(results):
    """
    Saves the analysis results to a CSV file.
    """
    results.to_csv("portfolio_analysis.csv", index=False)
    print("Results saved to portfolio_analysis.csv")

def plot_portfolio(results):
    """
    Plots the portfolio analysis.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(results["Ticker"], results["Profit/Loss"])
    plt.xlabel("Ticker")
    plt.ylabel("Profit/Loss")
    plt.title("Portfolio Analysis")
    plt.show()

if __name__ == "__main__":
    while True:
        choice = input("Enter 'a' to add a purchase, 'v' to view analysis, 's' to save results, 'p' to plot portfolio, or 'q' to quit: ")
        if choice.lower() == 'q':
            break
        elif choice.lower() == 'a':
            stock_info = get_user_input()
            update_bought_stocks(stock_info)
        elif choice.lower() == 'v':
            results = analyze_portfolio()
            if not results.empty:
                print(results.to_string())  # Display analysis table
        elif choice.lower() == 's':
            results = analyze_portfolio()
            if not results.empty:
                save_results(results)
        elif choice.lower() == 'p':
            results = analyze_portfolio()
            if not results.empty:
                plot_portfolio(results)
