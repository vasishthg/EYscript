import yfinance as yf

def fetch_stock_data(ticker):
    """
    Fetch stock data from Yahoo Finance.
    :param ticker: str, stock ticker symbol
    :return: tuple (market_price, eps) or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        market_price = stock.history(period="1d")['Close'].iloc[-1]  # Latest closing price
        eps = stock.info.get('trailingEps', None)  # Earnings per share (EPS)
        return market_price, eps
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_pe_ratio(market_price, eps):
    """
    Calculate the P/E ratio.
    :param market_price: float, market price of a single share
    :param eps: float, earnings per share (EPS)
    :return: float, P/E ratio
    """
    if eps is None or eps == 0:
        return None  # EPS must be valid and non-zero
    return market_price / eps

def main():
    # Define your portfolio
    portfolio = [
        {"ticker": "AAPL", "shares": 10},  # Example: Apple Inc.
        {"ticker": "MSFT", "shares": 15}, # Example: Microsoft
        {"ticker": "GOOGL", "shares": 5}, # Example: Alphabet Inc.
    ]

    print("P/E Ratios for your portfolio:")
    for stock in portfolio:
        ticker = stock['ticker']
        data = fetch_stock_data(ticker)
        if data:
            market_price, eps = data
            pe_ratio = calculate_pe_ratio(market_price, eps)
            if pe_ratio:
                print(f"{ticker}: Market Price = ${market_price:.2f}, EPS = ${eps:.2f}, P/E Ratio = {pe_ratio:.2f}")
            else:
                print(f"{ticker}: EPS not available or zero; cannot calculate P/E ratio.")
        else:
            print(f"{ticker}: Unable to fetch data.")