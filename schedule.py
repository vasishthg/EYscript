import schedule
import time
from datetime import datetime
from plyer import notification

# Sample data structure to hold transactions
transactions = []

def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name="Stock Market Notifications",
        timeout=10
    )

def market_open():
    send_notification("Market Status", "The stock market is now open for trading.")

def market_close():
    send_notification("Market Status", "The stock market has closed for trading.")
    summarize_transactions()

def summarize_transactions():
    if not transactions:
        send_notification("Daily Summary", "No transactions were made today.")
        return
    
    total_transactions = len(transactions)
    total_profit_loss = sum(tx['Profit/Loss'] for tx in transactions)
    
    highest_gainer = max(transactions, key=lambda x: x['Profit/Loss'])
    highest_loser = min(transactions, key=lambda x: x['Profit/Loss'])
    
    summary_message = (
        f"Total Transactions: {total_transactions}\n"
        f"Total Profit/Loss: {total_profit_loss:.2f}\n"
        f"Highest Gainer: {highest_gainer['Ticker']} with Profit/Loss: {highest_gainer['Profit/Loss']:.2f}\n"
        f"Highest Loser: {highest_loser['Ticker']} with Profit/Loss: {highest_loser['Profit/Loss']:.2f}"
    )
    
    send_notification("Daily Summary", summary_message)

def record_transaction(ticker, shares, buy_price, market_price):
    profit_loss = (market_price - buy_price) * shares
    transactions.append({
        'Ticker': ticker,
        'Shares': shares,
        'Buy Price': buy_price,
        'Market Price': market_price,
        'Profit/Loss': profit_loss
    })

# Schedule market open and close times
schedule.every().monday.at("09:30").do(market_open)
schedule.every().monday.at("16:00").do(market_close)
schedule.every().tuesday.at("09:30").do(market_open)
schedule.every().tuesday.at("16:00").do(market_close)
schedule.every().wednesday.at("09:30").do(market_open)
schedule.every().wednesday.at("16:00").do(market_close)
schedule.every().thursday.at("09:30").do(market_open)
schedule.every().thursday.at("16:00").do(market_close)
schedule.every().friday.at("09:30").do(market_open)
schedule.every().friday.at("16:00").do(market_close)

# Example of recording a transaction
# record_transaction('AAPL', 5, 1000, 153.71)  # Example transaction

while True:
    schedule.run_pending()
    time.sleep(1)