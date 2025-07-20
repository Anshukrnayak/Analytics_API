import os
import asyncio
import pandas as pd
import pandas_ta as ta
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alpaca API credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Simulate a data store (in production, use Redis or a database)
data_store = {}

# Trading client for account management and order execution
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

async def fetch_alpaca_data(tickers=["AAPL", "MSFT", "TSLA", "SPY"]):
    if not API_KEY or not API_SECRET:
        logger.error("ALPACA_API_KEY or ALPACA_API_SECRET not set")
        return
    
    client = StockDataStream(API_KEY, API_SECRET, feed=DataFeed.IEX)
    
    async def handle_bar(bar):
        symbol = bar.symbol
        data_store.setdefault(symbol, pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
        new_data = pd.DataFrame({
            "timestamp": [bar.timestamp],
            "open": [bar.open],
            "high": [bar.high],
            "low": [bar.low],
            "close": [bar.close],
            "volume": [bar.volume]
        })
        data_store[symbol] = pd.concat([data_store[symbol], new_data], ignore_index=True)
        data_store[symbol]["Returns"] = data_store[symbol]["close"].pct_change().fillna(0)
        data_store[symbol]["Volatility"] = data_store[symbol]["Returns"].rolling(window=20).std().fillna(0)
        data_store[symbol]["RSI"] = ta.rsi(data_store[symbol]["close"], length=14).fillna(0)
        data_store[symbol]["Sentiment"] = 0.5 * data_store[symbol]["RSI"] / 100 + 0.5 * data_store[symbol]["Volatility"] / data_store[symbol]["Volatility"].max()
        logger.info(f"Updated data for {symbol}: {data_store[symbol].iloc[-1]}")
    
    for ticker in tickers:
        client.subscribe_bars(handle_bar, ticker)
    
    await client.run()

def execute_order(symbol, qty, side, order_type="market"):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        response = trading_client.submit_order(order)
        logger.info(f"Order executed: {symbol}, {side}, qty={qty}, id={response.id}")
        return response
    except Exception as e:
        logger.error(f"Order execution failed for {symbol}: {str(e)}")
        return None

def get_account_info():
    try:
        account = trading_client.get_account()
        return {
            "account_number": account.account_number,
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "status": account.status
        }
    except Exception as e:
        logger.error(f"Failed to fetch account info: {str(e)}")
        return None

if __name__ == "__main__":
    # Start WebSocket data streaming
    asyncio.run(fetch_alpaca_data())