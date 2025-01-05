import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_prev_day_data():
    def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD without using TA-Lib"""
        # Calculate EMAs
        exp1 = close_prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd = exp1 - exp2

        # Calculate signal line
        signal = macd.ewm(span=signal_period, adjust=False).mean()

        return macd, signal

    def calculate_rsi(close_prices, period=14):
        """Calculate RSI without using TA-Lib"""
        # Calculate price changes
        delta = close_prices.diff()

        # Split gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS
        rs = gain / loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_top_stocks():

        return [
            "ADANIGREEN.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
            "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BPCL.NS", "BHARTIARTL.NS",
            "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
            "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
            "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS",
            "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS",
            "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
            "ULTRACEMCO.NS", "WIPRO.NS"
        ]

    def get_stock_data(symbol, start_date, end_date):
        """Fetch 1-minute stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='5m')
            return df
        except Exception as e:
            return None

    def calculate_indicators(df):
        """Calculate MACD and RSI indicators"""
        # Calculate MACD
        df['macd'], df['macd_signal'] = calculate_macd(df['Close'])

        # Calculate RSI
        df['rsi'] = calculate_rsi(df['Close'])

        return df

    def generate_signals(df, weights):

        macd_weight = weights['macd']
        rsi_weight = weights['rsi']
        
        # Generate buy signals: Weighted conditions for MACD crossover and RSI < 30
        df['buy_signal'] = (
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & (df['macd'] >= df['macd_signal']) * macd_weight +
            (df['rsi'] < 30) * rsi_weight
        ) >= 1  # Ensure the combined weighted score is at least 1

        # Generate sell signals: Weighted conditions for MACD crossover and RSI > 70
        df['sell_signal'] = (
            (df['macd'].shift(1) > df['macd_signal'].shift(1)) & (df['macd'] <= df['macd_signal']) * macd_weight +
            (df['rsi'] > 70) * rsi_weight
        ) >= 1  # Ensure the combined weighted score is at least 1

        return df

    def execute_trades(results):
        """Simulate trades with 1 lakh capital based on buy/sell signals"""
        trade_results = []

        for symbol, data in results.items():
            capital = 100000  # Starting capital
            position = 0  # Current stock holding
            num_trades = 0
            profitable_trades = 0

            for record in data:
                if record['buy_signal'] and capital > 0:
                    position = capital / record['Close']  # Buy stocks
                    capital = 0
                    num_trades += 1

                elif record['sell_signal'] and position > 0:
                    capital = position * record['Close']  # Sell stocks
                    position = 0
                    num_trades += 1
                    if capital > 100000:
                        profitable_trades += 1

            final_returns = (capital + position * (data[-1]['Close'] if position > 0 else 0)) / 100000
            success_rate = (profitable_trades / num_trades * 100) if num_trades > 0 else 0

            trade_results.append({
                "symbol": symbol,
                "num_trades": num_trades,
                "profitable_trades": profitable_trades,
                "returns": final_returns,
                "success_rate": success_rate
            })

        return trade_results


    def trade_stocks():
        # Get the last trading day's date
        today = datetime.now()
        stock = yf.Ticker("RELIANCE.NS")  # Use a reliable stock symbol to get trading dates
        trading_data = stock.history(period="5d")  # Fetch the last 5 days' data
        last_trading_day = trading_data.index[-1].date()  # Get the most recent trading day

        stock_weights = {
            'RELIANCE.NS': {'macd': 0.6, 'rsi': 0.4},
            'TCS.NS': {'macd': 0.5, 'rsi': 0.5},
            'HDFCBANK.NS': {'macd': 0.4, 'rsi': 0.6},
            'ICICIBANK.NS': {'macd': 0.6, 'rsi': 0.4},
            'HINDUNILVR.NS': {'macd': 0.5, 'rsi': 0.5},
            'INFY.NS': {'macd': 0.4, 'rsi': 0.6},
            'SUNPHARMA.NS': {'macd': 0.6, 'rsi': 0.4},
            'ITC.NS': {'macd': 0.5, 'rsi': 0.5},
            'SBIN.NS': {'macd': 0.4, 'rsi': 0.6},
            'BHARTIARTL.NS': {'macd': 0.5, 'rsi': 0.5},
            'KOTAKBANK.NS': {'macd': 0.6, 'rsi': 0.4},
            'BAJFINANCE.NS': {'macd': 0.5, 'rsi': 0.5},
            'LICI.NS': {'macd': 0.4, 'rsi': 0.6},
            'LT.NS': {'macd': 0.5, 'rsi': 0.5},
            'M&M.NS': {'macd': 0.4, 'rsi': 0.6}
        }

        results = {}
        for symbol in get_top_stocks():
            # Get stock data for the last trading day
            start_date = last_trading_day
            end_date = last_trading_day + timedelta(days=1)

            df = get_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                continue

            # Calculate indicators
            df = calculate_indicators(df)

            # Generate signals
            weights = stock_weights.get(symbol, {'macd': 0.5, 'rsi': 0.5})
            df = generate_signals(df, weights)

            # Convert to JSON-serializable format
            results[symbol] = df[['Close', 'macd', 'macd_signal', 'rsi', 'buy_signal', 'sell_signal']].to_dict(orient='records')
        return results

    
    def calculate_total_investment_and_returns(trade_results):
        """Calculate total investment and returns across all trades"""
        total_investment = len(trade_results) * 100000  # 1 lakh per stock
        total_returns = sum(result['returns'] * 100000 for result in trade_results)
        
        # Calculate total trades and profitable trades
        total_trades = sum(result['num_trades'] for result in trade_results)
        total_profitable_trades = sum(result['profitable_trades'] for result in trade_results)
        
        # Calculate success rate
        success_rate = (total_profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_investment': total_investment,
            'total_returns': total_returns,
            'net_profit_loss': total_returns - total_investment,
            'total_trades': total_trades,
            'total_profitable_trades': total_profitable_trades,
            'success_rate': success_rate
        }

    results = trade_stocks()
    # df = pd.DataFrame(results)
    # df.to_csv('results.csv', index=False)
    trade_results = execute_trades(results)
    
    totals = calculate_total_investment_and_returns(trade_results)
        
    # Add totals to the response
    trade_results.append({
        'summary': totals
    })
    
    return trade_results      
