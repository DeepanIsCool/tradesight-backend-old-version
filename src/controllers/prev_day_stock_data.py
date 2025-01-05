import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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
        
        # Calculate histogram
        hist = macd - signal
        
        return macd, signal, hist

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
        """Get top 10 stocks by market cap from NSE"""
        # You might want to replace this with actual top stocks
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
                "INFY.NS", "SUNPHARMA.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", 
                "KOTAKBANK.NS", "BAJFINANCE.NS", "LICI.NS", "LT.NS", "M&M.NS"]

    def get_stock_data(symbol, start_date, end_date):
        """Fetch 1-minute stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='1m')
            return df
        except Exception as e:
            return None

    def calculate_indicators(df):
        """Calculate MACD and RSI indicators"""
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['Close'])
        
        return df

    def generate_signals(df, stock_weights):
        """Generate trading signals using weighted MACD and RSI"""
        macd_weight = stock_weights['macd']
        rsi_weight = stock_weights['rsi']
        
        # MACD signals (-1, 0, 1)
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1,
                                    np.where(df['macd'] < df['macd_signal'], -1, 0))
        
        # RSI signals (-1, 0, 1)
        df['rsi_signal'] = np.where(df['rsi'] > 70, -1,
                                np.where(df['rsi'] < 30, 1, 0))
        
        # Combine signals using weights and tanh
        df['combined_signal'] = np.tanh(
            macd_weight * df['macd_signal'] + rsi_weight * df['rsi_signal']
        )
        
        # Convert to discrete signals
        df['final_signal'] = np.where(df['combined_signal'] > 0.5, 1,
                                    np.where(df['combined_signal'] < -0.5, -1, 0))
        
        return df

    def backtest_strategy(df, initial_capital=100000):
        """Backtest the trading strategy"""
        position = 0
        balance = initial_capital
        portfolio = []
        trades = []
        profitable_trades = 0
        entry_price = 0
        
        for i in range(len(df)):
            signal = df['final_signal'].iloc[i]
            price = df['Close'].iloc[i]
            
            if signal == 1 and position == 0:  # Buy signal
                shares = int(balance / price)
                if shares > 0:
                    position = shares
                    entry_price = price
                    balance -= shares * price
                    trades.append({
                        'date': df.index[i],
                        'type': 'buy',
                        'shares': shares,
                        'price': price
                    })
            
            elif signal == -1 and position > 0:  # Sell signal
                profit = position * (price - entry_price)
                if profit > 0:
                    profitable_trades += 1
                    
                balance += position * price
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'shares': position,
                    'price': price,
                    'profit': profit
                })
                position = 0
            
            # Calculate portfolio value
            portfolio_value = balance + (position * price)
            portfolio.append({
                'date': df.index[i],
                'portfolio_value': portfolio_value
            })
        
        # Calculate success rate
        total_trades = len([t for t in trades if t['type'] == 'sell' or t['type'] == 'buy'])
        success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Return early with empty DataFrames if no trades occurred
        if len(trades) == 0:
            return pd.DataFrame(portfolio).set_index('date'), pd.DataFrame(), 0, 0
        
        # Create DataFrames
        portfolio_df = pd.DataFrame(portfolio).set_index('date')
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.set_index('date')
            
        return portfolio_df, trades_df, profitable_trades, success_rate

    def trade_stocks():
        # Set dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Stock-specific weights (example)
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
            # Get stock data
            df = get_stock_data(symbol, start_date, end_date)
            if df is None:
                continue
                
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Generate signals
            weights = stock_weights.get(symbol, {'macd': 0.5, 'rsi': 0.5})
            df = generate_signals(df, weights)
            
            # Backtest
            portfolio_df, trades_df, profitable_trades, success_rate = backtest_strategy(df)
            
            results[symbol] = {
                'portfolio': portfolio_df,
                'trades': trades_df,
                'profitable_trades': profitable_trades,
                'success_rate': success_rate
            }
        
        return results


    def process_stock_result(symbol_data):
        symbol, data = symbol_data
        initial_value = data['portfolio']['portfolio_value'].iloc[0]
        final_value = data['portfolio']['portfolio_value'].iloc[-1]
        returns = (final_value - initial_value) / initial_value * 100
        num_trades = len(data['trades'])
        profitable_trades = data['profitable_trades']
        success_rate = data['success_rate']
        
        return {
            'symbol': symbol,
            'returns': returns,
            'num_trades': num_trades,
            'profitable_trades': profitable_trades,
            'success_rate': success_rate
        }

    results = trade_stocks()

    # Process results using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        stock_results = list(executor.map(process_stock_result, results.items()))

    return stock_results
