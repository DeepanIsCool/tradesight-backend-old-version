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
    
    def calculate_obv(close_prices, volumes):
        obv = np.where(close_prices > close_prices.shift(1), volumes, np.where(close_prices < close_prices.shift(1), -volumes, 0))
        return obv.cumsum()

    def get_top_stocks():

        return [
            'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 
        'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',
        'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
        ]

    def get_stock_data(symbol, start_date, end_date):
        """Fetch 1-minute stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='1d')
            return df
        except Exception as e:
            return None

    def calculate_indicators(df):
        """Calculate MACD and RSI indicators"""
        # Calculate MACD
        df['macd'], df['macd_signal'] = calculate_macd(df['Close'])

        # Calculate RSI
        df['rsi'] = calculate_rsi(df['Close'])

        df['obv'] = calculate_obv(df['Close'], df['Volume'])

        return df

    def generate_signals(df, weights):

        macd_weight = weights['macd']
        rsi_weight = weights['rsi']
        obv_weight = weights['obv']
        
        # Generate buy signals: Weighted conditions for MACD crossover and RSI < 30
        df['buy_signal'] = (
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & (df['macd'] >= df['macd_signal']) * macd_weight +
            (df['rsi'] < 70) * rsi_weight & df['obv'] > 0 * obv_weight
        ) >= 1  # Ensure the combined weighted score is at least 1

        # Generate sell signals: Weighted conditions for MACD crossover and RSI > 70
        df['sell_signal'] = (
            (df['macd'].shift(1) > df['macd_signal'].shift(1)) & (df['macd'] <= df['macd_signal']) * macd_weight +
            (df['rsi'] > 30) * rsi_weight & df['obv'] * obv_weight
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
            'ADANIENT.NS': {'macd': -0.0924, 'rsi': -0.2291, 'obv': 1.3214},
            'ADANIPORTS.NS': {'macd': -0.0601, 'rsi': 0.7466, 'obv': 0.3135},
            'APOLLOHOSP.NS': {'macd': 0.1681, 'rsi': 0.5224, 'obv': 0.3095},
            'ASIANPAINT.NS': {'macd': 0.0034, 'rsi': 0.5540, 'obv': 0.4426},
            'AXISBANK.NS': {'macd': 0.1863, 'rsi': 0.4888, 'obv': 0.3249},
            'BAJAJ-AUTO.NS': {'macd': 0.0634, 'rsi': -0.0868, 'obv': 1.0235},
            'BAJFINANCE.NS': {'macd': 0.9914, 'rsi': -0.3934, 'obv': 0.4020},
            'BAJAJFINSV.NS': {'macd': 0.4018, 'rsi': -0.9254, 'obv': 1.5236},
            'BEL.NS': {'macd': 0.4969, 'rsi': 0.2071, 'obv': 0.2960},
            'BPCL.NS': {'macd': 0.4055, 'rsi': 0.2034, 'obv': 0.3911},
            'BHARTIARTL.NS': {'macd': -0.4589, 'rsi': 1.4404, 'obv': 0.0185},
            'BRITANNIA.NS': {'macd': 0.2827, 'rsi': 0.2400, 'obv': 0.4774},
            'CIPLA.NS': {'macd': -0.3384, 'rsi': 0.0657, 'obv': 1.2726},
            'COALINDIA.NS': {'macd': 0.0651, 'rsi': 0.4928, 'obv': 0.4421},
            'DRREDDY.NS': {'macd': -0.2207, 'rsi': -0.1389, 'obv': 1.3596},
            'EICHERMOT.NS': {'macd': 0.2893, 'rsi': 0.1187, 'obv': 0.5920},
            'GRASIM.NS': {'macd': -0.6447, 'rsi': 2.2561, 'obv': -0.6114},
            'HCLTECH.NS': {'macd': -0.2809, 'rsi': 1.0785, 'obv': 0.2024},
            'HDFCBANK.NS': {'macd': -1.0983, 'rsi': 0.6041, 'obv': 1.4942},
            'HDFCLIFE.NS': {'macd': -6.1898, 'rsi': 0.1393, 'obv': 7.0505},
            'HEROMOTOCO.NS': {'macd': 0.2988, 'rsi': 0.4808, 'obv': 0.2205},
            'HINDALCO.NS': {'macd': 0.0906, 'rsi': -0.2384, 'obv': 1.1478},
            'HINDUNILVR.NS': {'macd': -0.0567, 'rsi': 0.5606, 'obv': 0.4961},
            'ICICIBANK.NS': {'macd': 0.1211, 'rsi': 0.2393, 'obv': 0.6396},
            'ITC.NS': {'macd': 0.1430, 'rsi': 0.2569, 'obv': 0.6001},
            'INDUSINDBK.NS': {'macd': 0.0432, 'rsi': 0.2051, 'obv': 0.7517},
            'INFY.NS': {'macd': 0.0109, 'rsi': 0.2272, 'obv': 0.7619},
            'JSWSTEEL.NS': {'macd': -1.7790, 'rsi': -0.9649, 'obv': 3.7439},
            'KOTAKBANK.NS': {'macd': -0.0096, 'rsi': 0.1553, 'obv': 0.8543},
            'LT.NS': {'macd': 0.0025, 'rsi': 0.2662, 'obv': 0.7313},
            'M&M.NS': {'macd': 0.3581, 'rsi': 0.0542, 'obv': 0.5877},
            'MARUTI.NS': {'macd': 0.4491, 'rsi': -0.0918, 'obv': 0.6427},
            'NTPC.NS': {'macd': 0.1054, 'rsi': 0.7292, 'obv': 0.1654},
            'NESTLEIND.NS': {'macd': 0.1186, 'rsi': 0.5294, 'obv': 0.3520},
            'ONGC.NS': {'macd': 0.0129, 'rsi': 0.4115, 'obv': 0.5756},
            'POWERGRID.NS': {'macd': 0.3072, 'rsi': 0.7155, 'obv': -0.0227},
            'RELIANCE.NS': {'macd': 3.2604, 'rsi': 1.4752, 'obv': -3.7357},
            'SBILIFE.NS': {'macd': 0.0201, 'rsi': 0.8585, 'obv': 0.1215},
            'SHRIRAMFIN.NS': {'macd': 0.2207, 'rsi': 0.3234, 'obv': 0.4558},
            'SBIN.NS': {'macd': 0.3066, 'rsi': -0.3846, 'obv': 1.0781},
            'SUNPHARMA.NS': {'macd': 0.1835, 'rsi': 0.4710, 'obv': 0.3455},
            'TCS.NS': {'macd': 0.4241, 'rsi': 0.1568, 'obv': 0.4191},
            'TATACONSUM.NS': {'macd': 0.7298, 'rsi': -0.4907, 'obv': 0.7610},
            'TATAMOTORS.NS': {'macd': 0.3362, 'rsi': 0.0168, 'obv': 0.6470},
            'TATASTEEL.NS': {'macd': 0.0505, 'rsi': -0.4761, 'obv': 1.4255},
            'TECHM.NS': {'macd': 0.8604, 'rsi': -0.9295, 'obv': 1.0691},
            'TITAN.NS': {'macd': -0.5167, 'rsi': 3.8376, 'obv': -2.3208},
            'TRENT.NS': {'macd': 0.3679, 'rsi': 0.4782, 'obv': 0.1539},
            'ULTRACEMCO.NS': {'macd': 0.3488, 'rsi': 0.2173, 'obv': 0.4339},
            'WIPRO.NS': {'macd': -0.2881, 'rsi': 2.0849, 'obv': -0.7968},
        }

        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in get_top_stocks():
                # Get stock data for the last trading day
                # Calculate start and end dates based on current date
                start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')  # Last 30 days
                end_date = today.strftime('%Y-%m-%d')  # Today's date

                futures.append(executor.submit(get_stock_data, symbol, start_date, end_date))

            for future, symbol in zip(futures, get_top_stocks()):
                df = future.result()
                if df is None or df.empty:
                    continue
                df = calculate_indicators(df)
                weights = stock_weights.get(symbol, {'macd': 1, 'rsi': 1})
                df = generate_signals(df, weights)
                results[symbol] = df[['Close', 'macd', 'macd_signal', 'rsi', 'buy_signal', 'sell_signal']].to_dict(orient='records')
                df.to_csv(f'data/{symbol}_prev_day_data.csv', index=False)
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
