from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
from src.models.db_models import db_models
from db import fetch_Nifty_All_stocks, fetch_Nifty_50_stocks, fetch_Nifty_100_stocks, fetch_Nifty_200_stocks, fetch_Nifty_500_stocks
from concurrent.futures import ThreadPoolExecutor
from src.controllers.prev_day_stock_data import get_prev_day_data

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tradesight.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "TradeSight Backend API is running"
    })

    
@app.route('/api/create_db')
def create_tables():
    data = request.args.get('model')
    Stock_Model = db_models(db, data)
    try:
        # Create tables
        with app.app_context():
            db.create_all()
        
        return jsonify({"status": "success", "message": "Created the DB Successfully"}), 200
    
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

@app.route('/api/fetch_stocks')
def fetch_stocks():
    data = request.args.get('model')
    try:
        if data == 'nifty_all':
            stocks_df = fetch_Nifty_All_stocks(db, app)
        elif data == 'nifty_50':
            stocks_df = fetch_Nifty_50_stocks(db, app)
        elif data == 'nifty_100':
            stocks_df = fetch_Nifty_100_stocks(db, app)
        elif data == 'nifty_200':
            stocks_df = fetch_Nifty_200_stocks(db, app)
        elif data == 'nifty_500':
            stocks_df = fetch_Nifty_500_stocks(db, app)
        else:
            return jsonify({"status": "Failure", "message": "Invalid model"}), 400

        return jsonify({"status": "success", "data": stocks_df.to_dict(orient="records")}), 200
    
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

@app.route('/api/suggestions')
def get_suggestions():
    try:
        data = request.args.get('data')
        Stock_Model = db_models(db, "nifty_all")
        
        stocks = Stock_Model.query.filter(
            db.or_(
                Stock_Model.name.ilike(f'%{data}%'),
                Stock_Model.symbol.ilike(f'%{data}%')
            )
        ).all()
        
        def get_match_priority(stock):
            name_lower = stock.name.lower()
            symbol_lower = stock.symbol.lower()
            search_term = data.lower()
            
            if name_lower.startswith(search_term) or symbol_lower.startswith(search_term):
                return 0
            elif search_term in name_lower or search_term in symbol_lower:
                return 1
            else:
                return 2

        sorted_stocks = sorted(stocks, key=get_match_priority)
        results = [{"symbol": stock.symbol, "name": stock.name} for stock in sorted_stocks]
        
        return jsonify({"status": "Success", "data": results}), 200

    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

@app.route('/api/top_gainers')
def top_gainers():
    try:
        Stock_Model = db_models(db, "nifty_100")
        stocks = Stock_Model.query.all()
        symbols = [f"{stock.symbol}.NS" for stock in stocks]
        
        def fetch_stock_data(symbol):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return {
                        'symbol': symbol.replace('.NS', ''),
                        'price': hist['Close'].iloc[-1],
                        'change_percent': ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                    }
            except:
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(filter(None, executor.map(fetch_stock_data, symbols)))
        
        top_gainers = sorted(results, key=lambda x: x['change_percent'], reverse=True)[:10]
        
        return jsonify({
            "status": "success", 
            "data": top_gainers
        }), 200
        
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

@app.route('/api/top_losers')
def top_losers():
    try:
        Stock_Model = db_models(db, "nifty_100")
        stocks = Stock_Model.query.all()
        symbols = [f"{stock.symbol}.NS" for stock in stocks]
        
        def fetch_stock_data(symbol):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return {
                        'symbol': symbol.replace('.NS', ''),
                        'price': hist['Close'].iloc[-1],
                        'change_percent': ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                    }
            except:
                return None
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(filter(None, executor.map(fetch_stock_data, symbols)))
        
        top_losers = sorted(results, key=lambda x: x['change_percent'], reverse=False)[:10]
        
        return jsonify({
            "status": "success", 
            "data": top_losers
        }), 200

    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)})
    

@app.route('/api/top_volume')
def top_volume():
    try:
        Stock_Model = db_models(db, "nifty_100")
        stocks = Stock_Model.query.all()
        symbols = [f"{stock.symbol}.NS" for stock in stocks]
        
        def fetch_stock_data(symbol):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period='1d')
                if not hist.empty:
                    return {
                        'symbol': symbol.replace('.NS', ''),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'price': hist['Close'].iloc[-1],
                    }
            except:
                return None
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(filter(None, executor.map(fetch_stock_data, symbols)))
        
        top_volume = sorted(results, key=lambda x: x['volume'], reverse=True)[:10]
        
        return jsonify({
            "status": "success", 
            "data": top_volume
        }), 200
    
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

@app.route('/api/previous_day_gains_by_bot')
def previous_day_gains_by_symbol():
    try:
        data = get_prev_day_data()
        
        return jsonify({
            "status": "success",
            "data": data
        }), 200
    
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
