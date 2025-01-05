from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
import pandas as pd
from src.models.db_models import db_models
from db import fetch_Nifty_All_stocks, fetch_Nifty_50_stocks, fetch_Nifty_100_stocks, fetch_Nifty_200_stocks, fetch_Nifty_500_stocks
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
from datetime import datetime, timedelta
from src.controllers.prev_day_stock_data import get_prev_day_data
from src.controllers.last_trading_day_data import get_last_trading_day
import requests
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {  # Apply to all routes under /api/
        "origins": ["*"],  # Allow your frontend origin
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True  # If you need to support credentials
    }
})

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
    
@app.route('/api/indices-data', methods=['GET'])
def get_indices_data():
    indices = {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
    
    data = []
    # Fetch history for the last 5 days to cover weekends and holidays
    today = datetime.now().date()
    start_date = today - timedelta(days=7)
    
    for name, symbol in indices.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
        print(hist)  # Debug print to verify the data
        
        if not hist.empty:
            last_close, prev_close = get_last_trading_day(hist)
            if last_close is not None and prev_close is not None:
                change = round(((last_close - prev_close) / prev_close) * 100, 2) if prev_close != 0 else 0.0
                data.append({"name": name, "change": change})
    
    return jsonify(data)

@app.route('/api/company-details', methods=['GET'])
def get_company_details():
    company_name = request.args.get('company') 
    
    if not company_name:
        return jsonify({"status": "Failure", "message": "Company name is required"}), 400
    
    try:
        search_url = f"https://ticker.finology.in/company/{company_name}"
        page=requests.get(search_url)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'html.parser')

        table=soup.find('div',id="mainContent_updAddRatios")
        if not table:
            return jsonify({'error': 'Unable to find the data section on the webpage. Verify the company name.'}), 404
        ratios=table.find_all('div')
        
        company_details = {}
        for ratio in ratios:
            try:
                key = ratio.find('small').text.strip()
                value = ratio.find('p').text.strip()
                key = key.replace('\xa0', ' ').strip()
                value = value.replace('\xa0', '').replace('\r', '').replace('\n', '').strip()
                company_details[key] = value
            except AttributeError:
                continue
        cleaned_details = {}
        for key, value in company_details.items():
            cleaned_value = re.sub(r'\s+', ' ', value).strip()
            cleaned_details[key] = cleaned_value

        return jsonify({'company_details': cleaned_details}), 200
    
    except requests.exceptions.HTTPError as http_err:
        return jsonify({'error': f'HTTP error occurred: {http_err}'}), 500

    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    
@app.route('/api/price-summery', methods=['GET'])
def get_price_summery():
    company_name = request.args.get('company') 
    
    if not company_name:
        return jsonify({"status": "Failure", "message": "Company name is required"}), 400
    
    try:
        search_url = f"https://ticker.finology.in/company/{company_name}"
        page=requests.get(search_url)
        page.raise_for_status()
        soup=BeautifulSoup(page.text,'html.parser')

        table=soup.find('div',id="mainContent_pricesummary").find('div',class_="row no-gutters")
        if not table:
            return jsonify({'error': 'Unable to find the data section on the webpage. Verify the company name.'}), 404
        table_details=table.find_all('div',class_="col-6 col-md-3 compess")
        
        price_details={}
        for detail in table_details:
            key=detail.find('small').text.strip()
            value=detail.find('p').text.strip().replace('\xa0', '')
            price_details[key]=value

        return jsonify({'price_details': price_details}), 200
    
    except requests.exceptions.HTTPError as http_err:
        return jsonify({'error': f'HTTP error occurred: {http_err}'}), 500

    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 500
    

# for investors details
def capitalize_words(text):
    return text.title()

def extract_table_data_with_headers(table):
    headers = [capitalize_words(header.text.strip()) for header in table.find("thead").find_all("th")]
    
    rows = table.find("tbody").find_all("tr")
    data = [headers]  
    for row in rows:
        cols = [capitalize_words(col.text.strip()) for col in row.find_all("td")]
        data.append(cols)
    return data

def scrape_data(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')

    promoter_tab = soup.find("div", id="pills-Promoter")
    promoter_table = promoter_tab.find("table") if promoter_tab else None
    promoter_data = []
    if promoter_table:
        promoter_data = extract_table_data_with_headers(promoter_table)

    investor_tab = soup.find("div", id="pills-Investors")
    investor_table = investor_tab.find("table") if investor_tab else None
    investor_data = []
    if investor_table:
        investor_data = extract_table_data_with_headers(investor_table)

    output = {
        "Promoter": promoter_data,
        "Investor": investor_data
    }
    return output

@app.route('/api/investors-details', methods=['GET'])
def scrape():
    company_name = request.args.get('company')
    if not company_name:
        return jsonify({"status": "Failure", "message": "Company name is required"}), 400

    try:
        url = f"https://ticker.finology.in/company/{company_name}"
        data = scrape_data(url)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
