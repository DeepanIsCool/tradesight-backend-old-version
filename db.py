import requests
import pandas as pd
from io import StringIO
from src.models.db_models import db_models

def fetch_Nifty_All_stocks(db, app):
    """Fetch all stock listings from National Stock Exchange (NSE)"""
    
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    Stock_Model = db_models(db, "nifty_all")
    
    try:
        print("Fetching data from NSE...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        stocks_df = df[['SYMBOL', 'NAME OF COMPANY']].rename(
            columns={
                'SYMBOL': 'symbol',
                'NAME OF COMPANY': 'name'
            }
        )
        
        with app.app_context():
            for index, row in stocks_df.iterrows():
                stock = Stock_Model(
                    symbol=row['symbol'],
                    name=row['name']
                )
                db.session.add(stock)
            
            db.session.commit()
            print("Successfully inserted NSE stocks into database")
            
            stocks_count = Stock_Model.query.count()
            print(f"Total stocks in database: {stocks_count}")
        
        return stocks_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return None
    
def fetch_Nifty_500_stocks(db, app):
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    Stock_Model = db_models(db, "nifty_500")

    try:
        print("Fetching data from NSE...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        stocks_df = df[['Symbol', 'Company Name']].rename(
            columns={
                'Symbol': 'symbol',
                'Company Name': 'name'
            }
        )

        with app.app_context():
            for index, row in stocks_df.iterrows():
                stock = Stock_Model(
                    symbol=row['symbol'],
                    name=row['name']
                )
                db.session.add(stock)

            db.session.commit()
            print("Successfully inserted NSE stocks into database")
            stocks_count = Stock_Model.query.count()
            print(f"Total stocks in database: {stocks_count}")
            return stocks_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return None
    

def fetch_Nifty_100_stocks(db, app):
    """Fetch all stock listings from National Stock Exchange (NSE)"""

    url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    Stock_Model = db_models(db, "nifty_100")

    try:
        print("Fetching data from NSE...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        stocks_df = df[['Symbol', 'Company Name']].rename(
            columns={
                'Symbol': 'symbol',
                'Company Name': 'name'
            }
        )

        with app.app_context():
            for index, row in stocks_df.iterrows():
                stock = Stock_Model(
                    symbol=row['symbol'],
                    name=row['name']
                )
                db.session.add(stock)

            db.session.commit()
            print("Successfully inserted NSE stocks into database")
            stocks_count = Stock_Model.query.count()
            print(f"Total stocks in database: {stocks_count}")

            return stocks_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return None
    
def fetch_Nifty_50_stocks(db, app):
    """Fetch all stock listings from National Stock Exchange (NSE)"""

    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    Stock_Model = db_models(db, "nifty_50")

    try:
        print("Fetching data from NSE...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        stocks_df = df[['Symbol', 'Company Name']].rename(
            columns={
                'Symbol': 'symbol',
                'Company Name': 'name'
            }
        )

        with app.app_context():
            for index, row in stocks_df.iterrows():
                stock = Stock_Model(
                    symbol=row['symbol'],
                    name=row['name']
                )
                db.session.add(stock)

            db.session.commit
            print("Successfully inserted NSE stocks into database")
            stocks_count = Stock_Model.query.count()
            print(f"Total stocks in database: {stocks_count}")

            return stocks_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return None
    
def fetch_Nifty_200_stocks(db, app):
    """Fetch all stock listings from National Stock Exchange (NSE)"""

    url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    Stock_Model = db_models(db, "nifty_200")

    try:
        print("Fetching data from NSE...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        stocks_df = df[['Symbol', 'Company Name']].rename(
            columns={
                'Symbol': 'symbol',
                'Company Name': 'name'
            }
        )

        with app.app_context():
            for index, row in stocks_df.iterrows():
                stock = Stock_Model(
                    symbol=row['symbol'],
                    name=row['name']
                )
                db.session.add(stock)

            db.session.commit()
            print("Successfully inserted NSE stocks into database")
            stocks_count = Stock_Model.query.count()
            print(f"Total stocks in database: {stocks_count}")

            return stocks_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.session.rollback()
        return None
