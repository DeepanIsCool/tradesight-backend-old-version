import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mydatabase.db")
DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY: str = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")  # Change this!
ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

class Settings:
    # Mock prices for testing - in production this would come from a real data source
    MOCK_PRICES = {
        "RELIANCE.NS": 2500.0,
        "TCS.NS": 3200.0,
        "HDFCBANK.NS": 1600.0,
        "INFY.NS": 1400.0,
        "HINDUNILVR.NS": 2400.0,
        "ICICIBANK.NS": 900.0,
        "KOTAKBANK.NS": 1800.0,
        "BHARTIARTL.NS": 800.0,
        "ITC.NS": 450.0,
        "SBIN.NS": 550.0,
        "ASIANPAINT.NS": 3000.0,
        "MARUTI.NS": 10000.0,
        "AXISBANK.NS": 1000.0,
        "LT.NS": 2800.0,
        "SUNPHARMA.NS": 1100.0,
        "TITAN.NS": 3200.0,
        "ULTRACEMCO.NS": 8000.0,
        "WIPRO.NS": 400.0,
        "NESTLEIND.NS": 22000.0,
        "POWERGRID.NS": 250.0,
        "NTPC.NS": 350.0,
        "TECHM.NS": 1600.0,
        "HCLTECH.NS": 1200.0,
        "BAJFINANCE.NS": 6500.0,
        "ONGC.NS": 250.0,
        "TATAMOTORS.NS": 900.0,
        "COALINDIA.NS": 400.0,
        "BAJAJFINSV.NS": 1500.0,
        "DRREDDY.NS": 1200.0,
        "EICHERMOT.NS": 4500.0,
        "GRASIM.NS": 2400.0,
        "BRITANNIA.NS": 4800.0,
        "CIPLA.NS": 1400.0,
        "DIVISLAB.NS": 5500.0,
        "HEROMOTOCO.NS": 4500.0,
        "JSWSTEEL.NS": 900.0,
        "HINDALCO.NS": 600.0,
        "INDUSINDBK.NS": 1000.0,
        "TATASTEEL.NS": 140.0,
        "ADANIENT.NS": 2500.0,
        "APOLLOHOSP.NS": 6000.0,
        "BAJAJ-AUTO.NS": 9000.0,
        "BPCL.NS": 300.0,
        "HDFCLIFE.NS": 650.0,
        "SBILIFE.NS": 1400.0,
        "TATACONSUM.NS": 900.0,
        "UPL.NS": 550.0,
        "M&M.NS": 2800.0,
        "ADANIPORTS.NS": 1200.0,
        "LTIM.NS": 5500.0
    }

settings = Settings()

