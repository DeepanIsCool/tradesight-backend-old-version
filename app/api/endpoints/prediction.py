from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from app.database.models import StockIndex
from fastapi import APIRouter, HTTPException, Path, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union
from datetime import datetime
from app.schemas.prediction import PredictionSignal
from app.services.prediction_service import get_latest_predictions
from sqlalchemy.orm import Session
from app.database.models import StockIndex, IndexStock
from app.database.database import get_db

router = APIRouter()

# --- Configuration --- (Keep as is)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "app/models_store")
DEFAULT_MODEL_NAME = "TCS.NS_model.h5"

# Function to get NIFTY_50 stocks from database
def get_nifty_50_stocks(db: Session):
    nifty_index = db.query(StockIndex).filter(StockIndex.name == "NIFTY_50").first()
    if not nifty_index:
        # Fallback to hardcoded list if not found in DB
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ITC.NS',
            'AXISBANK.NS', 'MARUTI.NS', 'LT.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
            'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'POWERGRID.NS',
            'TECHM.NS', 'WIPRO.NS', 'INDUSINDBK.NS', 'DIVISLAB.NS', 'JSWSTEEL.NS',
            'BHARTIARTL.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'TATAMOTORS.NS',
            'GRASIM.NS', 'ADANIPORTS.NS', 'CIPLA.NS', 'DRREDDY.NS', 'BRITANNIA.NS',
            'EICHERMOT.NS', 'SBILIFE.NS', 'HEROMOTOCO.NS', 'ONGC.NS', 'COALINDIA.NS',
            'UPL.NS', 'BPCL.NS', 'SHREECEM.NS', 'NTPC.NS', 'TATASTEEL.NS',
            'HINDALCO.NS', 'IOC.NS', 'ADANIENT.NS', 'GAIL.NS', 'VEDL.NS'
        ]
    
    active_stocks = db.query(IndexStock).filter(
        IndexStock.index_id == nifty_index.id,
        IndexStock.is_active == True
    ).all()
    
    return [stock.symbol for stock in active_stocks]

# --- Prediction Endpoint ---
@router.post("/predict")
async def predict_signals(
    ticker: str = Query(
        ..., title="Stock Ticker Symbol",
        description="The ticker symbol for the stock (e.g., 'TCS.NS', 'RELIANCE.NS', 'AAPL').",
        min_length=1
    ),
    prev: Optional[bool] = Query(
        False, title="Include Previous Data",
        description="If true, includes OHLC data for previous timestamps."
    )
):
    """
    Predicts trading signals for the given stock ticker for the latest available day.

    - **ticker**: The stock ticker symbol (required).
    - **prev**: Whether to include historical OHLC data (default False).
    """
    MODEL_NAME=f"{ticker}_model.h5"
    # Check if the model file exists in the models directory
    if os.path.exists(os.path.join(MODELS_DIR, MODEL_NAME)):
        print(f"Found model for {ticker}")
        model_filename = MODEL_NAME
        print(f"Model filenameeeee: {model_filename}")
    else:
        print("Model not found for this stock, using default model")
        model_filename = DEFAULT_MODEL_NAME
    model_path = os.path.join(MODELS_DIR, model_filename)

    print(f"Model pathhhhhh: {model_path}")

    try:
        print(f"Received prediction request for ticker: {ticker}")
        prediction_df = get_latest_predictions(ticker=ticker, model_path=model_path)
        print(f"Raw Prediction DataFrame for {ticker}: {prediction_df}")

        if not isinstance(prediction_df, dict):
            print("Error: Prediction function did not return a dictionary!")
            return []

        if not prediction_df:
            print(f"No predictions generated for {ticker} for the latest date.")
            return []

        formatted_prediction = {
            "Timestamp": prediction_df["Date"],
            "Open": prediction_df["Open"] if isinstance(prediction_df["Open"], list) else prediction_df["Open"],
            "High": prediction_df["High"],
            "Low": prediction_df["Low"],
            "Close": prediction_df["Close"],
            "Volume": prediction_df["Volume"],
            "Signal": prediction_df["Signal"],
            "Model_used": model_filename
        }

        response_data = {
            'current_prediction': formatted_prediction
        }

        if prev:
            prev_data_path = f"../prediction_logs/{ticker}/{ticker}_predictions.csv"
            if os.path.exists(prev_data_path):
                df = pd.read_csv(prev_data_path)

                # Remove brackets and convert to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].apply(lambda x: float(x.strip("[]")))

                # Exclude the latest row (which is already returned as 'current_prediction')
                prev_df = df.iloc[:-1]

                # Extract only necessary columns
                previous_ohlc = prev_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records')
                response_data['previous_data'] = previous_ohlc

        return JSONResponse({'data': response_data, 'status_code': 200})

    except Exception as e:
        print(f"Error: Unexpected error during prediction - {e}")
        return JSONResponse({'data': {}, 'status_code': 500})
    
@router.get("/predict/all")
async def predict_all_stocks():
    def process_stock(ticker):
        try:
            model_file = f"{ticker}_model.h5"
            model_path = os.path.join(MODELS_DIR, model_file)
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)
                model_file = DEFAULT_MODEL_NAME

            prediction = get_latest_predictions(ticker=ticker, model_path=model_path)

            if not prediction or not isinstance(prediction, dict):
                return (ticker, None)

            return (ticker, {
                "Timestamp": prediction["Date"],
                "Open": prediction["Open"],
                "High": prediction["High"],
                "Low": prediction["Low"],
                "Close": prediction["Close"],
                "Volume": prediction["Volume"],
                "Signal": prediction["Signal"],
                "Model_used": model_file
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return (ticker, None)

    results = {}
    with ThreadPoolExecutor(max_workers=24) as executor:
        future_to_ticker = {executor.submit(process_stock, ticker): ticker for ticker in NIFTY_50}
        for future in as_completed(future_to_ticker):
            ticker, prediction = future.result()
            if prediction:
                results[ticker] = prediction
            else:
                results[ticker] = {"error": "Prediction failed or data unavailable"}

    return JSONResponse(content={"data": results, "status_code": 200})