from fastapi import APIRouter, HTTPException, Path, Depends, Query
from typing import Dict, List, Union

from app.schemas.stock_models import (
    StockPrice, 
    StockPriceResponse, 
    AccountResponse, 
    NewAccountResponse, 
    ExistingAccountResponse,
    WatchlistResponse,
    EmptyWatchlistResponse
)
from app.services.stock_service import StockService

router = APIRouter()

def get_stock_service():
    return StockService()

@router.get("/getWatchList", response_model=Union[WatchlistResponse, EmptyWatchlistResponse])
async def get_watchlist(
    account_id: str = Query(..., description="Account ID"),
    stock_service: StockService = Depends(get_stock_service)
):
    """
    Get watchlist for an account:
    - For new accounts: returns empty object {}
    - For existing accounts: returns stock data for watchlisted symbols
    """
    watchlist_data = stock_service.get_watchlist(account_id)
    
    if not watchlist_data:
        # Empty response for new accounts
        return EmptyWatchlistResponse(data={})
    
    # Convert raw data to StockPrice models
    stock_models = {}
    for symbol, data in watchlist_data.items():
        stock_models[symbol] = StockPrice(**data)
    
    return WatchlistResponse(data=stock_models)

@router.post("/watchlist/{account_id}/add")
async def add_to_watchlist(
    account_id: str = Path(..., description="Account ID"),
    symbol: str = Query(..., description="Stock symbol to add to watchlist"),
    stock_service: StockService = Depends(get_stock_service)
):
    """Add a stock to user's watchlist"""
    success = stock_service.add_to_watchlist(account_id, symbol)
    if success:
        return {
            "status": "success",
            "message": f"Added {symbol} to watchlist for account {account_id}"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Could not add {symbol} to watchlist. Symbol may not exist or already in watchlist."
        )

@router.get("/account/{account_id}", response_model=AccountResponse)
async def get_account(
    account_id: str = Path(..., description="Account ID"), 
    stock_service: StockService = Depends(get_stock_service)
):
    """
    Get account information:
    - For new accounts: returns empty data object
    - For existing accounts: returns stock price data
    """
    return stock_service.get_account_data(account_id)

@router.post("/account/{account_id}", response_model=NewAccountResponse)
async def create_account(
    account_id: str = Path(..., description="Account ID to create"),
    stock_service: StockService = Depends(get_stock_service)
):
    """Create a new account"""
    success = stock_service.register_account(account_id)
    if success:
        return NewAccountResponse(
            status="success",
            message=f"Account {account_id} created successfully",
            data={}
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Account {account_id} already exists"
        )

@router.get("/stocks", response_model=StockPriceResponse)
async def get_all_stocks(stock_service: StockService = Depends(get_stock_service)):
    """Get all available stock prices"""
    stocks = stock_service.get_all_stocks()
    return StockPriceResponse(stocks=stocks)

@router.get("/stocks/{symbol}", response_model=StockPrice)
async def get_stock(
    symbol: str = Path(..., description="Stock symbol"),
    stock_service: StockService = Depends(get_stock_service)
):
    """Get price data for a specific stock"""
    stock = stock_service.get_stock_price(symbol)
    if stock:
        return stock
    raise HTTPException(
        status_code=404,
        detail=f"Stock {symbol} not found"
    )