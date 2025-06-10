from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
from datetime import datetime


class StockPrice(BaseModel):
    """Model for stock price data"""
    type: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    ltp: float
    volume: int
    tsInMillis: int
    lowPriceRange: Optional[float] = None
    highPriceRange: Optional[float] = None
    totalBuyQty: int
    totalSellQty: int
    dayChange: float
    dayChangePerc: float
    openInterest: Optional[float] = None
    lastTradeQty: int
    lastTradeTime: int
    prevOpenInterest: Optional[float] = None
    oiDayChange: float
    oiDayChangePerc: float
    lowTradeRange: Optional[float] = None
    highTradeRange: Optional[float] = None

    def formatted_timestamp(self) -> str:
        """Convert timestamp to readable format"""
        return datetime.fromtimestamp(self.tsInMillis / 1000).strftime('%Y-%m-%d %H:%M:%S')


class StockPriceResponse(BaseModel):
    """Response model containing multiple stock prices"""
    stocks: Dict[str, StockPrice]


class WatchlistResponse(BaseModel):
    """Response for watchlist request"""
    data: Dict[str, StockPrice] = {}


class EmptyWatchlistResponse(BaseModel):
    """Response for empty watchlist (new account)"""
    data: Dict[str, Any] = {}


class AccountResponse(BaseModel):
    """Base response model for account requests"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class NewAccountResponse(AccountResponse):
    """Response for new account"""
    pass


class ExistingAccountResponse(AccountResponse):
    """Response for existing account with stock data"""
    data: Dict[str, StockPrice]