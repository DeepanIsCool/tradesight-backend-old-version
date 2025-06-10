from typing import Dict, Optional, List, Any
from app.schemas.stock_models import (
    StockPrice, 
    AccountResponse, 
    NewAccountResponse, 
    ExistingAccountResponse,
    WatchlistResponse,
    EmptyWatchlistResponse
)

class StockService:
    """Service to handle stock data operations"""
    
    def __init__(self):
        self._stock_data = {}
        self._user_accounts = {}
        
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample stock data"""
        self._stock_data = {
            "BAJAJHLDNG": {
                "type": "LIVE_PRICE",
                "symbol": "BAJAJHLDNG",
                "open": 12360,
                "high": 12535,
                "low": 11841,
                "close": 12316,
                "ltp": 11973,
                "volume": 68305,
                "tsInMillis": 1746007910,
                "lowPriceRange": 9579,
                "highPriceRange": 14367,
                "totalBuyQty": 0,
                "totalSellQty": 26,
                "dayChange": -343,
                "dayChangePerc": -2.7849951282884056,
                "openInterest": None,
                "lastTradeQty": 1,
                "lastTradeTime": 1746007910,
                "prevOpenInterest": None,
                "oiDayChange": 0,
                "oiDayChangePerc": 0,
                "lowTradeRange": None,
                "highTradeRange": None
            },
            "BHARTIARTL": {
                "type": "LIVE_PRICE",
                "symbol": "BHARTIARTL",
                "open": 1837,
                "high": 1876.9,
                "low": 1827,
                "close": 1823.8,
                "ltp": 1864.5,
                "volume": 9139821,
                "tsInMillis": 1746008921,
                "lowPriceRange": 1678.1,
                "highPriceRange": 2050.9,
                "totalBuyQty": 0,
                "totalSellQty": 1443,
                "dayChange": 40.700000000000045,
                "dayChangePerc": 2.231604342581426,
                "openInterest": None,
                "lastTradeQty": 32,
                "lastTradeTime": 1746008921,
                "prevOpenInterest": None,
                "oiDayChange": 0,
                "oiDayChangePerc": 0,
                "lowTradeRange": None,
                "highTradeRange": None
            }
        }
        
        # Create a sample existing account with watchlist
        self._user_accounts["existing_user"] = {
            "created_at": 1746008000,
            "watchlist": ["BAJAJHLDNG", "BHARTIARTL"]
        }
    
    def get_account_data(self, account_id: str) -> AccountResponse:
        """Get account data - returns different response based on whether account exists"""
        if account_id not in self._user_accounts:
            # New account scenario
            return NewAccountResponse(
                status="success",
                message="New account created",
                data={}
            )
        else:
            # Existing account scenario with stock data
            return ExistingAccountResponse(
                status="success",
                message="Account data retrieved",
                data=self._stock_data
            )
    
    def get_watchlist(self, account_id: str) -> Dict:
        """Get watchlist for an account"""
        # For new accounts (non-existent), return empty response
        if account_id not in self._user_accounts:
            return {}
        
        # For existing accounts, return stock data from their watchlist
        watchlist_symbols = self._user_accounts[account_id]["watchlist"]
        result = {}
        
        for symbol in watchlist_symbols:
            if symbol in self._stock_data:
                result[symbol] = self._stock_data[symbol]
                
        return result
    
    def get_stock_price(self, symbol: str) -> Optional[StockPrice]:
        """Get price data for a specific stock"""
        if symbol in self._stock_data:
            return StockPrice(**self._stock_data[symbol])
        return None
    
    def get_all_stocks(self) -> Dict[str, StockPrice]:
        return {symbol: StockPrice(**data) for symbol, data in self._stock_data.items()}
    
    def register_account(self, account_id: str) -> bool:
        if account_id not in self._user_accounts:
            self._user_accounts[account_id] = {
                "created_at": 1746008921,
                "watchlist": []
            }
            return True
        return False
        
    def add_to_watchlist(self, account_id: str, symbol: str) -> bool:
        """Add a stock to user's watchlist"""
        # Create account if it doesn't exist
        if account_id not in self._user_accounts:
            self.register_account(account_id)
            
        # Add to watchlist if stock exists and not already in watchlist
        if (symbol in self._stock_data and 
            symbol not in self._user_accounts[account_id]["watchlist"]):
            self._user_accounts[account_id]["watchlist"].append(symbol)
            return True
        return False