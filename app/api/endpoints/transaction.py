from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import User
from app.core.security import get_current_user
from app.schemas.transaction import BuySellRequest, WatchlistRequest, BuyScripResponse, SellScripResponse, WatchlistResponse

from app.services.transaction_service import (
    buy_scrip_logic,
    sell_scrip_logic,
    add_watchlist_logic,
    remove_watchlist_logic,
)

router = APIRouter()

@router.post("/buyScrip", response_model=BuyScripResponse)
async def buy_scrip(
    request: BuySellRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return buy_scrip_logic(request, current_user, db)

@router.post("/sellScrip", response_model=SellScripResponse)
async def sell_scrip(
    request: BuySellRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return sell_scrip_logic(request, current_user, db)

@router.post("/addWatchlist", response_model=WatchlistResponse)
async def add_watchlist(
    request: WatchlistRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return add_watchlist_logic(request, current_user, db)

@router.post("/removeWatchList", response_model=WatchlistResponse)
async def remove_watchlist(
    request: WatchlistRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return remove_watchlist_logic(request, current_user, db)
