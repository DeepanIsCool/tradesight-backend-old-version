from fastapi import APIRouter, HTTPException, Depends
from app.schemas.symbol import CandleRequest
from app.services.mover_service import MoverService

router = APIRouter()

@router.post("/topmovers")
async def get_top_movers(service: MoverService = Depends()):
    return await service.fetch_top_movers()

@router.post("/getIndices")
async def get_indices(service: MoverService = Depends()):
    return await service.fetch_indices()

@router.post("/getCandles")
async def get_candles(request: CandleRequest, service: MoverService = Depends()):
    return await service.fetch_candles(request)

@router.post("/announcements")
async def get_news(service: MoverService = Depends()):
    return await service.fetch_announcements()

@router.post("/getMarketCap")
async def get_market_cap(service: MoverService = Depends()):
    return await service.fetch_market_cap()

@router.post("/getOrderBook")
async def get_order_book(request: CandleRequest, service: MoverService = Depends()):
    return await service.fetch_order_book(request)