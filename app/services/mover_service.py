from fastapi import HTTPException
from app.schemas.symbol import CandleRequest
import httpx

FOURSIGHT_BASE_URL = "https://foursight-backend.harshiyer.workers.dev/api/v1"

class MoverService:
    async def _post_request(self, endpoint: str, payload: dict = None) -> dict:
        url = f"{FOURSIGHT_BASE_URL}/{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"External API Error: {exc.response.text}")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request Error: {str(exc)}")

    async def fetch_top_movers(self) -> dict:
        payload = {"size": 10}
        return await self._post_request("topmovers", payload)

    async def fetch_indices(self) -> dict:
        return await self._post_request("getIndices")

    async def fetch_candles(self, request: CandleRequest) -> dict:
        payload = {"symbol": request.symbol}
        return await self._post_request("getCandles", payload)

    async def fetch_announcements(self) -> dict:
        return await self._post_request("announcements")
    
    async def fetch_market_cap(self) -> dict:
        return await self._post_request("getMarketCap")
    
    async def fetch_order_book(self, request: CandleRequest) -> dict:
        payload = {"symbol": request.symbol}
        return await self._post_request("getOrderBook", payload)
