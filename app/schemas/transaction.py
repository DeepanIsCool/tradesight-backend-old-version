from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ScripItem(BaseModel):
    scrip: str
    quantity: int
    buyPrice: float


class BuySellRequest(BaseModel):
    scrip: str
    quantity: str = Field(..., description="Quantity as a string, e.g. '01', '03'")


class WatchlistRequest(BaseModel):
    scrip: str


class BuyScripResponse(BaseModel):
    message: str
    scrip: str
    scripArray: List[ScripItem]
    quantity: int
    price: float
    totalValue: float
    time: datetime
    remainingCash: float
    spentCash: float
    averageBuyPrice: float


class SellScripResponse(BaseModel):
    message: str
    scrips: List[ScripItem]


class WatchlistResponse(BaseModel):
    message: str