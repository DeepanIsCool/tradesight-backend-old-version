from typing import List
from pydantic import BaseModel

class ProfitItem(BaseModel):
    scrip: str
    quantity: int
    buyPrice: float
    ltp: float
    profitPerShare: float
    profit: float
    dayChange: float
    dayChangePerc: float

class AccountProfitResponse(BaseModel):
    overallProfit: float
    profitArray: List[ProfitItem] = []