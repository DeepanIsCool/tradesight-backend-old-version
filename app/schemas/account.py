from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List

class Scrip(BaseModel):
    scrip: str
    quantity: int
    buyPrice: float
    
    class Config:
        from_attributes = True

class Order(BaseModel):
    id: int
    username: str
    clientID: str
    scrip: str
    quantity: int
    price: float
    time: datetime
    type: str
    
    class Config:
        from_attributes = True

class AccountResponse(BaseModel):
    spentCash: int
    remainingCash: int
    scrips: List[Scrip] = []
    orderBook: List[Order] = []
    
    class Config:
        from_attributes = True

class OrderRequest(BaseModel):
    scrip: str
    quantity: int
    price: float
    type: str = Field(..., pattern="^(BUY|SELL)$")
    
    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('quantity must be positive')
        return v
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('price must be positive')
        return v

class NewAccountRequest(BaseModel):
    username: str
    initial_cash: int = 250000