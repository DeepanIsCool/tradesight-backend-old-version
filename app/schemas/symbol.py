from pydantic import BaseModel

class CandleRequest(BaseModel):
    symbol: str