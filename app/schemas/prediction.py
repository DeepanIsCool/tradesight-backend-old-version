from pydantic import BaseModel
from datetime import datetime
from typing import Literal # For specific string values

class PredictionSignal(BaseModel):
    """Pydantic model for a single prediction output row."""
    Timestamp: datetime # Keep as datetime for proper JSON encoding
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float # Use float for volume in case it's large
    Signal: Literal['Buy', 'Sell', 'Hold'] # Enforce specific values

    class Config:
        from_attributes = True # Enable compatibility with ORM models (like dataframes with .to_dict('records'))