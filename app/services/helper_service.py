import random
import string
from fastapi import HTTPException
from app.schemas.account import AccountProfitResponse

def generate_client_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))

def calculate_spent_cash(conn, client_id):
    buys = conn.execute(
        "SELECT SUM(quantity * price) FROM order_book WHERE client_id = ? AND type = 'BUY'",
        (client_id,)
    ).fetchone()[0] or 0
    
    sells = conn.execute(
        "SELECT SUM(quantity * price) FROM order_book WHERE client_id = ? AND type = 'SELL'",
        (client_id,)
    ).fetchone()[0] or 0
    
    return int(buys - sells)

def get_user_profit(self, username: str) -> AccountProfitResponse:
        user = self.users.get(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user["is_new"]:
            return AccountProfitResponse(overallProfit=0, profitArray=[])
        else:
            return AccountProfitResponse(**self.profit_data.get(username, {"overallProfit": 0, "profitArray": []}))