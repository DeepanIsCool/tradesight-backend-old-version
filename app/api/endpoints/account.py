from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.account import AccountResponse, NewAccountRequest, OrderRequest
from app.services.helper_service import generate_client_id, calculate_spent_cash, get_user_profit
from app.database.database import get_db
from app.core.security import get_current_user
from typing import Dict, Any

router = APIRouter()
@router.post("/accounts", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
def create_account(account_data: NewAccountRequest):
    client_id = generate_client_id()
    
    with get_db() as conn:
        conn.execute(
            "INSERT INTO accounts (client_id, username, initial_cash, remaining_cash) VALUES (?, ?, ?, ?)",
            (client_id, account_data.username, account_data.initial_cash, account_data.initial_cash)
        )
        conn.commit()
        
        return {
            "spentCash": 0,
            "remainingCash": account_data.initial_cash,
            "scrips": [],
            "orderBook": []
        }

@router.get("/accounts/{client_id}", response_model=AccountResponse)
def get_account(client_id: str):
    with get_db() as conn:
        account = conn.execute(
            "SELECT * FROM accounts WHERE client_id = ?", 
            (client_id,)
        ).fetchone()
        
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        scrips = []
        scrips_rows = conn.execute(
            "SELECT scrip, quantity, buy_price FROM scrips WHERE client_id = ?",
            (client_id,)
        ).fetchall()
        
        for row in scrips_rows:
            scrips.append({
                "scrip": row["scrip"],
                "quantity": row["quantity"],
                "buyPrice": row["buy_price"]
            })
        
        orders = []
        order_rows = conn.execute(
            "SELECT id, username, client_id, scrip, quantity, price, time, type FROM order_book WHERE client_id = ?",
            (client_id,)
        ).fetchall()
        
        for row in order_rows:
            orders.append({
                "id": row["id"],
                "username": row["username"],
                "clientID": row["client_id"],
                "scrip": row["scrip"],
                "quantity": row["quantity"],
                "price": row["price"],
                "time": row["time"],
                "type": row["type"]
            })
        
        spent_cash = calculate_spent_cash(conn, client_id)
        
        return {
            "spentCash": spent_cash,
            "remainingCash": account["remaining_cash"],
            "scrips": scrips,
            "orderBook": orders
        }

@router.post("/accounts/{client_id}/orders", response_model=AccountResponse)
def place_order(client_id: str, order: OrderRequest):
    with get_db() as conn:
        account = conn.execute(
            "SELECT * FROM accounts WHERE client_id = ?", 
            (client_id,)
        ).fetchone()
        
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        username = account["username"]
        remaining_cash = account["remaining_cash"]
        
        if order.type == "BUY":
            total_cost = order.quantity * order.price
            
            if total_cost > remaining_cash:
                raise HTTPException(
                    status_code=400, 
                    detail="Insufficient funds to complete purchase"
                )
            
            conn.execute(
                "UPDATE accounts SET remaining_cash = remaining_cash - ? WHERE client_id = ?",
                (total_cost, client_id)
            )
            
            order_id = conn.execute(
                """
                INSERT INTO order_book (client_id, username, scrip, quantity, price, type) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (client_id, username, order.scrip, order.quantity, order.price, order.type)
            ).lastrowid
            
            existing_scrip = conn.execute(
                "SELECT id, quantity FROM scrips WHERE client_id = ? AND scrip = ?",
                (client_id, order.scrip)
            ).fetchone()
            
            if existing_scrip:
                new_quantity = existing_scrip["quantity"] + order.quantity
                conn.execute(
                    "UPDATE scrips SET quantity = ? WHERE id = ?",
                    (new_quantity, existing_scrip["id"])
                )
            else:
                conn.execute(
                    "INSERT INTO scrips (client_id, scrip, quantity, buy_price) VALUES (?, ?, ?, ?)",
                    (client_id, order.scrip, order.quantity, order.price)
                )
        
        elif order.type == "SELL":
            scrip = conn.execute(
                "SELECT id, quantity FROM scrips WHERE client_id = ? AND scrip = ?",
                (client_id, order.scrip)
            ).fetchone()
            
            if not scrip or scrip["quantity"] < order.quantity:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient {order.scrip} quantity to sell"
                )
            
            new_quantity = scrip["quantity"] - order.quantity
            if new_quantity == 0:
                conn.execute(
                    "DELETE FROM scrips WHERE id = ?",
                    (scrip["id"],)
                )
            else:
                conn.execute(
                    "UPDATE scrips SET quantity = ? WHERE id = ?",
                    (new_quantity, scrip["id"])
                )
            
            total_sale = order.quantity * order.price
            order_id = conn.execute(
                """
                INSERT INTO order_book (client_id, username, scrip, quantity, price, type) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (client_id, username, order.scrip, order.quantity, order.price, order.type)
            ).lastrowid
            
            conn.execute(
                "UPDATE accounts SET remaining_cash = remaining_cash + ? WHERE client_id = ?",
                (total_sale, client_id)
            )
        
        conn.commit()
        
        account = get_account(client_id).__dict__
        return account
    

@router.post("/auth/getAccountDetails", response_model=AccountResponse)
async def get_account_details(current_user: Dict[str, Any] = Depends(get_current_user)):
    client_id = current_user["client_id"]
    
    with get_db() as conn:
        account = conn.execute(
            "SELECT * FROM accounts WHERE client_id = ?", 
            (client_id,)
        ).fetchone()
        
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        scrips = []
        scrips_rows = conn.execute(
            "SELECT scrip, quantity, buy_price FROM scrips WHERE client_id = ?",
            (client_id,)
        ).fetchall()
        
        for row in scrips_rows:
            scrips.append({
                "scrip": row["scrip"],
                "quantity": row["quantity"],
                "buyPrice": row["buy_price"]
            })
        
        orders = []
        order_rows = conn.execute(
            "SELECT id, username, client_id, scrip, quantity, price, time, type FROM order_book WHERE client_id = ?",
            (client_id,)
        ).fetchall()
        
        for row in order_rows:
            orders.append({
                "id": row["id"],
                "username": row["username"],
                "clientID": row["client_id"],
                "scrip": row["scrip"],
                "quantity": row["quantity"],
                "price": row["price"],
                "time": row["time"],
                "type": row["type"]
            })
        
        spent_cash = calculate_spent_cash(conn, client_id)
        
        return {
            "spentCash": spent_cash,
            "remainingCash": account["remaining_cash"],
            "scrips": scrips,
            "orderBook": orders
        }
    
@router.post("/transaction/getAccountProfit")
async def get_account_profit(username: str = Depends(get_current_user)):
    try:
        profit_data = get_user_profit(username)
        return profit_data
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")