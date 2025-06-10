from sqlalchemy.orm import Session
from datetime import datetime
from app.database.models import User, Portfolio, Watchlist, Transaction
from app.schemas.transaction import ScripItem
from fastapi import HTTPException
from app.core.config import settings

def buy_scrip_logic(request, current_user: User, db: Session):
    quantity = int(request.quantity)
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    scrip = request.scrip
    if scrip not in settings.MOCK_PRICES:
        raise HTTPException(status_code=404, detail=f"Scrip {scrip} not found")

    price = settings.MOCK_PRICES[scrip]
    total_cost = price * quantity

    if current_user.cash_balance < total_cost:
        raise HTTPException(status_code=400, detail="Insufficient funds")

    portfolio_item = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.scrip == scrip
    ).first()

    if portfolio_item:
        new_total_quantity = portfolio_item.quantity + quantity
        new_total_cost = (portfolio_item.quantity * portfolio_item.buy_price) + (quantity * price)
        new_avg_price = new_total_cost / new_total_quantity

        portfolio_item.quantity = new_total_quantity
        portfolio_item.buy_price = new_avg_price
    else:
        portfolio_item = Portfolio(
            user_id=current_user.id,
            scrip=scrip,
            quantity=quantity,
            buy_price=price
        )
        db.add(portfolio_item)

    current_user.cash_balance -= total_cost

    transaction = Transaction(
        user_id=current_user.id,
        scrip=scrip,
        transaction_type="BUY",
        quantity=quantity,
        price=price,
        timestamp=datetime.utcnow()
    )
    db.add(transaction)
    db.commit()

    updated_portfolio = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.scrip == scrip
    ).first()

    from app.schemas.transaction import BuyScripResponse
    return BuyScripResponse(
        message="Scrip Bought Successfully!",
        scrip=scrip,
        scripArray=[ScripItem(
            scrip=updated_portfolio.scrip,
            quantity=updated_portfolio.quantity,
            buyPrice=updated_portfolio.buy_price
        )],
        quantity=quantity,
        price=price,
        totalValue=total_cost,
        time=datetime.utcnow(),
        remainingCash=current_user.cash_balance,
        spentCash=total_cost,
        averageBuyPrice=updated_portfolio.buy_price
    )


def sell_scrip_logic(request, current_user: User, db: Session):
    quantity = int(request.quantity)
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    scrip = request.scrip
    if scrip not in settings.MOCK_PRICES:
        raise HTTPException(status_code=404, detail=f"Scrip {scrip} not found")

    portfolio_item = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.scrip == scrip
    ).first()

    if not portfolio_item:
        raise HTTPException(status_code=404, detail=f"You don't own any {scrip}")

    if portfolio_item.quantity < quantity:
        raise HTTPException(status_code=400, detail=f"You only have {portfolio_item.quantity} of {scrip}")

    price = settings.MOCK_PRICES[scrip]
    sell_value = price * quantity

    portfolio_item.quantity -= quantity

    if portfolio_item.quantity == 0:
        db.delete(portfolio_item)

    current_user.cash_balance += sell_value

    transaction = Transaction(
        user_id=current_user.id,
        scrip=scrip,
        transaction_type="SELL",
        quantity=quantity,
        price=price,
        timestamp=datetime.utcnow()
    )
    db.add(transaction)
    db.commit()

    from app.schemas.transaction import SellScripResponse
    remaining_scrips = []
    if portfolio_item.quantity > 0:
        remaining_scrips.append(ScripItem(
            scrip=portfolio_item.scrip,
            quantity=portfolio_item.quantity,
            buyPrice=portfolio_item.buy_price
        ))

    return SellScripResponse(
        message="Scrip sold successfully",
        scrips=remaining_scrips
    )


def add_watchlist_logic(request, current_user: User, db: Session):
    scrip = request.scrip
    if scrip not in settings.MOCK_PRICES:
        raise HTTPException(status_code=404, detail=f"Scrip {scrip} not found")

    existing = db.query(Watchlist).filter(
        Watchlist.user_id == current_user.id,
        Watchlist.scrip == scrip
    ).first()

    from app.schemas.transaction import WatchlistResponse
    if existing:
        return WatchlistResponse(message="Already added")

    watchlist_item = Watchlist(user_id=current_user.id, scrip=scrip)
    db.add(watchlist_item)
    db.commit()

    return WatchlistResponse(message="Added to the watchlist")


def remove_watchlist_logic(request, current_user: User, db: Session):
    scrip = request.scrip
    watchlist_item = db.query(Watchlist).filter(
        Watchlist.user_id == current_user.id,
        Watchlist.scrip == scrip
    ).first()

    if not watchlist_item:
        raise HTTPException(status_code=404, detail=f"{scrip} not in watchlist")

    db.delete(watchlist_item)
    db.commit()

    from app.schemas.transaction import WatchlistResponse
    return WatchlistResponse(message=f"Removed {scrip} from watchlist")
