from fastapi import APIRouter
from .endpoints import users, market, wishlist,account, transaction

router = APIRouter(prefix="/api/v1")
router.include_router(users.router)
router.include_router(market.router)
router.include_router(wishlist.router)
router.include_router(account.router)
router.include_router(transaction.router)