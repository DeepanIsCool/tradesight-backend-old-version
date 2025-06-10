from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    cash_balance = Column(Float, default=250000.0)  # Default starting cash
    
    portfolio = relationship("Portfolio", back_populates="user")
    watchlist = relationship("Watchlist", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")


class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scrip = Column(String, index=True)
    quantity = Column(Integer)
    buy_price = Column(Float)  # Average buy price
    
    user = relationship("User", back_populates="portfolio")


class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scrip = Column(String, index=True)
    
    user = relationship("User", back_populates="watchlist")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scrip = Column(String, index=True)
    transaction_type = Column(String)  # "BUY" or "SELL"
    quantity = Column(Integer)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    
    user = relationship("User", back_populates="transactions")


class StockIndex(Base):
    __tablename__ = "stock_indices"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # e.g., "NIFTY_50"
    description = Column(String, nullable=True)
    
    stocks = relationship("IndexStock", back_populates="index")


class IndexStock(Base):
    __tablename__ = "index_stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    index_id = Column(Integer, ForeignKey("stock_indices.id"))
    symbol = Column(String, index=True)  # e.g., "RELIANCE.NS"
    company_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    added_date = Column(DateTime, default=datetime.now)
    
    index = relationship("StockIndex", back_populates="stocks")
