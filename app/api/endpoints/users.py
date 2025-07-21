from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models
from app.schemas.user import User, UserCreate, UserUpdate, Token
from typing import List
from app.core.security import  create_access_token, get_current_user #Removed unused functions
from datetime import timedelta
from fastapi.security import OAuth2PasswordRequestForm
import uuid # Import uuid

router = APIRouter()

@router.post("/login", response_model=Token) # Changed to /login
async def login(form_data: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    print(user)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if  user.hashed_password != form_data.password: # Use hashed_password field
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=int(1000))
    client_id = str(uuid.uuid4())  # Generate a unique client ID
    access_token = create_access_token(
        subject=user.username, client_id=client_id, email=user.email, expires_delta=access_token_expires
    )
    return {"token": access_token, "message": "Login Successful!", "clientID": client_id} # Return required data



@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user


@router.get("/users/", response_model=List[User])
async def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users



@router.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user



@router.post("/register", response_model=User, status_code=201) # Changed to /register
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # hashed_password = get_password_hash(user.password)  # No hashing
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if email already exists
    existing_email = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = models.User(username=user.username, hashed_password=user.password, email=user.email) # Use hashed_password field
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user



@router.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.password:
        # hashed_password = get_password_hash(user.password) # No hashing
        setattr(db_user, "hashed_password", user.password) # Use hashed_password field

    if user.username:
        setattr(db_user, "username", user.username)

    db.commit()
    db.refresh(db_user)
    return db_user



@router.delete("/users/{user_id}", response_model=User)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return db_user