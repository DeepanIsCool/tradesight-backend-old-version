from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Union
from jose import jwt
from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES  # Import from config
from sqlalchemy.orm import Session
from app.database.database import get_db
from app.database import models

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  # Not used
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def create_access_token(subject: Union[str, any], client_id: str, email: str, expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    claims = {"exp": expire, "sub": str(subject), "clientID": client_id, "email": email} #Added clientID and email
    encoded_jwt = jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
 


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> models.User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
