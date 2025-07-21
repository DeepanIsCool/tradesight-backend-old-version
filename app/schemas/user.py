from pydantic import BaseModel, constr


class User(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=8, max_length=128)
    email: constr(min_length=5, max_length=128) # added email


class UserUpdate(BaseModel):
    username: constr(min_length=3, max_length=50) | None = None
    password: constr(min_length=8, max_length=128) | None = None



class Token(BaseModel):
    token: str # changed access_token to token
    message: str
    clientID: str
    #token_type: str # Removed token_type
