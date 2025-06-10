import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mydatabase.db")
DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY: str = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY")  # Change this!
ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

