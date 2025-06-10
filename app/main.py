from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.database.database import engine
from app.database import models
from sqlalchemy import text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

def upgrade_database():
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN email STRING"))
    except Exception as e:
        print(f"Migration note: {e}")

# Run this before starting your application
upgrade_database()

models.Base.metadata.create_all(bind=engine)
