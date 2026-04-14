import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the asynchronous engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create the asynchronous session factory
SessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()

# Asynchronous dependency for FastAPI
async def get_db():
    async with SessionLocal() as db:
        yield db