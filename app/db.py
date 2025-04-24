import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

logger = logging.getLogger("db")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

DATABASE_URL = settings.database_url
if not DATABASE_URL:
    logger.error("DATABASE_URL not set; aborting.")
    raise ValueError("DATABASE_URL missing in .env")

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    logger.debug("Opening DB session")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("Closed DB session")