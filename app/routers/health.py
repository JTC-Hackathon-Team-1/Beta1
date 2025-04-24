from fastapi import APIRouter
from sqlalchemy import text
from app.db import engine

router = APIRouter()

@router.get("/health")
async def health():
    # Simple DB check
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    return {"status": "ok"}
