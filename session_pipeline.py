# app/session_pipeline.py

from prisma import Prisma
import uuid
import json
from datetime import datetime

# Initialize Prisma once
prisma = Prisma()

async def get_or_create_session(session_id: str) -> dict:
    await prisma.connect()
    existing = await prisma.sessionmemory.find_unique(where={"sessionId": session_id})
    if existing:
        return existing
    else:
        return await prisma.sessionmemory.create({
            "sessionId": session_id,
            "turns": []
        })

async def save_turn(session_id: str, user_input: str, bot_response: str):
    session = await get_or_create_session(session_id)
    turns = session.turns or []
    turns.append({
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_input,
        "bot": bot_response
    })
    await prisma.sessionmemory.update(
        where={"sessionId": session_id},
        data={"turns": turns}
    )

async def get_previous_turns(session_id: str) -> list:
    session = await get_or_create_session(session_id)
    return session.turns or []

async def close_connection():
    await prisma.disconnect()
