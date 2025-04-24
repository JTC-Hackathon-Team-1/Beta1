from prisma import Prisma
import uuid
import json
from datetime import datetime
import logging

logger = logging.getLogger("session_pipeline")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

prisma = Prisma()

async def get_or_create_session(session_id: str) -> dict:
    logger.debug(f"Opening Prisma connection for session {session_id}")
    await prisma.connect()
    existing = await prisma.sessionmemory.find_unique(where={"sessionId": session_id})
    if existing:
        logger.debug("Session found.")
        return existing
    logger.debug("Creating new session record.")
    return await prisma.sessionmemory.create(data={"sessionId": session_id, "turns": []})

async def save_turn(session_id: str, user_input: str, bot_response: str):
    logger.debug(f"Saving turn to session {session_id}")
    session = await prisma.sessionmemory.find_unique(where={"sessionId": session_id})
    turns = session.turns or []
    turns.append({
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_input,
        "bot": bot_response
    })
    await prisma.sessionmemory.update(where={"sessionId": session_id}, data={"turns": turns})

async def get_previous_turns(session_id: str) -> list:
    sess = await get_or_create_session(session_id)
    return sess.turns or []

async def close_connection():
    logger.debug("Disconnecting Prisma client")
    await prisma.disconnect()