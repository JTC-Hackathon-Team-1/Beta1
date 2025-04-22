# health_check.py
"""
CasaLingua System Health Check
Verifies model availability, database connection, and critical dependencies.
"""

import os
import subprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


def check_postgres():
    print("🔍 Checking PostgreSQL connection...")
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("✅ PostgreSQL connection successful.")
    except Exception as e:
        print(f"❌ PostgreSQL error: {e}")


def check_helsinki():
    print("🔍 Checking Helsinki translation model...")
    try:
        AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        print("✅ Helsinki model loaded from local cache.")
    except Exception as e:
        print(f"❌ Helsinki model error: {e}")


def check_bloom():
    print("🔍 Checking BLOOMZ model...")
    try:
        AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7")
        print("✅ BLOOMZ model loaded from local cache.")
    except Exception as e:
        print(f"❌ BLOOMZ model error: {e}")


def check_prisma():
    print("🔍 Checking Prisma client availability...")
    try:
        result = subprocess.run(["npx", "prisma", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Prisma CLI is installed.")
        else:
            print("❌ Prisma CLI not found.")
    except Exception as e:
        print(f"❌ Prisma check error: {e}")


if __name__ == "__main__":
    print("🩺 CasaLingua System Health Check")
    print("=" * 40)
    check_postgres()
    check_helsinki()
    check_bloom()
    check_prisma()
    print("=" * 40)
    print("✅ Health check complete.")
