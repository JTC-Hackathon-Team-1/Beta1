#!/usr/bin/env bash
set -e

echo "Applying Prisma migrations…"
python3 -m prisma migrate dev --name init

echo "Launching CasaLingua API…"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000