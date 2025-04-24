#!/usr/bin/env bash
set -e

echo "Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt
pip install pydantic-settings

echo "Installing SentencePiece…"
pip install sentencepiece

echo "Generating Prisma client…"
python3 -m prisma generate

echo "All dependencies and client code are up to date."