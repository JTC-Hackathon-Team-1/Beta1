"""
CasaLingua Admin Panel - FastAPI Interface
Provides pipeline monitoring, session controls, and data management.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "CasaLingua Admin Panel is running."}
