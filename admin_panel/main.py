
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
"""
CasaLingua Admin Panel - FastAPI Interface
Provides pipeline monitoring, session controls, and data management.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.session_pipeline import get_previous_turns
from prisma import Prisma
from typing import List, Dict
import os
import subprocess

app = FastAPI(
    title="CasaLingua Admin Panel",
    description="Monitor sessions and pipeline activity",
    version="1.0.0"
)

# Setup template engine
template_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=template_dir)

@app.get("/")
def root():
    return {"message": "CasaLingua Admin Panel is running."}

@app.get("/session/{session_id}")
async def get_session_context(session_id: str) -> Dict[str, List[Dict[str, str]]]:
    """View stored session turns by session_id"""
    turns = await get_previous_turns(session_id)
    return {"session_id": session_id, "turns": turns}

@app.get("/sessions")
async def list_all_sessions():
    """List all active session IDs in the database"""
    db = Prisma()
    await db.connect()
    sessions = await db.sessionmemory.find_many()
    session_ids = sorted(set(s.sessionId for s in sessions))
    await db.disconnect()
    return {"sessions": session_ids}

@app.get("/sessions/view", response_class=HTMLResponse)
async def view_sessions(request: Request):
    """Display session list in an HTML dashboard"""
    db = Prisma()
    await db.connect()
    sessions = await db.sessionmemory.find_many()
    session_ids = sorted(set(s.sessionId for s in sessions))
    await db.disconnect()
    return templates.TemplateResponse("session_list.html", {"request": request, "sessions": session_ids})

@app.get("/install", response_class=HTMLResponse)
async def install_form(request: Request):
    return templates.TemplateResponse("install_wizard.html", {"request": request})

@app.post("/install")
async def run_installer(request: Request, install_prisma: bool = Form(...), install_requirements: bool = Form(...)):
    messages = []
    if install_prisma:
        try:
            subprocess.run(["npx", "prisma", "migrate", "dev", "--name", "init"], check=True)
            messages.append("✅ Prisma migration applied.")
        except Exception as e:
            messages.append(f"❌ Prisma error: {e}")

    if install_requirements:
        try:
            subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
            messages.append("✅ Requirements installed.")
        except Exception as e:
            messages.append(f"❌ Pip install error: {e}")

    return templates.TemplateResponse("install_wizard.html", {"request": request, "messages": messages})

