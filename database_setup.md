# 🧬 CasaLingua: Prisma + PostgreSQL Integration Guide

This guide sets up Prisma and PostgreSQL for CasaLingua v21 to enable session memory, audit logging, and corpus tracking — matching Bloom’s backend tech stack.

---

## ✅ Prerequisites
- macOS with Homebrew
- Python 3.10+
- Node.js 18+
- PostgreSQL (local or Docker)
- Prisma CLI

---

## 🔧 Step 1: Install PostgreSQL
### Option A: Homebrew
```bash
brew install postgresql@15
brew services start postgresql@15
```

### Option B: Docker
```bash
docker run --name casalingua-pg -p 5432:5432 -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=casalingua -d postgres:15
```

---

## 📦 Step 2: Initialize Prisma
```bash
npm install prisma --save-dev
npx prisma init
```
This creates:
```
prisma/
  └── schema.prisma
.env
```

---

## ✍️ Step 3: Create Prisma Schema
**Edit `prisma/schema.prisma`:**
```prisma
// schema.prisma

generator client {
  provider = "prisma-client-py"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model SessionMemory {
  id        String   @id @default(uuid())
  sessionId String
  turns     Json
  createdAt DateTime @default(now())
}

model AuditLog {
  id        String   @id @default(uuid())
  sessionId String
  input     String
  output    String
  audit     Json
  createdAt DateTime @default(now())
}
```

---

## 🔐 Step 4: Configure `.env`
```ini
DATABASE_URL="postgresql://postgres:secret@localhost:5432/casalingua"
```

---

## 🚀 Step 5: Apply Migrations
```bash
npx prisma migrate dev --name init
```

---

## 🧪 Step 6: Generate Client (Python)
```bash
pip install prisma
prisma generate
```

---

## 🔌 Step 7: Use Prisma in FastAPI
```python
from prisma import Prisma

db = Prisma()
await db.connect()

await db.sessionmemory.create({
    "sessionId": "abc123",
    "turns": [{"user": "Hello", "bot": "Hi!"}]
})
```

---

## 🧹 Step 8: Optional Cleanup
```bash
rm -rf casalingua_env/
find . -name '__pycache__' -delete
```

