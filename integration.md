# 🤖 CasaLingua Assistant Entry Point Integration Guide

This guide helps GUI developers integrate the animated Clippy-style AI assistant into the CasaLingua application frontend.

---

## 🎯 Purpose
The assistant acts as a **friendly entry point** to:
- Receive user text or voice input
- Display step-by-step progress
- Fetch translations, simplifications, and audit results
- Provide accessibility cues (speech playback, highlighting)

---

## 🧩 Component Overview

### 🟡 `main.py` (Back-End Entry Point)
- Accepts user text (or via `/api/input`)
- Routes through:
  - `input_funnel`
  - `text_pipeline`
  - `translation_llm`
  - `governance_llm`
- Returns JSON:
```json
{
  "translated_text": "...",
  "bias_check": "PASS",
  "semantic_fidelity": "HIGH",
  "confidence_score": 0.96,
  "entity_alignment": true
}
```

### 🟢 Frontend Entry Module
- 📍 Location: `frontend/src/components/Assistant.vue` or `.jsx`
- Starts on click or voice input
- Shows character animation (idle → thinking → output)
- Sends `POST /api/input` with payload:
```json
{
  "text": "Le locataire..."
}
```

---

## 🔌 Integration Endpoints

### `POST /api/input`
Send user message to CasaLingua.
```http
POST /api/input
Content-Type: application/json

{
  "text": "User input or transcribed audio"
}
```

### `GET /admin/settings`
UI may optionally show model/corpus info.

### `GET /admin/corpus`
Optional for corpus list dropdown.

---

## 🔄 States to Animate
| State         | Description                     |
|---------------|----------------------------------|
| `idle`        | Waiting for user input          |
| `listening`   | Recording audio / input active  |
| `thinking`    | Request sent, waiting on reply  |
| `responding`  | Assistant reveals translation   |
| `auditing`    | Shows score results + summary   |

---

## 🧠 Output Visualization Suggestions
- 🎯 Display each audit result as a colored chip:
  - ✅ PASS → green
  - ⚠️ WARN → yellow
  - ❌ FAIL → red
- 📊 Show confidence score as progress ring
- 📚 Glossary items can be underlined or tooltip-linked
- 🗂 Allow download of `audit.json` session

---

## 📣 Optional Audio Feedback
- Use TTS to speak translated text
- Highlight glossary replacements

---

## ✅ Summary
The CasaLingua Assistant is your user’s first guide to understanding complex language. The GUI developer should:
- Animate clear user flow
- Send user input to `/api/input`
- Display JSON results interactively
- Optionally use `/admin/settings` for metadata


