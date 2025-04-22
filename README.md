# CasaLingua v0.21.0

## 🧠 Overview

CasaLingua is a multilingual AI voice and text assistant for simplifying housing applications. It automatically processes user inputs—whether audio, image, or text—and outputs readable, simplified forms with ethical compliance and audit capability.

---

## 📦 Modules Overview

### 1. `app/input_funnel.py`
- Detects file types: `.txt`, `.pdf`, `.jpg/.png`, `.wav/.mp3`
- Routes input to OCR or ASR processors

### 2. `app/text_pipeline.py`
- Orchestrates NER, translation, and governance checks

### 3. `app/translation_llm.py`
- Translates text to English or another target language
- Simulates simplification for legal or complex phrases

### 4. `app/ner_module.py`
- Extracts entities like name, address, DOB for form auto-fill

### 5. `app/governance_llm.py`
- Applies ethical audits (bias, fidelity, PII compliance)
- Returns a confidence score and alignment check

### 6. `app/utils/ocr.py`
- Extracts text from image or PDF using placeholder OCR

### 7. `app/utils/audio_tools.py`
- Transcribes audio to text using placeholder ASR

---

## 🧪 Tests

- `tests/test_pipeline.py`: Validates end-to-end output from raw text input

---

## 🚀 Running the App

```bash
python main.py
```

## 🎛️ Admin Panel (Optional)

```bash
uvicorn admin_panel.main:app --reload
```

## 🔍 Future Features

- Whisper integration
- PDF input support with OCR confidence
- Multilingual conversation agent with memory

---

## 🧠 Logic Ladder Diagram

```mermaid
graph TD
    A[User Input (Audio/Text/Image)] --> B[Input Funnel]
    B --> C[Text Pipeline]
    C --> D[NER Extraction]
    C --> E[Translation Module]
    C --> F[Governance & Audit]
    D & E & F --> G[Final Output & Audit Report]
```
