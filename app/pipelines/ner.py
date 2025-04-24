# app/pipelines/ner.py
import spacy
from asyncio import to_thread

_nlp = spacy.load("en_core_web_sm")

async def extract_entities(text: str) -> list[dict]:
    """
    Run spaCy NER and return a list of {"text":..., "label":...}.
    """
    doc = await to_thread(_nlp, text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]