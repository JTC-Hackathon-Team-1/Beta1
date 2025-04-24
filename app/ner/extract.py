# app/ner/extract.py
import spacy
from typing import List, Dict

# load the spaCy model once at import time
_nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Run NER on the given text and return a list of entities.
    """
    doc = _nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]