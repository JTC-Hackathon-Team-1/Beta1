# app/utils/readability.py

import textstat

def reading_grade_level(text: str) -> float:
    """
    Compute the Flesch–Kincaid Grade Level of the text.
    Falls back to 0.0 on error.
    """
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return 0.0