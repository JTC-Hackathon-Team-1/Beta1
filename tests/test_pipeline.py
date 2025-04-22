"""
Unit Test - CasaLingua Text Pipeline
"""

from app.text_pipeline import run_pipeline

def test_pipeline_basic():
    result = run_pipeline("My name is Jane Doe and I live at 45 Spring Street.", "en")
    assert "translated_text" in result
    assert "entities" in result
    assert result["audit_report"]["bias_check"] == "PASS"
