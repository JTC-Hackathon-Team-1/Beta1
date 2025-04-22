"""
Unit Test - CasaLingua Text Pipeline
"""

from app.text_pipeline import run_pipeline


def assert_pipeline_result(result):
    assert "translated_text" in result
    assert "entities" in result
    assert "audit_report" in result
    assert result["audit_report"]["bias_check"] in ["PASS", "WARN", "FAIL"]


def test_pipeline_basic_en():
    result = run_pipeline(
        "My name is Jane Doe and I live at 45 Spring Street.",
        source_lang="en",
        target_lang="en",
        previous_turns=[]
    )
    assert_pipeline_result(result)


def test_pipeline_spanish():
    result = run_pipeline(
        "Mi nombre es Juan y necesito ayuda para encontrar una vivienda asequible.",
        source_lang="es",
        target_lang="en",
        previous_turns=[]
    )
    assert_pipeline_result(result)


def test_pipeline_chinese():
    result = run_pipeline(
        "我正在寻找一个可负担得起的住房选项。",
        source_lang="zh",
        target_lang="en",
        previous_turns=[]
    )
    assert_pipeline_result(result)


def test_pipeline_tagalog():
    result = run_pipeline(
        "Kailangan ko ng tulong sa pag-aapply para sa bahay.",
        source_lang="tl",
        target_lang="en",
        previous_turns=[]
    )
    assert_pipeline_result(result)


def test_pipeline_legalese():
    result = run_pipeline(
        "Pursuant to Section 8 of the Housing Act, the applicant shall submit income documentation within 30 days.",
        source_lang="en",
        target_lang="en",
        previous_turns=[]
    )
    assert_pipeline_result(result)
