# tests/pipelines/test_orchestrator.py
import pytest

from app.pipelines.orchestrator import run_pipeline


@pytest.mark.asyncio
async def test_run_pipeline_basic():
    # A simple English sentence
    text = "Barack Obama was born in Hawaii."
    out = await run_pipeline(text, src_lang="en", tgt_lang="fr")

    # Check structure
    assert isinstance(out, dict)
    assert out["original_text"] == text
    assert isinstance(out["original_entities"], list)
    assert isinstance(out["translated_text"], str)
    assert out["translated_text"] != text  # should have changed
    assert isinstance(out["translated_entities"], list)

    # Spot‐check that NER found at least one entity
    assert any(ent["label"] for ent in out["original_entities"])
    assert any(ent["label"] for ent in out["translated_entities"])