# app/text_pipeline.py
from app.translation_llm import translate
from app.ner_module import extract_entities
from app.governance_llm import GovernanceLLM


def build_prompt(previous_turns: list, current_input: str) -> str:
    """Constructs a multi-turn prompt string"""
    history = ""
    for turn in previous_turns:
        history += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    history += f"User: {current_input}"
    return history


def run_pipeline(text: str, source_lang="auto", target_lang="en", previous_turns=None):
    previous_turns = previous_turns or []

    # Step 1: Construct full prompt from history
    full_prompt = build_prompt(previous_turns, text)

    # Step 2: Translate
    result = translate(full_prompt)
    translated_text = result["translated_text"]

    # Step 3: Named Entity Recognition (NER)
    entities = extract_entities(translated_text)

    # Step 4: Governance Audit
    governor = GovernanceLLM()
    audit = governor.audit_translation(text, translated_text, entities)

    return {
        "translated_text": translated_text,
        "audit_report": audit,
        "entities": entities
    }
