import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

_MODEL_NAME = "facebook/m2m100_418M"

# load once
_tokenizer = M2M100Tokenizer.from_pretrained(_MODEL_NAME)
_model = M2M100ForConditionalGeneration.from_pretrained(_MODEL_NAME)


async def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate between any language pair using M2M100.
    
    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en', 'fr', 'de')
        target_lang: Target language code (e.g., 'en', 'fr', 'de')
    
    Returns:
        Translated text
    """
    # Set the source language
    _tokenizer.src_lang = source_lang
    
    # Encode the input text
    encoded = _tokenizer(text, return_tensors="pt", truncation=True)
    
    # Generate translation with the target language
    with torch.no_grad():
        generated = _model.generate(
            **encoded,
            forced_bos_token_id=_tokenizer.get_lang_id(target_lang),
            max_length=512,
        )
    
    # Decode and return the translated text
    return _tokenizer.batch_decode(generated, skip_special_tokens=True)[0]