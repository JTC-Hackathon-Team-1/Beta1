# translation/translate_text.py

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
import time
import hashlib
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationLLM:
    """
    CasaLingua - Translation LLM
    Handles multilingual translation and simplification.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 config_path: Optional[str] = None,
                 default_model: str = "general",
                 default_simplification_level: str = "medium",
                 enable_caching: bool = True,
                 cache_size: int = 100):
        """
        Initialize the Translation LLM.
        
        Args:
            api_key: API key for external translation service (if used)
            config_path: Path to configuration file
            default_model: Default translation model ('general', 'legal', 'housing')
            default_simplification_level: Default text simplification level ('none', 'light', 'medium', 'heavy')
            enable_caching: Whether to cache translation results
            cache_size: Maximum number of translations to cache
        """
        self.api_key = api_key or os.environ.get("TRANSLATION_API_KEY")
        self.default_model = default_model
        self.default_simplification_level = default_simplification_level
        self.enable_caching = enable_caching
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded translation configuration from {config_path}")
                
                # Override defaults with config values if present
                self.api_key = self.config.get("api_key", self.api_key)
                self.default_model = self.config.get("default_model", self.default_model)
                self.default_simplification_level = self.config.get("default_simplification_level", 
                                                                  self.default_simplification_level)
                self.enable_caching = self.config.get("enable_caching", self.enable_caching)
        
        # Initialize translation cache if enabled
        if self.enable_caching:
            # Setup LRU cache for translations
            self._translation_cache = lru_cache(maxsize=cache_size)(self._translate_uncached)
            logger.info(f"Initialized translation cache with size {cache_size}")
        
        # Define supported languages
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "zh": "Chinese",
            "fr": "French",
            "ar": "Arabic",
            "ru": "Russian",
            "hi": "Hindi",
            "bn": "Bengali",
            "pt": "Portuguese",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "vi": "Vietnamese",
            "tl": "Tagalog",
            "ur": "Urdu",
            "fa": "Farsi",
            "tr": "Turkish",
            "it": "Italian",
            "th": "Thai",
            "pl": "Polish"
        }
        
        # Define translation models
        self.models = {
            "general": {
                "description": "General purpose translation",
                "prompt_template": "Translate the following text to {target_lang}: {input_text}"
            },
            "legal": {
                "description": "Legal document translation with terminology precision",
                "prompt_template": "Translate the following legal text to {target_lang}, maintaining legal terminology precision: {input_text}"
            },
            "housing": {
                "description": "Housing-specific translation with domain terminology",
                "prompt_template": "Translate the following housing-related text to {target_lang}, using appropriate housing terminology: {input_text}"
            },
            "forms": {
                "description": "Form translation preserving field labels and content",
                "prompt_template": "Translate the following form content to {target_lang}, preserving all form fields and labels: {input_text}"
            }
        }
        
        # Define simplification levels
        self.simplification_levels = {
            "none": {
                "description": "No simplification, just translation",
                "prompt_suffix": ""
            },
            "light": {
                "description": "Slightly simpler language while preserving most technical terms",
                "prompt_suffix": " Use slightly simpler language while preserving most technical terms."
            },
            "medium": {
                "description": "Moderately simplified language with plain language substitutions for technical terms",
                "prompt_suffix": " Use moderately simplified language with plain language substitutions for technical terms. Aim for approximately 8th-grade reading level."
            },
            "heavy": {
                "description": "Heavily simplified language with definitions for technical terms",
                "prompt_suffix": " Use heavily simplified language (elementary school level) and provide brief definitions in parentheses for any technical terms that must be kept."
            }
        }
        
        logger.info(f"Translation LLM initialized with {len(self.supported_languages)} supported languages")
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """
        Detect the language of input text.
        
        In a real implementation, this would call a language detection API/model.
        This is a simplified version for demonstration purposes.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping language codes to confidence scores
        """
        # Simple language detection based on common words
        # This is a very basic implementation for demonstration
        common_words = {
            "en": ["the", "and", "is", "in", "to", "you", "that", "was"],
            "es": ["el", "la", "y", "es", "en", "que", "por", "los"],
            "fr": ["le", "la", "et", "est", "en", "que", "pour", "dans"],
            "de": ["der", "die", "und", "ist", "in", "zu", "den", "das"],
            "pt": ["o", "a", "e", "é", "em", "que", "para", "de"],
            "it": ["il", "la", "e", "è", "in", "che", "per", "di"]
        }
        
        # Count word occurrences
        text_lower = text.lower()
        words = ''.join(c if c.isalnum() else ' ' for c in text_lower).split()
        
        scores = {}
        total_words = len(words)
        
        if total_words == 0:
            # Default to English if no words
            return {"en": 1.0}
        
        # Count occurrences of common words in each language
        for lang, lang_words in common_words.items():
            count = sum(1 for word in words if word in lang_words)
            scores[lang] = count / total_words
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {lang: score / total_score for lang, score in scores.items()}
        else:
            # Default to English if no matches
            scores = {"en": 1.0}
            
        # Add small scores for undetected languages
        for lang in self.supported_languages:
            if lang not in scores:
                scores[lang] = 0.01
                
        # Ensure the scores sum to 1.0
        total_score = sum(scores.values())
        scores = {lang: score / total_score for lang, score in scores.items()}
        
        return scores
    
    def _build_translation_prompt(self, 
                                input_text: str, 
                                target_lang: str,
                                model: str = None,
                                simplification_level: str = None) -> str:
        """
        Build the translation prompt for the LLM.
        
        Args:
            input_text: Text to translate
            target_lang: Target language code or name
            model: Translation model to use
            simplification_level: Text simplification level
            
        Returns:
            Formatted translation prompt
        """
        # Use defaults if not specified
        model = model or self.default_model
        simplification_level = simplification_level or self.default_simplification_level
        
        # Get model template
        if model not in self.models:
            logger.warning(f"Unknown model '{model}', falling back to general model")
            model = "general"
            
        model_template = self.models[model]["prompt_template"]
        
        # Get simplification suffix
        if simplification_level not in self.simplification_levels:
            logger.warning(f"Unknown simplification level '{simplification_level}', falling back to none")
            simplification_level = "none"
            
        simplification_suffix = self.simplification_levels[simplification_level]["prompt_suffix"]
        
        # Get target language name if code was provided
        target_lang_name = target_lang
        if target_lang in self.supported_languages:
            target_lang_name = self.supported_languages[target_lang]
            
        # Format prompt
        prompt = model_template.format(
            target_lang=target_lang_name,
            input_text=input_text
        )
        
        # Add simplification suffix if needed
        if simplification_level != "none":
            prompt += simplification_suffix
            
        return prompt
    
    def _translate_with_local_model(self, prompt: str) -> str:
        """
        Translate text using a local LLM model.
        
        This is a placeholder implementation.
        In a real system, this would use a local LLM for translation.
        
        Args:
            prompt: Translation prompt
            
        Returns:
            Translated text
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a local LLM
        logger.info("Using local translation model")
        
        # Simulate translation (for demonstration only)
        if "Spanish" in prompt:
            return "Esto es un texto traducido de ejemplo."
        elif "French" in prompt:
            return "Ceci est un exemple de texte traduit."
        elif "Chinese" in prompt:
            return "这是一个翻译文本的例子。"
        else:
            return "This is a translated text example."
    
    def _translate_with_api(self, prompt: str) -> str:
        """
        Translate text using an external API.
        
        This is a placeholder implementation.
        In a real system, this would call an external translation API.
        
        Args:
            prompt: Translation prompt
            
        Returns:
            Translated text
        """
        # This is a placeholder implementation
        # In a real implementation, this would call an external translation API
        logger.info("Using external translation API")
        
        # Check if API key is available
        if not self.api_key:
            raise ValueError("No API key available for external translation service")
            
        # Simulate API call (for demonstration only)
        time.sleep(0.5)  # Simulate network delay
        
        # Simulate translation based on target language
        if "Spanish" in prompt:
            return "Esto es un texto traducido mediante API."
        elif "French" in prompt:
            return "Ceci est un texte traduit via API."
        elif "Chinese" in prompt:
            return "这是通过API翻译的文本。"
        else:
            return "This is an API-translated text."
    
    def _translate_uncached(self, 
                          input_text: str, 
                          target_lang: str,
                          model: str,
                          simplification_level: str,
                          use_api: bool) -> str:
        """
        Perform translation without caching.
        
        Args:
            input_text: Text to translate
            target_lang: Target language code or name
            model: Translation model to use
            simplification_level: Text simplification level
            use_api: Whether to use external API
            
        Returns:
            Translated text
        """
        # Build translation prompt
        prompt = self._build_translation_prompt(
            input_text=input_text,
            target_lang=target_lang,
            model=model,
            simplification_level=simplification_level
        )
        
        # Perform translation with local model or API
        if use_api and self.api_key:
            translated_text = self._translate_with_api(prompt)
        else:
            translated_text = self._translate_with_local_model(prompt)
            
        return translated_text
    
    def translate_text(self, 
                      input_text: str, 
                      target_lang: str = "en",
                      source_lang: Optional[str] = None,
                      model: Optional[str] = None,
                      simplification_level: Optional[str] = None,
                      use_api: bool = False) -> Dict[str, Any]:
        """
        Translate text to the target language with optional simplification.
        
        Args:
            input_text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (if None, will be auto-detected)
            model: Translation model to use
            simplification_level: Text simplification level
            use_api: Whether to use external API
            
        Returns:
            Dictionary containing:
                - translated_text: Translated text
                - source_lang: Detected or provided source language
                - target_lang: Target language
                - model: Translation model used
                - simplification_level: Simplification level used
        """
        # Validate target language
        if target_lang not in self.supported_languages and target_lang not in self.supported_languages.values():
            raise ValueError(f"Unsupported target language: {target_lang}")
            
        # Detect source language if not provided
        detected_lang = None
        if not source_lang:
            lang_scores = self.detect_language(input_text)
            detected_lang = max(lang_scores.items(), key=lambda x: x[1])[0]
            source_lang = detected_lang
            logger.info(f"Detected source language: {source_lang} ({self.supported_languages.get(source_lang, 'Unknown')})")
        
        # Use defaults if not specified
        model = model or self.default_model
        simplification_level = simplification_level or self.default_simplification_level
        
        logger.info(f"Translating from {source_lang} to {target_lang} using {model} model with {simplification_level} simplification")
        
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            logger.info("Source and target languages are the same, skipping translation")
            return {
                "translated_text": input_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "model": model,
                "simplification_level": simplification_level,
                "skipped": True
            }
        
        # Perform translation with caching if enabled
        if self.enable_caching:
            # Create a cache key from all parameters
            cache_key = f"{input_text}_{target_lang}_{model}_{simplification_level}_{use_api}"
            translated_text = self._translation_cache(
                input_text, target_lang, model, simplification_level, use_api
            )
        else:
            translated_text = self._translate_uncached(
                input_text, target_lang, model, simplification_level, use_api
            )
        
        # Return result
        return {
            "translated_text": translated_text,
            "source_lang": source_lang,
            "detected_lang": detected_lang,
            "target_lang": target_lang,
            "model": model,
            "simplification_level": simplification_level,
            "skipped": False
        }
    
    def simplify_text(self, 
                     input_text: str, 
                     simplification_level: str = "medium",
                     preserve_language: bool = True) -> Dict[str, Any]:
        """
        Simplify text without changing the language.
        
        Args:
            input_text: Text to simplify
            simplification_level: Text simplification level
            preserve_language: Whether to maintain the original language
            
        Returns:
            Dictionary containing:
                - simplified_text: Simplified text
                - source_lang: Detected source language
                - simplification_level: Simplification level used
        """
        # Detect language
        lang_scores = self.detect_language(input_text)
        source_lang = max(lang_scores.items(), key=lambda x: x[1])[0]
        
        # If preserve_language is True, target language is the same as source
        target_lang = source_lang if preserve_language else "en"
        
        logger.info(f"Simplifying text in {self.supported_languages.get(source_lang, 'Unknown')} at {simplification_level} level")
        
        # Skip simplification if level is "none"
        if simplification_level == "none":
            logger.info("Simplification level is 'none', skipping simplification")
            return {
                "simplified_text": input_text,
                "source_lang": source_lang,
                "simplification_level": simplification_level,
                "skipped": True
            }
        
        # Perform translation/simplification
        result = self.translate_text(
            input_text=input_text,
            target_lang=target_lang,
            source_lang=source_lang,
            model="general",
            simplification_level=simplification_level
        )
        
        # Return result
        return {
            "simplified_text": result["translated_text"],
            "source_lang": source_lang,
            "simplification_level": simplification_level,
            "skipped": False
        }
    
    def translate_document(self, 
                          document_sections: Dict[str, str],
                          target_lang: str = "en",
                          source_lang: Optional[str] = None,
                          model: Optional[str] = None,
                          simplification_level: Optional[str] = None) -> Dict[str, Any]:
        """
        Translate a document with multiple sections.
        
        Args:
            document_sections: Dictionary mapping section names to text content
            target_lang: Target language code
            source_lang: Source language code (if None, will be auto-detected)
            model: Translation model to use
            simplification_level: Text simplification level
            
        Returns:
            Dictionary containing:
                - translated_sections: Dictionary of translated sections
                - source_lang: Detected or provided source language
                - target_lang: Target language
                - model: Translation model used
                - simplification_level: Simplification level used
        """
        # Use defaults if not specified
        model = model or self.default_model
        simplification_level = simplification_level or self.default_simplification_level
        
        # Detect source language if not provided using the longest section
        if not source_lang:
            longest_section = max(document_sections.items(), key=lambda x: len(x[1]))[1]
            lang_scores = self.detect_language(longest_section)
            source_lang = max(lang_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected source language: {source_lang} ({self.supported_languages.get(source_lang, 'Unknown')})")
        
        logger.info(f"Translating document with {len(document_sections)} sections from {source_lang} to {target_lang}")
        
        # Translate each section
        translated_sections = {}
        for section_name, text in document_sections.items():
            logger.info(f"Translating section: {section_name}")
            
            result = self.translate_text(
                input_text=text,
                target_lang=target_lang,
                source_lang=source_lang,
                model=model,
                simplification_level=simplification_level
            )
            
            translated_sections[section_name] = result["translated_text"]
        
        # Return result
        return {
            "translated_sections": translated_sections,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model,
            "simplification_level": simplification_level,
            "section_count": len(document_sections)
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.
        
        Returns:
            Dictionary mapping language codes to language names
        """
        return self.supported_languages
    
    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of available translation models.
        
        Returns:
            Dictionary mapping model names to model info
        """
        return self.models
    
    def get_simplification_levels(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of available simplification levels.
        
        Returns:
            Dictionary mapping level names to level info
        """
        return self.simplification_levels