# ner/extract_entities.py

import os
import logging
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import dateparser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    CasaLingua - Named Entity Recognition Module
    Extracts form-relevant fields such as name, address, date of birth.
    """
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 custom_rules_path: Optional[str] = None,
                 enable_confidence: bool = True):
        """
        Initialize the Entity Extractor.
        
        Args:
            models: List of spaCy model names to load (default: ["en_core_web_sm"])
            custom_rules_path: Path to JSON file with custom extraction rules
            enable_confidence: Whether to include confidence scores in output
        """
        self.enable_confidence = enable_confidence
        
        # Default to English model if none specified
        if not models:
            models = ["en_core_web_sm"]
            
        # Load spaCy models
        self.nlp_models = {}
        for model_name in models:
            try:
                self.nlp_models[model_name] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model {model_name}: {str(e)}")
                raise
        
        # Set primary model (first one loaded)
        self.primary_model = models[0]
        
        # Load custom rules if provided
        self.custom_rules = {}
        if custom_rules_path and os.path.exists(custom_rules_path):
            with open(custom_rules_path, 'r') as f:
                self.custom_rules = json.load(f)
            logger.info(f"Loaded custom NER rules from {custom_rules_path}")
            
        # Add custom entity patterns to models
        if self.custom_rules.get("patterns"):
            for model_name, nlp in self.nlp_models.items():
                self._add_custom_patterns(nlp, self.custom_rules["patterns"])
                
        # Register custom components
        if "address" not in self.custom_rules.get("disabled_components", []):
            for nlp in self.nlp_models.values():
                if not nlp.has_pipe("address_detector"):
                    nlp.add_pipe("address_detector", last=True)
                    
        # Set up regex patterns for various entities
        self._setup_regex_patterns()
        
        logger.info("Entity Extractor initialized successfully")
    
    def _setup_regex_patterns(self):
        """Set up regex patterns for entity extraction."""
        self.patterns = {
            "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "phone": r'\b\(\d{3}\)\s?\d{3}[-\s]?\d{4}\b|\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "time": r'\b\d{1,2}:\d{2}\s?(?:[aApP][mM])?\b',
            "zip_code": r'\b\d{5}(?:-\d{4})?\b',
            "currency": r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
            "age": r'\b(?:age|aged)\s+\d+\b|\b\d+\s+years\s+old\b',
            "url": r'\bhttps?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        }
    
    def _add_custom_patterns(self, nlp: Language, patterns: List[Dict[str, Any]]):
        """
        Add custom entity patterns to the model.
        
        Args:
            nlp: spaCy Language model
            patterns: List of pattern dictionaries
        """
        # Check if the model has an entity ruler
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = nlp.get_pipe("entity_ruler")
            
        # Add patterns
        ruler.add_patterns(patterns)
        logger.info(f"Added {len(patterns)} custom patterns to {nlp.meta['name']}")
    
    @Language.component("address_detector")
    def address_detector(self, doc: Doc) -> Doc:
        """
        Custom spaCy component to detect address patterns.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Doc with address entities added
        """
        # Simple address pattern: number + words + comma + words + state/zip
        address_pattern = r'\b\d+\s+[A-Za-z\s]+(?:,\s+[A-Za-z\s]+)(?:,\s+[A-Za-z]{2}\s+\d{5}(?:-\d{4})?)?\b'
        
        for match in re.finditer(address_pattern, doc.text):
            start, end = match.span()
            start_char = start
            end_char = end
            
            # Find the token span
            start_token = None
            end_token = None
            
            for i, token in enumerate(doc):
                if token.idx <= start_char and token.idx + len(token.text) > start_char:
                    start_token = i
                if token.idx < end_char and token.idx + len(token.text) >= end_char:
                    end_token = i
                    break
            
            if start_token is not None and end_token is not None:
                # Avoid overlapping with existing entities
                if any(start_token <= ent.start <= end_token or 
                       start_token <= ent.end <= end_token or
                       ent.start <= start_token <= ent.end
                       for ent in doc.ents):
                    continue
                    
                span = doc[start_token:end_token+1]
                doc.ents = list(doc.ents) + [(span, "ADDRESS")]
        
        return doc
    
    def _extract_with_regex(self, text: str) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """
        Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of (value, span) tuples
        """
        results = {}
        
        for entity_type, pattern in self.patterns.items():
            if entity_type in self.custom_rules.get("disabled_entities", []):
                continue
                
            matches = []
            for match in re.finditer(pattern, text):
                value = match.group(0)
                span = match.span()
                matches.append((value, span))
                
            if matches:
                results[entity_type] = matches
                
        return results
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """
        Parse date text into a datetime object.
        
        Args:
            date_text: Text containing a date
            
        Returns:
            Parsed datetime object or None if parsing fails
        """
        try:
            return dateparser.parse(date_text)
        except Exception:
            return None
    
    def _normalize_entity(self, entity_type: str, value: str) -> Any:
        """
        Normalize entity values based on type.
        
        Args:
            entity_type: Type of entity
            value: Raw entity value
            
        Returns:
            Normalized value
        """
        if entity_type in ["DATE", "DOB", "date"]:
            date_obj = self._parse_date(value)
            if date_obj:
                return date_obj.strftime("%Y-%m-%d")
            
        elif entity_type in ["PHONE_NUMBER", "phone"]:
            # Extract digits only
            digits = re.sub(r'\D', '', value)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            
        elif entity_type in ["ZIP_CODE", "zip_code"]:
            # Ensure consistent format
            digits = re.sub(r'\D', '', value)
            if len(digits) == 5:
                return digits
            elif len(digits) == 9:
                return f"{digits[:5]}-{digits[5:]}"
                
        # Default: return as is
        return value
    
    def _merge_entities(self, spacy_entities: Dict[str, List[Dict[str, Any]]], 
                       regex_entities: Dict[str, List[Tuple[str, Tuple[int, int]]]]) -> Dict[str, Any]:
        """
        Merge entities from spaCy and regex extraction.
        
        Args:
            spacy_entities: Entities from spaCy models
            regex_entities: Entities from regex patterns
            
        Returns:
            Merged entity dictionary
        """
        merged = {}
        
        # Add spaCy entities
        for entity_type, entities in spacy_entities.items():
            if entities:
                # Use the first entity with highest confidence
                best_entity = max(entities, key=lambda x: x["confidence"])
                normalized = self._normalize_entity(entity_type, best_entity["text"])
                
                if self.enable_confidence:
                    merged[entity_type.lower()] = {
                        "value": normalized,
                        "confidence": best_entity["confidence"],
                        "source": "spacy"
                    }
                else:
                    merged[entity_type.lower()] = normalized
        
        # Add regex entities
        for entity_type, entities in regex_entities.items():
            if entity_type.lower() not in merged and entities:
                normalized = self._normalize_entity(entity_type, entities[0][0])
                
                if self.enable_confidence:
                    merged[entity_type.lower()] = {
                        "value": normalized,
                        "confidence": 0.85,  # Default confidence for regex matches
                        "source": "regex"
                    }
                else:
                    merged[entity_type.lower()] = normalized
        
        # Perform entity-specific post-processing
        self._post_process_entities(merged)
        
        return merged
    
    def _post_process_entities(self, entities: Dict[str, Any]) -> None:
        """
        Perform post-processing on extracted entities.
        
        Args:
            entities: Dictionary of extracted entities
        """
        # Extract name components if a full name is present
        if "person" in entities:
            name_value = entities["person"]["value"] if isinstance(entities["person"], dict) else entities["person"]
            name_parts = name_value.split()
            
            if len(name_parts) >= 2:
                # Assume first name, last name format
                if self.enable_confidence:
                    confidence = entities["person"]["confidence"] if isinstance(entities["person"], dict) else 0.8
                    
                    if "first_name" not in entities:
                        entities["first_name"] = {
                            "value": name_parts[0],
                            "confidence": confidence * 0.9,
                            "source": "derived"
                        }
                        
                    if "last_name" not in entities:
                        entities["last_name"] = {
                            "value": name_parts[-1],
                            "confidence": confidence * 0.9,
                            "source": "derived"
                        }
                else:
                    if "first_name" not in entities:
                        entities["first_name"] = name_parts[0]
                        
                    if "last_name" not in entities:
                        entities["last_name"] = name_parts[-1]
        
        # Standardize address
        if "address" in entities:
            addr_value = entities["address"]["value"] if isinstance(entities["address"], dict) else entities["address"]
            
            # Try to extract city, state, zip from address
            addr_parts = addr_value.split(',')
            if len(addr_parts) >= 2:
                # Last part might contain state and zip
                last_part = addr_parts[-1].strip()
                state_zip_match = re.search(r'([A-Z]{2})\s+(\d{5}(?:-\d{4})?)', last_part)
                
                if state_zip_match:
                    state, zip_code = state_zip_match.groups()
                    
                    if self.enable_confidence:
                        confidence = entities["address"]["confidence"] if isinstance(entities["address"], dict) else 0.8
                        
                        if "state" not in entities:
                            entities["state"] = {
                                "value": state,
                                "confidence": confidence * 0.9,
                                "source": "derived"
                            }
                            
                        if "zip_code" not in entities:
                            entities["zip_code"] = {
                                "value": zip_code,
                                "confidence": confidence * 0.9,
                                "source": "derived"
                            }
                    else:
                        if "state" not in entities:
                            entities["state"] = state
                            
                        if "zip_code" not in entities:
                            entities["zip_code"] = zip_code
    
    def extract_entities(self, text: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence threshold for entities
            
        Returns:
            Dictionary of extracted entities
        """
        logger.info("Starting entity extraction")
        
        # Perform regex-based extraction
        regex_entities = self._extract_with_regex(text)
        
        # Perform spaCy-based extraction
        spacy_entities = {}
        
        for model_name, nlp in self.nlp_models.items():
            doc = nlp(text)
            
            # Extract entities from spaCy doc
            for ent in doc.ents:
                if ent.label_ not in spacy_entities:
                    spacy_entities[ent.label_] = []
                    
                # Map spaCy entity types to our normalized types
                entity_type = ent.label_
                
                # Calculate confidence (in a real system, this would be model-specific)
                # Here we use a simplified approach based on entity length
                confidence = min(0.95, 0.5 + (len(ent.text) / 100))
                
                spacy_entities[entity_type].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": confidence
                })
        
        # Merge entities from different sources
        merged_entities = self._merge_entities(spacy_entities, regex_entities)
        
        # Filter by confidence threshold if enabled
        if self.enable_confidence:
            merged_entities = {
                k: v for k, v in merged_entities.items()
                if not isinstance(v, dict) or v["confidence"] >= confidence_threshold
            }
        
        # Apply custom transformations based on rules
        if "transformations" in self.custom_rules:
            for transform in self.custom_rules["transformations"]:
                if transform["type"] == "map" and transform["source"] in merged_entities:
                    source_val = merged_entities[transform["source"]]
                    if isinstance(source_val, dict):
                        source_val = source_val["value"]
                        
                    if transform["source_value"] == "*" or source_val == transform["source_value"]:
                        if transform["target"] not in merged_entities:
                            if self.enable_confidence:
                                merged_entities[transform["target"]] = {
                                    "value": transform["target_value"],
                                    "confidence": 0.7,
                                    "source": "rule"
                                }
                            else:
                                merged_entities[transform["target"]] = transform["target_value"]
        
        # Log extraction results
        entity_count = len(merged_entities)
        logger.info(f"Extracted {entity_count} unique entities")
        
        return merged_entities
    
    def map_entities_to_form(self, entities: Dict[str, Any], form_fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map extracted entities to form fields.
        
        Args:
            entities: Dictionary of extracted entities
            form_fields: Dictionary of form field definitions
            
        Returns:
            Dictionary with form fields populated with entity values
        """
        logger.info(f"Mapping {len(entities)} entities to {len(form_fields)} form fields")
        
        mapped_form = {}
        
        for field_id, field_def in form_fields.items():
            # Skip fields with no mapping info
            if "entity_type" not in field_def:
                mapped_form[field_id] = ""
                continue
                
            entity_type = field_def["entity_type"]
            
            # Check if we have this entity type
            if entity_type in entities:
                entity_value = entities[entity_type]
                if isinstance(entity_value, dict) and "value" in entity_value:
                    entity_value = entity_value["value"]
                    
                # Apply any field-specific formatting
                if "format" in field_def:
                    if field_def["format"] == "uppercase":
                        entity_value = str(entity_value).upper()
                    elif field_def["format"] == "lowercase":
                        entity_value = str(entity_value).lower()
                    elif field_def["format"] == "title":
                        entity_value = str(entity_value).title()
                    elif field_def["format"] == "date" and isinstance(entity_value, str):
                        date_obj = self._parse_date(entity_value)
                        if date_obj:
                            entity_value = date_obj.strftime(field_def.get("date_format", "%Y-%m-%d"))
                            
                mapped_form[field_id] = entity_value
            else:
                # No matching entity found
                mapped_form[field_id] = ""
        
        return mapped_form
    
    def get_entity_highlights(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get entity highlights for text visualization.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of highlight info
        """
        # Extract with spaCy for spans
        highlights = {}
        doc = self.nlp_models[self.primary_model](text)
        
        for ent in doc.ents:
            entity_type = ent.label_
            
            if entity_type not in highlights:
                highlights[entity_type] = []
                
            highlights[entity_type].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Add regex matches
        regex_entities = self._extract_with_regex(text)
        
        for entity_type, matches in regex_entities.items():
            if entity_type not in highlights:
                highlights[entity_type] = []
                
            for value, (start, end) in matches:
                # Check if this span overlaps with existing highlights
                overlap = False
                for entity_list in highlights.values():
                    for entity in entity_list:
                        if (start <= entity["start"] <= end or
                            start <= entity["end"] <= end or
                            entity["start"] <= start <= entity["end"]):
                            overlap = True
                            break
                            
                if not overlap:
                    highlights[entity_type].append({
                        "text": value,
                        "start": start,
                        "end": end
                    })
        
        return highlights