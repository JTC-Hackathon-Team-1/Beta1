# pipeline.py

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid

# Import component modules
from ner.extract_entities import EntityExtractor
from translation.translate_text import TranslationLLM
from llms.governance import GovernanceLLM
from llms.simplifier import SimplifierLLM  # We'll also need to integrate the simplifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CasaLinguaPipeline:
    """
    CasaLingua - Text Pipeline
    This module coordinates the full pipeline: NER, Translation, Simplification, and Governance.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 ner_models: Optional[List[str]] = None,
                 enable_governance: bool = True,
                 enable_caching: bool = True,
                 session_store_path: Optional[str] = None):
        """
        Initialize the CasaLingua pipeline.
        
        Args:
            config_path: Path to configuration file
            ner_models: List of spaCy model names for NER
            enable_governance: Whether to enable governance checks
            enable_caching: Whether to enable result caching
            session_store_path: Path to store session data
        """
        self.enable_governance = enable_governance
        self.enable_caching = enable_caching
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded pipeline configuration from {config_path}")
        
        # Set up session store
        self.session_store_path = session_store_path or self.config.get("session_store_path", "sessions")
        os.makedirs(self.session_store_path, exist_ok=True)
        
        # Initialize component modules
        logger.info("Initializing pipeline components")
        
        # Initialize NER module
        ner_models = ner_models or self.config.get("ner_models", ["en_core_web_sm"])
        ner_rules_path = self.config.get("ner_rules_path")
        self.entity_extractor = EntityExtractor(
            models=ner_models,
            custom_rules_path=ner_rules_path,
            enable_confidence=True
        )
        
        # Initialize Translation LLM
        translation_config_path = self.config.get("translation_config_path")
        self.translation_llm = TranslationLLM(
            config_path=translation_config_path,
            enable_caching=enable_caching
        )
        
        # Initialize Simplifier LLM
        simplifier_config_path = self.config.get("simplifier_config_path")
        self.simplifier_llm = SimplifierLLM(
            config_path=simplifier_config_path,
            enable_caching=enable_caching
        )
        
        # Initialize Governance LLM if enabled
        if enable_governance:
            governance_config_path = self.config.get("governance_config_path")
            self.governance_llm = GovernanceLLM(
                config_path=governance_config_path,
                enable_audit_log=True
            )
        
        logger.info("Pipeline initialization complete")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    def _save_session_data(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save session data to persistent storage."""
        if not self.enable_caching:
            return
            
        session_file = os.path.join(self.session_store_path, f"{session_id}.json")
        
        try:
            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session data: {str(e)}")
    
    def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from persistent storage."""
        session_file = os.path.join(self.session_store_path, f"{session_id}.json")
        
        if not os.path.exists(session_file):
            return None
            
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session data: {str(e)}")
            return None
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """
        Detect the language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping language codes to confidence scores
        """
        return self.translation_llm.detect_language(text)
    
    def run_pipeline(self, 
                    raw_text: str, 
                    target_lang: str = "en",
                    source_lang: Optional[str] = None,
                    simplification_level: str = "medium",
                    translation_model: str = "housing",
                    session_id: Optional[str] = None,
                    extract_entities: bool = True) -> Dict[str, Any]:
        """
        Run the full CasaLingua pipeline on the input text.
        
        Args:
            raw_text: Input text to process
            target_lang: Target language code
            source_lang: Source language code (if None, will be auto-detected)
            simplification_level: Text simplification level
            translation_model: Translation model to use
            session_id: Session ID for continuity
            extract_entities: Whether to extract entities
            
        Returns:
            Dictionary containing:
                - processed_text: Translated/simplified text
                - entities: Extracted entities (if requested)
                - audit_report: Governance audit report (if enabled)
                - source_lang: Source language
                - target_lang: Target language
                - session_id: Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = self._generate_session_id()
            
        logger.info(f"Starting pipeline run with session ID: {session_id}")
        
        # Record start time
        start_time = datetime.now()
        
        # Step 1: Detect language if not provided
        if not source_lang:
            logger.info("Detecting source language")
            lang_scores = self.detect_language(raw_text)
            source_lang = max(lang_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected source language: {source_lang}")
        
        # Step 2: Extract entities if requested
        entities = {}
        if extract_entities:
            logger.info("Extracting entities")
            entities = self.entity_extractor.extract_entities(raw_text)
            logger.info(f"Extracted {len(entities)} entities")
        
        # Step 3: Translate text if target language is different from source
        translated_text = raw_text
        translation_result = {}
        
        if target_lang != source_lang:
            logger.info(f"Translating from {source_lang} to {target_lang}")
            translation_result = self.translation_llm.translate_text(
                input_text=raw_text,
                target_lang=target_lang,
                source_lang=source_lang,
                model=translation_model,
                simplification_level="none"  # We'll handle simplification separately
            )
            translated_text = translation_result.get("translated_text", raw_text)
        
        # Step 4: Apply text simplification
        logger.info(f"Simplifying text at {simplification_level} level")
        simplification_result = self.simplifier_llm.simplify_text(
            input_text=translated_text,
            level=simplification_level,
            domain="housing",
            preserve_entities=list(entities.keys()) if entities else []
        )
        processed_text = simplification_result.get("simplified_text", translated_text)
        
        # Step 5: Run governance checks if enabled
        audit_report = {}
        if self.enable_governance:
            logger.info("Running governance checks")
            audit_report = self.governance_llm.audit_translation(
                original_text=raw_text,
                translated_text=processed_text,
                extracted_entities=entities,
                session_id=session_id
            )
        
        # Record end time and duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Prepare result
        result = {
            "processed_text": processed_text,
            "entities": entities,
            "audit_report": audit_report,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
            "processing_time_ms": duration_ms,
            "simplification_level": simplification_level,
            "translation_model": translation_model
        }
        
        # Save session data
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "raw_text": raw_text,
                "target_lang": target_lang,
                "source_lang": source_lang,
                "simplification_level": simplification_level,
                "translation_model": translation_model
            },
            "output": result
        }
        self._save_session_data(session_id, session_data)
        
        logger.info(f"Pipeline run completed in {duration_ms:.2f}ms")
        return result
    
    def process_document(self, 
                        document_sections: Dict[str, str],
                        target_lang: str = "en",
                        source_lang: Optional[str] = None,
                        simplification_level: str = "medium",
                        translation_model: str = "housing",
                        session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with multiple sections through the pipeline.
        
        Args:
            document_sections: Dictionary mapping section names to text content
            target_lang: Target language code
            source_lang: Source language code (if None, will be auto-detected)
            simplification_level: Text simplification level
            translation_model: Translation model to use
            session_id: Session ID for continuity
            
        Returns:
            Dictionary containing processed document sections and metadata
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = self._generate_session_id()
            
        logger.info(f"Starting document processing with session ID: {session_id}")
        
        # Record start time
        start_time = datetime.now()
        
        # Step 1: Detect language from the largest section if not provided
        if not source_lang:
            logger.info("Detecting source language from document")
            largest_section = max(document_sections.items(), key=lambda x: len(x[1]))[1]
            lang_scores = self.detect_language(largest_section)
            source_lang = max(lang_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected source language: {source_lang}")
        
        # Step 2: Extract entities from the full document
        logger.info("Extracting entities from document")
        full_text = "\n\n".join(document_sections.values())
        entities = self.entity_extractor.extract_entities(full_text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Step 3: Process each section
        processed_sections = {}
        section_audit_reports = {}
        
        for section_name, section_text in document_sections.items():
            logger.info(f"Processing section: {section_name}")
            
            # Process this section
            section_result = self.run_pipeline(
                raw_text=section_text,
                target_lang=target_lang,
                source_lang=source_lang,
                simplification_level=simplification_level,
                translation_model=translation_model,
                session_id=f"{session_id}_{section_name}",
                extract_entities=False  # We already extracted entities from the full document
            )
            
            processed_sections[section_name] = section_result["processed_text"]
            
            if self.enable_governance:
                section_audit_reports[section_name] = section_result["audit_report"]
        
        # Step 4: Run governance checks on the full document if enabled
        full_audit_report = {}
        if self.enable_governance:
            logger.info("Running governance checks on full document")
            full_processed_text = "\n\n".join(processed_sections.values())
            full_audit_report = self.governance_llm.audit_translation(
                original_text=full_text,
                translated_text=full_processed_text,
                extracted_entities=entities,
                session_id=session_id
            )
        
        # Record end time and duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Prepare result
        result = {
            "processed_sections": processed_sections,
            "entities": entities,
            "section_audit_reports": section_audit_reports,
            "document_audit_report": full_audit_report,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
            "processing_time_ms": duration_ms,
            "simplification_level": simplification_level,
            "translation_model": translation_model,
            "section_count": len(document_sections)
        }
        
        # Save session data
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "document_sections": document_sections,
                "target_lang": target_lang,
                "source_lang": source_lang,
                "simplification_level": simplification_level,
                "translation_model": translation_model
            },
            "output": result
        }
        self._save_session_data(session_id, session_data)
        
        logger.info(f"Document processing completed in {duration_ms:.2f}ms")
        return result
    
    def process_form(self,
                    form_template: Dict[str, Any],
                    user_data: Optional[Dict[str, str]] = None,
                    conversation_text: Optional[str] = None,
                    target_lang: str = "en",
                    source_lang: Optional[str] = None,
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a form through the pipeline.
        
        Args:
            form_template: Form template definition
            user_data: User-provided form data (if any)
            conversation_text: Text from conversation to extract data from
            target_lang: Target language code
            source_lang: Source language code (if None, will be auto-detected)
            session_id: Session ID for continuity
            
        Returns:
            Dictionary containing processed form data and metadata
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = self._generate_session_id()
            
        logger.info(f"Starting form processing with session ID: {session_id}")
        
        # Record start time
        start_time = datetime.now()
        
        # Step 1: Extract entities from conversation if provided
        entities = {}
        if conversation_text:
            logger.info("Extracting entities from conversation")
            entities = self.entity_extractor.extract_entities(conversation_text)
            logger.info(f"Extracted {len(entities)} entities")
        
        # Step 2: Map entities to form fields
        form_data = {}
        if entities:
            logger.info("Mapping entities to form fields")
            form_data = self.entity_extractor.map_entities_to_form(entities, form_template)
        
        # Step 3: Merge with user-provided data if available
        if user_data:
            logger.info("Merging with user-provided form data")
            for field_id, value in user_data.items():
                if field_id in form_data and not value:
                    # Skip empty user values if we already have an entity-derived value
                    continue
                form_data[field_id] = value
        
        # Step 4: Translate form fields if needed
        if target_lang != source_lang and source_lang:
            logger.info(f"Translating form labels from {source_lang} to {target_lang}")
            
            # Translate form field labels
            translated_template = {}
            for field_id, field_def in form_template.items():
                translated_field = field_def.copy()
                
                if "label" in field_def:
                    translation_result = self.translation_llm.translate_text(
                        input_text=field_def["label"],
                        target_lang=target_lang,
                        source_lang=source_lang
                    )
                    translated_field["label"] = translation_result["translated_text"]
                
                if "description" in field_def:
                    translation_result = self.translation_llm.translate_text(
                        input_text=field_def["description"],
                        target_lang=target_lang,
                        source_lang=source_lang
                    )
                    translated_field["description"] = translation_result["translated_text"]
                
                translated_template[field_id] = translated_field
            
            form_template = translated_template
        
        # Step 5: Run governance checks on form data if enabled
        audit_report = {}
        if self.enable_governance and form_data:
            logger.info("Running governance checks on form data")
            audit_report = self.governance_llm.audit_form_completion(
                original_form=form_template,
                filled_form=form_data,
                extracted_entities=entities,
                session_id=session_id
            )
        
        # Record end time and duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Prepare result
        result = {
            "form_template": form_template,
            "form_data": form_data,
            "entities": entities,
            "audit_report": audit_report,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
            "processing_time_ms": duration_ms,
            "completion_rate": len([v for v in form_data.values() if v]) / len(form_template) if form_template else 0
        }
        
        # Save session data
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "input": {
                "form_template": form_template,
                "user_data": user_data,
                "conversation_text": conversation_text,
                "target_lang": target_lang,
                "source_lang": source_lang
            },
            "output": result
        }
        self._save_session_data(session_id, session_data)
        
        logger.info(f"Form processing completed in {duration_ms:.2f}ms")
        return result
    
    def continue_session(self, 
                        session_id: str,
                        new_text: Optional[str] = None,
                        user_data: Optional[Dict[str, str]] = None,
                        action: str = "process") -> Dict[str, Any]:
        """
        Continue processing with an existing session.
        
        Args:
            session_id: Session ID to continue
            new_text: New text to process
            user_data: New user data to incorporate
            action: Action to perform ('process', 'translate', 'simplify')
            
        Returns:
            Updated session data
        """
        # Load existing session data
        session_data = self._load_session_data(session_id)
        
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")
            
        logger.info(f"Continuing session: {session_id}")
        
        # Extract previous input/output
        prev_input = session_data["input"]
        prev_output = session_data["output"]
        
        # Determine action to perform
        if action == "process" and new_text:
            # Process new text with same parameters as before
            result = self.run_pipeline(
                raw_text=new_text,
                target_lang=prev_input.get("target_lang", "en"),
                source_lang=prev_input.get("source_lang"),
                simplification_level=prev_input.get("simplification_level", "medium"),
                translation_model=prev_input.get("translation_model", "housing"),
                session_id=session_id
            )
            
        elif action == "translate" and new_text:
            # Just translate the new text
            translation_result = self.translation_llm.translate_text(
                input_text=new_text,
                target_lang=prev_input.get("target_lang", "en"),
                source_lang=prev_input.get("source_lang"),
                model=prev_input.get("translation_model", "housing")
            )
            
            result = {
                "processed_text": translation_result["translated_text"],
                "source_lang": translation_result["source_lang"],
                "target_lang": translation_result["target_lang"],
                "session_id": session_id
            }
            
        elif action == "simplify" and new_text:
            # Just simplify the new text
            simplification_result = self.simplifier_llm.simplify_text(
                input_text=new_text,
                level=prev_input.get("simplification_level", "medium"),
                domain="housing"
            )
            
            result = {
                "processed_text": simplification_result["simplified_text"],
                "simplification_level": prev_input.get("simplification_level", "medium"),
                "session_id": session_id
            }
            
        elif "form_template" in prev_input and user_data:
            # Update form with new user data
            result = self.process_form(
                form_template=prev_input["form_template"],
                user_data=user_data,
                target_lang=prev_input.get("target_lang", "en"),
                source_lang=prev_input.get("source_lang"),
                session_id=session_id
            )
            
        else:
            raise ValueError(f"Invalid continuation action: {action}")
        
        # Update session data
        session_data["timestamp"] = datetime.now().isoformat()
        
        if new_text:
            session_data["input"]["additional_text"] = new_text
            
        if user_data:
            session_data["input"]["additional_user_data"] = user_data
            
        session_data["output"] = result
        
        # Save updated session data
        self._save_session_data(session_id, session_data)
        
        logger.info(f"Session {session_id} continued with action: {action}")
        return result