# llms/governance.py

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import re
import hashlib
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiasRating(str, Enum):
    """Enumeration of bias check ratings."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

class PIICompliance(str, Enum):
    """Enumeration of PII compliance ratings."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

class SemanticFidelity(str, Enum):
    """Enumeration of semantic fidelity ratings."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class GovernanceLLM:
    """
    CasaLingua - Governance LLM
    Ensures ethical compliance, fidelity, and confidence scoring.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 pii_threshold: float = 0.7,
                 bias_threshold: float = 0.7,
                 fidelity_threshold: float = 0.8,
                 enable_audit_log: bool = True):
        """
        Initialize the Governance LLM.
        
        Args:
            config_path: Path to configuration file
            pii_threshold: Threshold for PII detection (0-1)
            bias_threshold: Threshold for bias detection (0-1)
            fidelity_threshold: Threshold for semantic fidelity (0-1)
            enable_audit_log: Whether to save audit logs to file
        """
        self.pii_threshold = pii_threshold
        self.bias_threshold = bias_threshold
        self.fidelity_threshold = fidelity_threshold
        self.enable_audit_log = enable_audit_log
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded governance configuration from {config_path}")
                
                # Override defaults with config values if present
                self.pii_threshold = self.config.get("pii_threshold", self.pii_threshold)
                self.bias_threshold = self.config.get("bias_threshold", self.bias_threshold)
                self.fidelity_threshold = self.config.get("fidelity_threshold", self.fidelity_threshold)
                self.enable_audit_log = self.config.get("enable_audit_log", self.enable_audit_log)
        
        # Initialize PII detection patterns
        self.pii_patterns = {
            "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "phone": r'\b\(\d{3}\)\s?\d{3}[-\s]?\d{4}\b|\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "dob": r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](\d{4}|\d{2})\b'
        }
        
        # Initialize bias terms (simplified for demonstration)
        self.bias_terms = {
            "gender": ["he", "she", "man", "woman", "male", "female"],
            "race": ["black", "white", "asian", "hispanic", "latino", "latina"],
            "religion": ["christian", "muslim", "jewish", "hindu", "buddhist"],
            "age": ["young", "old", "elderly", "senior", "juvenile"]
        }
        
        # Set up audit log directory
        self.audit_log_dir = self.config.get("audit_log_dir", "audit_logs")
        if self.enable_audit_log and not os.path.exists(self.audit_log_dir):
            os.makedirs(self.audit_log_dir)
            logger.info(f"Created audit log directory: {self.audit_log_dir}")
            
        logger.info("Governance LLM initialized successfully")
    
    def _detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect personally identifiable information (PII) in text.
        
        Args:
            text: Input text to check
            
        Returns:
            Dictionary mapping PII type to list of found instances
        """
        found_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found_pii[pii_type] = matches
                
        return found_pii
    
    def _check_bias(self, text: str) -> Dict[str, List[str]]:
        """
        Check for potentially biased language in text.
        
        Args:
            text: Input text to check
            
        Returns:
            Dictionary mapping bias category to list of found terms
        """
        found_bias = {}
        text_lower = text.lower()
        
        for category, terms in self.bias_terms.items():
            found = []
            for term in terms:
                # Check for whole word matches
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(term)
            
            if found:
                found_bias[category] = found
                
        return found_bias
    
    def _check_semantic_fidelity(self, original_text: str, translated_text: str) -> float:
        """
        Estimate semantic fidelity between original and translated text.
        
        In a real implementation, this would use embeddings or an LLM to compare the texts.
        This is a simplified version for demonstration purposes.
        
        Args:
            original_text: Source text
            translated_text: Target text to compare
            
        Returns:
            Fidelity score (0-1)
        """
        # Very simplified fidelity check - word count ratio
        if not original_text or not translated_text:
            return 0.0
            
        orig_words = len(original_text.split())
        trans_words = len(translated_text.split())
        
        # If lengths are very different, fidelity is likely low
        if orig_words == 0 or trans_words == 0:
            return 0.0
            
        ratio = min(orig_words, trans_words) / max(orig_words, trans_words)
        
        # This is a simplified simulation - in reality would use semantic comparison
        # We'll add some noise to simulate a more realistic score
        import random
        noise = random.uniform(-0.1, 0.1)
        score = min(1.0, max(0.0, ratio + noise))
        
        return score
    
    def _check_entity_alignment(self, text: str, entities: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check if all extracted entities are present in the translated text.
        
        Args:
            text: Text to check
            entities: Dictionary of entities to check for
            
        Returns:
            Dictionary mapping entity keys to boolean presence indicator
        """
        alignment = {}
        
        for key, value in entities.items():
            # For string entities, check if they're in the text
            if isinstance(value, str):
                alignment[key] = value.lower() in text.lower()
            # For numeric entities, convert to string and check
            elif isinstance(value, (int, float)):
                alignment[key] = str(value) in text
            # For lists, check if any item is in the text
            elif isinstance(value, list):
                alignment[key] = any(str(item).lower() in text.lower() for item in value)
            # For dictionaries, check keys and values
            elif isinstance(value, dict):
                sub_alignment = all(
                    str(v).lower() in text.lower() 
                    for v in value.values() 
                    if str(v).strip()
                )
                alignment[key] = sub_alignment
            else:
                alignment[key] = False
                
        return alignment
    
    def _calculate_confidence_score(self, bias_check: BiasRating, pii_compliance: PIICompliance, 
                                   semantic_fidelity: float, entity_alignment: Dict[str, bool]) -> float:
        """
        Calculate overall confidence score based on all checks.
        
        Args:
            bias_check: Bias check rating
            pii_compliance: PII compliance rating
            semantic_fidelity: Semantic fidelity score (0-1)
            entity_alignment: Entity alignment check results
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from semantic fidelity (40% weight)
        confidence = 0.4 * semantic_fidelity
        
        # Add bias check component (20% weight)
        if bias_check == BiasRating.PASS:
            confidence += 0.2
        elif bias_check == BiasRating.WARN:
            confidence += 0.1
        
        # Add PII compliance component (20% weight)
        if pii_compliance == PIICompliance.PASS:
            confidence += 0.2
        elif pii_compliance == PIICompliance.WARN:
            confidence += 0.1
        
        # Add entity alignment component (20% weight)
        if entity_alignment:
            alignment_score = sum(1 for v in entity_alignment.values() if v) / len(entity_alignment) if entity_alignment else 0
            confidence += 0.2 * alignment_score
        
        return min(1.0, max(0.0, confidence))
    
    def audit_translation(self, 
                         original_text: str,
                         translated_text: str, 
                         extracted_entities: Dict[str, Any],
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit a translation for ethical compliance, fidelity, and confidence.
        
        Args:
            original_text: Source text
            translated_text: Translated or simplified text
            extracted_entities: Dictionary of extracted entities from original text
            session_id: Optional session identifier for audit logging
            
        Returns:
            Dictionary containing audit results including:
                - bias_check: Rating of potential bias
                - pii_compliance: Rating of PII handling
                - semantic_fidelity: Rating of meaning preservation
                - entity_alignment: Whether entities were preserved
                - confidence_score: Overall confidence score
                - audit_report: Detailed audit report
        """
        logger.info("Starting translation audit")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = hashlib.md5(f"{time.time()}-{translated_text[:50]}".encode()).hexdigest()
        
        # 1. Check for PII in translated text
        pii_found = self._detect_pii(translated_text)
        
        if pii_found:
            # Check if PII should have been redacted
            orig_pii = self._detect_pii(original_text)
            
            # If original text had PII that should be redacted but wasn't
            if orig_pii and any(pii_type not in self.config.get("allowed_pii", []) for pii_type in orig_pii):
                pii_compliance = PIICompliance.FAIL
                pii_details = f"Found unredacted PII: {list(pii_found.keys())}"
            else:
                # PII is present but may be allowed
                pii_compliance = PIICompliance.WARN
                pii_details = f"Found allowable PII: {list(pii_found.keys())}"
        else:
            pii_compliance = PIICompliance.PASS
            pii_details = "No PII detected"
        
        # 2. Check for potential bias
        bias_found = self._check_bias(translated_text)
        
        if bias_found:
            # Count total bias terms
            bias_term_count = sum(len(terms) for terms in bias_found.values())
            word_count = len(translated_text.split())
            
            # Calculate bias density
            bias_density = bias_term_count / word_count if word_count > 0 else 0
            
            if bias_density > self.bias_threshold:
                bias_check = BiasRating.FAIL
                bias_details = f"High bias density: {bias_density:.2f}"
            else:
                bias_check = BiasRating.WARN
                bias_details = f"Found potential bias terms: {bias_found}"
        else:
            bias_check = BiasRating.PASS
            bias_details = "No potential bias detected"
        
        # 3. Check semantic fidelity
        fidelity_score = self._check_semantic_fidelity(original_text, translated_text)
        
        if fidelity_score >= self.fidelity_threshold:
            semantic_fidelity = SemanticFidelity.HIGH
        elif fidelity_score >= self.fidelity_threshold * 0.7:
            semantic_fidelity = SemanticFidelity.MEDIUM
        else:
            semantic_fidelity = SemanticFidelity.LOW
            
        fidelity_details = f"Semantic fidelity score: {fidelity_score:.2f}"
        
        # 4. Check entity alignment
        entity_alignment = self._check_entity_alignment(translated_text, extracted_entities)
        entity_score = sum(1 for v in entity_alignment.values() if v) / len(entity_alignment) if entity_alignment else 0
        
        if entity_score == 1.0:
            entity_details = "All entities preserved"
        elif entity_score >= 0.8:
            entity_details = f"Most entities preserved ({entity_score:.2f})"
        else:
            entity_details = f"Significant entity loss: only {entity_score:.2f} preserved"
            
        # 5. Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            bias_check, 
            pii_compliance, 
            fidelity_score, 
            entity_alignment
        )
        
        # 6. Create detailed audit report
        audit_report = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "pii_check": {
                "rating": pii_compliance,
                "details": pii_details,
                "found_pii": pii_found
            },
            "bias_check": {
                "rating": bias_check,
                "details": bias_details,
                "found_bias": bias_found
            },
            "semantic_fidelity": {
                "rating": semantic_fidelity,
                "score": fidelity_score,
                "details": fidelity_details
            },
            "entity_alignment": {
                "score": entity_score,
                "details": entity_details,
                "alignments": entity_alignment
            },
            "confidence_score": confidence_score
        }
        
        # 7. Save audit log if enabled
        if self.enable_audit_log:
            audit_log_path = os.path.join(self.audit_log_dir, f"audit_{session_id}.json")
            with open(audit_log_path, 'w') as f:
                json.dump(audit_report, f, indent=2)
            logger.info(f"Saved audit log to {audit_log_path}")
        
        # 8. Create and return audit summary
        audit_summary = {
            "bias_check": bias_check,
            "pii_compliance": pii_compliance,
            "semantic_fidelity": semantic_fidelity,
            "entity_alignment": all(entity_alignment.values()),
            "confidence_score": confidence_score,
            "audit_report": audit_report,
            "session_id": session_id
        }
        
        logger.info(f"Completed translation audit with confidence score: {confidence_score:.2f}")
        return audit_summary
    
    def audit_form_completion(self, 
                             original_form: Dict[str, Any],
                             filled_form: Dict[str, Any],
                             extracted_entities: Dict[str, Any],
                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Audit a form completion for accuracy and ethical compliance.
        
        Args:
            original_form: Original form template with field definitions
            filled_form: Completed form with user data
            extracted_entities: Entities extracted from conversation
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing audit results
        """
        logger.info("Starting form completion audit")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = hashlib.md5(f"{time.time()}-form".encode()).hexdigest()
        
        # 1. Check form completion rate
        total_fields = len(original_form.keys())
        filled_fields = sum(1 for k, v in filled_form.items() if v and str(v).strip())
        completion_rate = filled_fields / total_fields if total_fields > 0 else 0
        
        # 2. Check for PII in form
        all_values = " ".join(str(v) for v in filled_form.values() if v)
        pii_found = self._detect_pii(all_values)
        
        if pii_found:
            # Check if form fields should contain PII (based on field types)
            expected_pii = False
            for field in original_form.values():
                if isinstance(field, dict) and field.get("type") in ["ssn", "phone", "email", "dob"]:
                    expected_pii = True
                    break
            
            if expected_pii:
                pii_compliance = PIICompliance.PASS
                pii_details = "PII found in appropriate fields"
            else:
                pii_compliance = PIICompliance.WARN
                pii_details = f"Found unexpected PII: {list(pii_found.keys())}"
        else:
            # Check if PII was expected but not found
            missing_pii = False
            for field_key, field in original_form.items():
                if isinstance(field, dict) and field.get("type") in ["ssn", "phone", "email", "dob"]:
                    if field_key in filled_form and not filled_form[field_key]:
                        missing_pii = True
                        break
            
            if missing_pii:
                pii_compliance = PIICompliance.WARN
                pii_details = "Missing expected PII in designated fields"
            else:
                pii_compliance = PIICompliance.PASS
                pii_details = "No PII issues detected"
        
        # 3. Check entity alignment with form fields
        entity_alignment = {}
        for key, entity in extracted_entities.items():
            # Check if this entity appears in any form field
            found = False
            entity_str = str(entity)
            
            for field_value in filled_form.values():
                if isinstance(field_value, str) and entity_str.lower() in field_value.lower():
                    found = True
                    break
            
            entity_alignment[key] = found
        
        entity_score = sum(1 for v in entity_alignment.values() if v) / len(entity_alignment) if entity_alignment else 0
        
        # 4. Calculate confidence score for form filling
        confidence_score = 0.5 * completion_rate + 0.3 * entity_score
        
        # Add PII compliance component (20% weight)
        if pii_compliance == PIICompliance.PASS:
            confidence_score += 0.2
        elif pii_compliance == PIICompliance.WARN:
            confidence_score += 0.1
        
        confidence_score = min(1.0, max(0.0, confidence_score))
        
        # 5. Create audit report
        audit_report = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "form_completion": {
                "total_fields": total_fields,
                "filled_fields": filled_fields,
                "completion_rate": completion_rate
            },
            "pii_check": {
                "rating": pii_compliance,
                "details": pii_details,
                "found_pii": pii_found
            },
            "entity_alignment": {
                "score": entity_score,
                "alignments": entity_alignment
            },
            "confidence_score": confidence_score
        }
        
        # 6. Save audit log if enabled
        if self.enable_audit_log:
            audit_log_path = os.path.join(self.audit_log_dir, f"form_audit_{session_id}.json")
            with open(audit_log_path, 'w') as f:
                json.dump(audit_report, f, indent=2)
            logger.info(f"Saved form audit log to {audit_log_path}")
        
        # 7. Create and return audit summary
        audit_summary = {
            "completion_rate": completion_rate,
            "pii_compliance": pii_compliance,
            "entity_alignment": entity_score >= 0.8,
            "confidence_score": confidence_score,
            "audit_report": audit_report,
            "session_id": session_id
        }
        
        logger.info(f"Completed form audit with confidence score: {confidence_score:.2f}")
        return audit_summary
    
    def verify_compliance(self, text: str, rules: List[str]) -> Dict[str, Any]:
        """
        Verify compliance of text with given rules or regulations.
        
        Args:
            text: Text to verify
            rules: List of rules to check against
            
        Returns:
            Dictionary containing compliance verification results
        """
        # This is a simplified implementation
        # In a real system, this would use an LLM to check rule compliance
        
        compliance_results = {}
        overall_compliance = True
        
        for rule in rules:
            # Simulate rule checking (would use LLM in real implementation)
            import random
            compliant = random.random() > 0.2  # 80% chance of compliance
            
            compliance_results[rule] = compliant
            if not compliant:
                overall_compliance = False
        
        return {
            "overall_compliance": overall_compliance,
            "rule_compliance": compliance_results
        }
    
    def log_session_activity(self, 
                            session_id: str,
                            activity_type: str,
                            details: Dict[str, Any]) -> None:
        """
        Log session activity for audit purposes.
        
        Args:
            session_id: Session identifier
            activity_type: Type of activity (e.g., 'translation', 'form_completion')
            details: Dictionary of activity details
        """
        if not self.enable_audit_log:
            logger.info("Audit logging disabled, skipping activity log")
            return
            
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "details": details
        }
        
        # Save to session activity log
        log_path = os.path.join(self.audit_log_dir, f"activity_{session_id}.json")
        
        # Append to existing log if it exists
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    
                if not isinstance(log_data, list):
                    log_data = [log_data]
                    
                log_data.append(log_entry)
                
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to append to activity log: {str(e)}")
                # Create new log file if appending failed
                with open(log_path, 'w') as f:
                    json.dump([log_entry], f, indent=2)
        else:
            # Create new log file
            with open(log_path, 'w') as f:
                json.dump([log_entry], f, indent=2)
                
        logger.info(f"Logged {activity_type} activity for session {session_id}")