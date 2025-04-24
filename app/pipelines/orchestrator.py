# app/pipelines/orchestrator.py

import spacy
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime

from colorama import Fore, Style, init
from app.schemas.pipeline import PipelineRequest, PipelineResponse, Entity
from app.llms.translation_llm import translate
from app.llms.simplify_llm import simplify_text
from app.utils.readability import reading_grade_level
from app.veracity.checker import check_veracity
from app.veracity.bias import check_bias, get_bias_details
from app.veracity.audit import log_pipeline_event, build_pipeline_audit_log

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure prettier, more verbose logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.orchestrator")

# Load spaCy English model once for efficiency (singleton pattern)
# This prevents having to reload the model for each request
_nlp = spacy.load("en_core_web_sm")

# Define pretty console formatting helpers
def format_header(text: str) -> str:
    """Create a visually distinct header for important pipeline steps"""
    return f"\n{Fore.YELLOW}{'='*50}\n{Fore.WHITE}{Style.BRIGHT}{text}\n{Fore.YELLOW}{'='*50}{Style.RESET_ALL}"

def format_step(step_num: int, description: str) -> str:
    """Format a pipeline step in a visually distinct way"""
    return f"{Fore.GREEN}[Step {step_num}]{Style.RESET_ALL} {Fore.CYAN}{description}{Style.RESET_ALL}"

def format_info(label: str, value: Any) -> str:
    """Format a key-value pair for logging"""
    return f"  {Fore.BLUE}{label}:{Style.RESET_ALL} {value}"

def format_time(seconds: float) -> str:
    """Format time duration with appropriate color based on duration"""
    if seconds < 0.1:
        color = Fore.GREEN
    elif seconds < 0.5:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    return f"{color}{seconds:.3f}s{Style.RESET_ALL}"


async def run_pipeline(req: PipelineRequest) -> PipelineResponse:
    """
    Orchestrates the full text processing pipeline for the CasaLingua application.
    
    Pipeline steps:
    1. Translation: Converts non-English text to English if needed
    2. Simplification: Reduces complexity for high reading grade material
    3. Analysis: Performs named entity recognition, veracity and bias checks
    4. Response construction: Builds a structured response with metadata
    
    This function acts as a directed acyclic graph (DAG) of NLP operations,
    tracking performance and providing rich diagnostic information.
    
    Args:
        req (PipelineRequest): Contains source/target languages and the text
                              to be processed
                              
    Returns:
        PipelineResponse: Processed text with analytical metadata
    """
    # Log the start of pipeline processing with request details
    start_time = time.time()
    logger.info(format_header(f"Starting pipeline for session: {req.session_id}"))
    logger.info(format_info("Source language", f"{Fore.YELLOW}{req.source_language}{Style.RESET_ALL}"))
    logger.info(format_info("Target language", f"{Fore.YELLOW}{req.target_language}{Style.RESET_ALL}"))
    logger.info(format_info("Input text length", f"{len(req.text)} chars"))
    logger.info(format_info("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Create an initial audit event for the request
    await log_pipeline_event(
        session_id=req.session_id,
        event_type="pipeline_request",
        data={
            "source_language": req.source_language,
            "target_language": req.target_language, 
            "text_length": len(req.text),
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # Initialize our working text variable - will be transformed through pipeline
    text = req.text
    
    # =========================================================================
    # CRITICAL FIX: Add specific handling for "legalese" to "plain_english"
    # =========================================================================
    if req.source_language.lower() == "legalese" and req.target_language.lower() == "plain_english":
        logger.info(format_step(1, "LEGALESE DETECTED - Routing to specialized simplification model"))
        step_start = time.time()
        
        # Use simplification model directly without translation step
        final_text = await simplify_text(text, target_grade=9)
        
        step_duration = time.time() - step_start
        logger.info(f"  {Fore.GREEN}✓ Legalese simplified{Style.RESET_ALL} in {format_time(step_duration)}")
        logger.info(format_info("Simplified text sample", f"\"{final_text[:50]}...\""))
        
        # Create an audit event for legalese simplification
        await log_pipeline_event(
            session_id=req.session_id,
            event_type="legalese_simplification",
            data={
                "input_length": len(text),
                "output_length": len(final_text),
                "duration_seconds": step_duration
            }
        )
        
        # Calculate grade level for audit log
        grade = reading_grade_level(final_text)
    else:
        # Normal pipeline processing for standard language pairs
        # Step 1: Translation if needed
        logger.info(format_step(1, "Translation Layer"))
        step_start = time.time()
        
        if req.source_language.lower() not in ("en", "english"):
            logger.info(f"  {Fore.YELLOW}Source not English - translating to English first{Style.RESET_ALL}")
            translation = await translate(text, req.source_language, "en")
            step_duration = time.time() - step_start
            logger.info(f"  {Fore.GREEN}✓ Translated{Style.RESET_ALL} in {format_time(step_duration)}")
            logger.info(format_info("Translation sample", f"\"{translation[:50]}...\""))
            
            # Create an audit event for translation
            await log_pipeline_event(
                session_id=req.session_id,
                event_type="translation",
                data={
                    "source_language": req.source_language,
                    "target_language": "en",
                    "input_length": len(text),
                    "output_length": len(translation),
                    "duration_seconds": step_duration
                }
            )
        else:
            translation = text
            logger.info(f"  {Fore.GREEN}✓ Source already English - no translation needed{Style.RESET_ALL}")
        
        # Step 2: Simplification if reading grade is too high
        logger.info(format_step(2, "Simplification Layer"))
        step_start = time.time()
        
        # Calculate reading grade level for the current text
        grade = reading_grade_level(translation)
        logger.info(format_info("Reading grade level", f"{Fore.MAGENTA}{grade:.1f}{Style.RESET_ALL}"))
        
        if grade > 9:
            logger.info(f"  {Fore.YELLOW}Reading grade > 9 - simplifying{Style.RESET_ALL}")
            final_text = await simplify_text(translation, target_grade=9)
            step_duration = time.time() - step_start
            logger.info(f"  {Fore.GREEN}✓ Simplified{Style.RESET_ALL} in {format_time(step_duration)}")
            logger.info(format_info("Simplified text sample", f"\"{final_text[:50]}...\""))
            
            # Create an audit event for simplification
            await log_pipeline_event(
                session_id=req.session_id,
                event_type="simplification",
                data={
                    "original_grade": grade,
                    "target_grade": 9,
                    "input_length": len(translation),
                    "output_length": len(final_text),
                    "duration_seconds": step_duration
                }
            )
        else:
            final_text = translation
            logger.info(f"  {Fore.GREEN}✓ Text already simple enough (grade <= 9){Style.RESET_ALL}")
    
    # Step 3: Named Entity Recognition (for all pipeline paths)
    logger.info(format_step(3, "Entity Recognition"))
    step_start = time.time()
    
    # Process text through spaCy NLP pipeline
    doc = _nlp(final_text)
    
    # FIX: Changed from doc.ents to create Entity objects with the correct attributes
    # The issue was that we were using ent.label_ but our Entity class uses 'label'
    entities: List[Entity] = [
        Entity(text=ent.text, label=ent.label_) for ent in doc.ents
    ]
    
    step_duration = time.time() - step_start
    logger.info(f"  {Fore.GREEN}✓ Entity Recognition complete{Style.RESET_ALL} in {format_time(step_duration)}")
    
    if entities:
        logger.info(format_info("Entities found", f"{len(entities)}"))
        # Log first few entities as examples (up to 3)
        for i, entity in enumerate(entities[:3]):
            # FIX: Changed from entity.label_ to entity.label to match our schema
            logger.info(f"    {Fore.CYAN}{entity.text}{Style.RESET_ALL} ({Fore.MAGENTA}{entity.label}{Style.RESET_ALL})")
        if len(entities) > 3:
            logger.info(f"    {Fore.BLUE}... and {len(entities) - 3} more{Style.RESET_ALL}")
        
        # Create an audit event for entity recognition
        # FIX: Use the correct attribute name in the audit data too
        entity_types = {}
        for e in entities:
            if e.label not in entity_types:
                entity_types[e.label] = 0
            entity_types[e.label] += 1
            
        await log_pipeline_event(
            session_id=req.session_id,
            event_type="entity_recognition",
            data={
                "entity_count": len(entities),
                "entity_types": entity_types,
                "duration_seconds": step_duration
            }
        )
    else:
        logger.info(f"  {Fore.YELLOW}No entities detected{Style.RESET_ALL}")
    
    # Step 4: Veracity & Bias checks
    logger.info(format_step(4, "Semantic Analysis"))
    step_start = time.time()
    
    # Run parallel checks for veracity and bias
    veracity_score = await check_veracity(final_text)
    bias_score = await check_bias(final_text)
    
    # Get detailed bias information for audit purposes
    bias_details = await get_bias_details(final_text)
    
    step_duration = time.time() - step_start
    logger.info(f"  {Fore.GREEN}✓ Semantic Analysis complete{Style.RESET_ALL} in {format_time(step_duration)}")
    
    # Color code veracity score
    if veracity_score > 0.7:
        veracity_color = Fore.GREEN
    elif veracity_score > 0.4:
        veracity_color = Fore.YELLOW
    else:
        veracity_color = Fore.RED
    
    # Color code bias score
    if bias_score < 0.3:
        bias_color = Fore.GREEN
    elif bias_score < 0.6:
        bias_color = Fore.YELLOW
    else:
        bias_color = Fore.RED
    
    logger.info(format_info("Veracity score", f"{veracity_color}{veracity_score:.2f}{Style.RESET_ALL}"))
    logger.info(format_info("Bias score", f"{bias_color}{bias_score:.2f}{Style.RESET_ALL}"))
    
    # Create an audit event for semantic analysis
    await log_pipeline_event(
        session_id=req.session_id,
        event_type="semantic_analysis",
        data={
            "veracity_score": veracity_score,
            "bias_score": bias_score,
            "bias_categories": bias_details["categories"],
            "duration_seconds": step_duration
        }
    )
    
    # Step 5: Build comprehensive audit log
    total_duration = time.time() - start_time
    
    # Build audit log with performance metrics
    audit_log = await build_pipeline_audit_log(
        translation_length=float(len(final_text)),
        entity_count=float(len(entities)),
        reading_grade=grade,
        processing_time_seconds=total_duration
    )
    
    # Construct the final response object
    response = PipelineResponse(
        translation=final_text,
        entities=entities,
        veracity_score=veracity_score,
        bias_score=bias_score,
        audit_log=audit_log,
    )
    
    # Create a final audit event for the completed pipeline
    await log_pipeline_event(
        session_id=req.session_id,
        event_type="pipeline_complete",
        data={
            "total_duration_seconds": total_duration,
            "input_length": len(req.text),
            "output_length": len(final_text),
            "entity_count": len(entities),
            "reading_grade": grade,
            "veracity_score": veracity_score,
            "bias_score": bias_score
        }
    )
    
    # Log pipeline completion
    logger.info(format_header(f"Pipeline complete in {format_time(total_duration)}"))
    
    return response