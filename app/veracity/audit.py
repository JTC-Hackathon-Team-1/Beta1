# app/veracity/audit.py

import logging
import time
import json
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from colorama import Fore, Style, init
import uuid

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.audit")

# Define the audit log directory (create if it doesn't exist)
AUDIT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "audit")
os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

# Global settings
AUDIT_ENABLED = True  # Can be toggled for testing or development
AUDIT_PERSISTENCE = True  # Whether to save audit logs to disk


async def log_pipeline_event(
    session_id: str,
    event_type: str,
    data: Dict[str, Any],
    save_to_disk: bool = True
) -> Dict[str, Any]:
    """
    Records an audit event for the pipeline process.
    
    This function creates a structured audit record for various pipeline
    events such as translation requests, simplification operations,
    or entity recognition. These logs are essential for:
    1. Accountability and compliance
    2. Performance monitoring
    3. Error tracking and debugging
    4. Usage analytics
    
    Args:
        session_id: Unique identifier for the user session
        event_type: Type of event (e.g., "translation", "simplification")
        data: Event-specific data to record
        save_to_disk: Whether to persist the audit log to disk
        
    Returns:
        Dict[str, Any]: The complete audit record that was created
    """
    # Skip if auditing is disabled
    if not AUDIT_ENABLED:
        return {}
    
    # Start timing
    start_time = time.time()
    
    # Log the audit initiation
    logger.info(f"{Fore.YELLOW}Creating audit record{Style.RESET_ALL} for {Fore.CYAN}{event_type}{Style.RESET_ALL}")
    logger.info(f"  {Fore.BLUE}Session ID:{Style.RESET_ALL} {session_id}")
    
    # Create a unique ID for this audit record
    audit_id = str(uuid.uuid4())
    
    # Build the complete audit record
    audit_record = {
        "audit_id": audit_id,
        "session_id": session_id,
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    # Simulate some processing time
    await asyncio.sleep(0.01)
    
    # Save to disk if requested and persistence is enabled
    if save_to_disk and AUDIT_PERSISTENCE:
        await _save_audit_record(audit_record)
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Log completion
    logger.info(f"{Fore.GREEN}✓ Audit record created{Style.RESET_ALL} in {elapsed_time:.3f}s")
    logger.info(f"  {Fore.BLUE}Audit ID:{Style.RESET_ALL} {audit_id}")
    
    return audit_record


async def build_pipeline_audit_log(
    translation_length: float,
    entity_count: float,
    reading_grade: float,
    processing_time_seconds: Optional[float] = None
) -> Dict[str, float]:
    """
    Constructs a standardized audit log for a pipeline process.
    
    This function aggregates key metrics from the pipeline process into
    a standardized format for consistent recording and analysis.
    
    Args:
        translation_length: Length of the processed text
        entity_count: Number of entities detected
        reading_grade: Reading grade level of the text
        processing_time_seconds: Total processing time in seconds (optional)
        
    Returns:
        Dict[str, float]: Standardized audit log with metrics
    """
    # Start timing if processing_time_seconds wasn't provided
    if processing_time_seconds is None:
        processing_time_seconds = 0.0
    
    # Build the audit log
    audit_log = {
        "translation_length": float(translation_length),
        "entity_count": float(entity_count),
        "reading_grade": float(reading_grade),
        "processing_time_seconds": float(processing_time_seconds),
        "timestamp": time.time()
    }
    
    # Log the audit log creation
    logger.info(f"{Fore.GREEN}Built pipeline audit log{Style.RESET_ALL}")
    logger.info(f"  {Fore.BLUE}Metrics recorded:{Style.RESET_ALL} {len(audit_log)} items")
    
    # Format each metric with appropriate colors
    for key, value in audit_log.items():
        if key == "timestamp":
            continue
            
        if key == "reading_grade":
            # Color code reading grade (lower is generally better)
            if value < 8:
                color = Fore.GREEN
            elif value < 12:
                color = Fore.YELLOW
            else:
                color = Fore.RED
        elif key == "processing_time_seconds":
            # Color code processing time (lower is better)
            if value < 0.5:
                color = Fore.GREEN
            elif value < 1.0:
                color = Fore.YELLOW
            else:
                color = Fore.RED
        else:
            # Default color for other metrics
            color = Fore.CYAN
            
        logger.info(f"    {Fore.BLUE}{key}:{Style.RESET_ALL} {color}{value}{Style.RESET_ALL}")
    
    return audit_log


async def _save_audit_record(audit_record: Dict[str, Any]) -> None:
    """
    Saves an audit record to disk as a JSON file.
    
    This is a private helper function for persistent audit logging.
    
    Args:
        audit_record: The audit record to save
    """
    # Create a filename based on the audit ID
    filename = f"{audit_record['audit_id']}.json"
    filepath = os.path.join(AUDIT_LOG_DIR, filename)
    
    try:
        # Write the audit record to disk
        with open(filepath, 'w') as f:
            json.dump(audit_record, f, indent=2)
        logger.info(f"  {Fore.GREEN}✓ Saved audit record to disk{Style.RESET_ALL}: {filename}")
    except Exception as e:
        # Log any errors but don't raise exceptions
        logger.error(f"  {Fore.RED}Failed to save audit record{Style.RESET_ALL}: {str(e)}")


async def get_session_audit_records(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all audit records for a specific session.
    
    This is useful for retrospective analysis of a user's interactions
    with the system, for debugging or analytics purposes.
    
    Args:
        session_id: The session ID to retrieve records for
        
    Returns:
        List[Dict[str, Any]]: All audit records for the session
    """
    records = []
    
    logger.info(f"{Fore.YELLOW}Retrieving audit records{Style.RESET_ALL} for session {Fore.CYAN}{session_id}{Style.RESET_ALL}")
    
    # This is a simplified implementation for the stub
    # In a real system, this would query a database or scan files more efficiently
    try:
        # Scan the audit log directory for matching records
        for filename in os.listdir(AUDIT_LOG_DIR):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(AUDIT_LOG_DIR, filename)
            
            try:
                with open(filepath, 'r') as f:
                    record = json.load(f)
                    
                if record.get('session_id') == session_id:
                    records.append(record)
            except Exception as e:
                logger.error(f"  {Fore.RED}Error reading audit file {filename}{Style.RESET_ALL}: {str(e)}")
    except Exception as e:
        logger.error(f"  {Fore.RED}Error scanning audit directory{Style.RESET_ALL}: {str(e)}")
    
    logger.info(f"{Fore.GREEN}✓ Retrieved {len(records)} audit records{Style.RESET_ALL} for session {session_id}")
    
    return records


# For demonstration/testing purposes
if __name__ == "__main__":
    async def test_audit_module():
        """Test function to demonstrate the audit module."""
        print(f"\n{Fore.YELLOW}Testing Audit Module{Style.RESET_ALL}\n")
        
        # Create a test session ID
        test_session_id = f"test-session-{int(time.time())}"
        
        # Test creating an audit record
        print(f"{Fore.CYAN}Creating test audit record...{Style.RESET_ALL}")
        test_data = {
            "source_language": "en",
            "target_language": "plain_english",
            "input_length": 150,
            "output_length": 125,
        }
        
        record = await log_pipeline_event(
            session_id=test_session_id,
            event_type="translation",
            data=test_data
        )
        
        print(f"{Fore.GREEN}✓ Created audit record:{Style.RESET_ALL} {record['audit_id']}")
        
        # Test building an audit log
        print(f"\n{Fore.CYAN}Building test audit log...{Style.RESET_ALL}")
        audit_log = await build_pipeline_audit_log(
            translation_length=125.0,
            entity_count=3.0,
            reading_grade=8.5,
            processing_time_seconds=0.45
        )
        
        print(f"{Fore.GREEN}✓ Built audit log with {len(audit_log)} metrics{Style.RESET_ALL}")
        
        # Test retrieving session records
        print(f"\n{Fore.CYAN}Retrieving session records...{Style.RESET_ALL}")
        session_records = await get_session_audit_records(test_session_id)
        
        print(f"{Fore.GREEN}✓ Retrieved {len(session_records)} records for session {test_session_id}{Style.RESET_ALL}")
        
    # Run the test
    asyncio.run(test_audit_module())