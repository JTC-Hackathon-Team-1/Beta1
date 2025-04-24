# app/routers/pipeline.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import json
from pydantic import ValidationError
import logging
from colorama import Fore, Style, init
import time
from datetime import datetime
import uuid

from app.schemas.pipeline import PipelineRequest, PipelineResponse
from app.pipelines.orchestrator import run_pipeline

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.api")

# Create router with appropriate tags for OpenAPI docs
router = APIRouter(
    tags=["translation"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


# Custom JSON encoder for pretty terminal output of API requests/responses
class PrettyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that produces colorful and formatted JSON strings
    for better readability in terminal output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation = 0
        
    def encode(self, obj):
        if isinstance(obj, dict):
            self.indentation += 4
            items = []
            for key, value in obj.items():
                # Use colors for different parts of the JSON
                key_str = f"{Fore.GREEN}\"{key}\"{Style.RESET_ALL}"
                
                # Format different value types differently
                if isinstance(value, str):
                    value_str = f"{Fore.YELLOW}\"{value}\"{Style.RESET_ALL}"
                elif isinstance(value, (int, float)):
                    value_str = f"{Fore.CYAN}{value}{Style.RESET_ALL}"
                elif isinstance(value, bool):
                    value_str = f"{Fore.MAGENTA}{str(value).lower()}{Style.RESET_ALL}"
                elif value is None:
                    value_str = f"{Fore.RED}null{Style.RESET_ALL}"
                else:
                    value_str = self.encode(value)
                    
                items.append(f"{' ' * self.indentation}{key_str}: {value_str}")
            self.indentation -= 4
            return "{\n" + ",\n".join(items) + "\n" + (' ' * self.indentation) + "}"
        elif isinstance(obj, list):
            self.indentation += 4
            items = [f"{' ' * self.indentation}{self.encode(item)}" for item in obj]
            self.indentation -= 4
            return "[\n" + ",\n".join(items) + "\n" + (' ' * self.indentation) + "]"
        return super().encode(obj)


def log_pretty_json(prefix: str, data: dict, truncate_text: bool = True):
    """
    Log a dictionary as pretty, colorized JSON with optional text truncation.
    
    Args:
        prefix: Description of what's being logged
        data: Dictionary to log as pretty JSON
        truncate_text: Whether to truncate long text fields
    """
    # Make a copy to avoid modifying the original
    data_copy = data.copy() if isinstance(data, dict) else data
    
    # Truncate long text fields for cleaner logs
    if truncate_text and isinstance(data_copy, dict):
        for key, value in data_copy.items():
            if isinstance(value, str) and len(value) > 100:
                data_copy[key] = value[:100] + "..."
    
    # Format and log with a pretty header
    logger.info(f"\n{Fore.CYAN}{'='*20} {prefix} {'='*20}{Style.RESET_ALL}")
    
    # Use our custom encoder for pretty printing
    try:
        logger.info(PrettyJSONEncoder().encode(data_copy))
    except (TypeError, ValueError):
        # Fallback if custom encoder fails
        logger.info(json.dumps(data_copy, indent=2, default=str))
    
    logger.info(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")


@router.post("", response_model=PipelineResponse)
async def pipeline_endpoint(request: Request):
    """
    Main endpoint for the CasaLingua translation pipeline.
    
    This endpoint processes the incoming request through:
    1. Validation of the request schema
    2. Orchestration of the translation/simplification pipeline
    3. Generation of a structured, colorful response
    
    The pipeline handles various language pairs, including special
    cases like "legalese" to "plain_english" which trigger the
    text simplification model instead of translation.
    
    Returns:
        JSONResponse: Processed text with analytical metadata
    """
    # Start timing for performance logging
    start_time = time.time()
    
    # Generate unique request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    
    # Log the incoming request with colorful formatting
    logger.info(f"\n{Fore.YELLOW}{'='*20} INCOMING REQUEST {request_id} {'='*20}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Timestamp:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{Fore.CYAN}Client IP:{Style.RESET_ALL} {request.client.host if request.client else 'unknown'}")
    
    # Parse and validate the incoming JSON request
    try:
        # Get raw request body first for logging
        body = await request.json()
        log_pretty_json("Request Body", body)
        
        # Validate and parse the request
        pipe_request = PipelineRequest(**body)
        
        # Special highlighting for legalese to plain_english
        if (pipe_request.source_language.lower() == "legalese" and 
            pipe_request.target_language.lower() == "plain_english"):
            logger.info(f"{Fore.MAGENTA}Special Case Detected:{Style.RESET_ALL} " +
                       f"Legalese → Plain English")
        
        # Process the request through our pipeline
        logger.info(f"{Fore.CYAN}Starting pipeline processing...{Style.RESET_ALL}")
        response = await run_pipeline(pipe_request)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"{Fore.GREEN}Request {request_id} completed in {processing_time:.3f}s{Style.RESET_ALL}")
        
        # Log the response (truncated for readability)
        response_dict = response.model_dump()
        log_pretty_json("Response", response_dict)
        
        # Return the response
        return JSONResponse(
            content=response_dict,
            status_code=200,
        )
        
    except json.JSONDecodeError as e:
        # Handle malformed JSON
        error_msg = f"Invalid JSON: {str(e)}"
        logger.error(f"{Fore.RED}JSON Parse Error:{Style.RESET_ALL} {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"detail": [{"type": "json_invalid", "loc": ["body", e.pos], "msg": "JSON decode error"}]},
        )
    except ValidationError as e:
        # Handle schema validation errors
        logger.error(f"{Fore.RED}Validation Error:{Style.RESET_ALL} {str(e)}")
        return JSONResponse(status_code=422, content={"detail": e.errors()})
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"{Fore.RED}Pipeline Error:{Style.RESET_ALL} {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing request: {str(e)}"},
        )