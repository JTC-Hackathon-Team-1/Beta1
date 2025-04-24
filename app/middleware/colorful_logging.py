# app/middleware/colorful_logging.py

import json
import time
from datetime import datetime
import logging
from typing import Dict, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.api")

class ColorfulLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds colorful request and response logging to all API endpoints.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log details in a colorful format, and then pass to the next
        middleware or route handler.
        """
        # Generate a unique request ID
        request_id = f"req-{int(time.time() * 1000)}"
        
        # Start timer for request duration
        start_time = time.time()
        
        # Log the request in a colorful format
        await self._log_request(request, request_id)
        
        # Process the request
        try:
            # Call the next middleware or endpoint handler
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log the response in a colorful format
            self._log_response(request, response, duration, request_id)
            
            return response
        except Exception as e:
            # Log exceptions in a colorful format
            duration = time.time() - start_time
            self._log_exception(request, e, duration, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str) -> None:
        """
        Log the incoming request in a colorful format.
        """
        # Print a colorful header
        logger.info(f"\n{Fore.YELLOW}{'═' * 60}{Style.RESET_ALL}")
        logger.info(f"{Fore.GREEN}➤ INCOMING REQUEST {Fore.CYAN}[{request_id}]{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'─' * 60}{Style.RESET_ALL}")
        
        # Log basic request info
        logger.info(f"{Fore.MAGENTA}Method:{Style.RESET_ALL} {Fore.CYAN}{request.method}{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}URL:{Style.RESET_ALL} {Fore.CYAN}{request.url.path}{Style.RESET_ALL}")
        
        # We'll skip body extraction as it's tricky and can cause issues
        if request.headers.get('content-type') == 'application/json':
            logger.info(f"{Fore.MAGENTA}Content-Type:{Style.RESET_ALL} {Fore.CYAN}application/json{Style.RESET_ALL}")
    
    def _log_response(self, request: Request, response: Response, duration: float, request_id: str) -> None:
        """
        Log the outgoing response in a colorful format.
        """
        # Determine color based on status code
        if response.status_code < 300:
            status_color = Fore.GREEN
        elif response.status_code < 400:
            status_color = Fore.YELLOW
        else:
            status_color = Fore.RED
        
        # Format duration with appropriate color
        if duration < 0.1:
            duration_color = Fore.GREEN
        elif duration < 0.5:
            duration_color = Fore.YELLOW
        else:
            duration_color = Fore.RED
        
        # Print a colorful response header
        logger.info(f"{Fore.YELLOW}{'─' * 60}{Style.RESET_ALL}")
        logger.info(f"{Fore.BLUE}◄ OUTGOING RESPONSE {Fore.CYAN}[{request_id}]{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}{'─' * 60}{Style.RESET_ALL}")
        
        # Log basic response info
        logger.info(f"{Fore.MAGENTA}Status:{Style.RESET_ALL} {status_color}{response.status_code}{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}Duration:{Style.RESET_ALL} {duration_color}{duration:.4f}s{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}URL:{Style.RESET_ALL} {Fore.CYAN}{request.method} {request.url.path}{Style.RESET_ALL}")
        
        # Print footer
        logger.info(f"{Fore.YELLOW}{'═' * 60}{Style.RESET_ALL}\n")
    
    def _log_exception(self, request: Request, exception: Exception, duration: float, request_id: str) -> None:
        """
        Log exceptions in a colorful format.
        """
        logger.info(f"{Fore.RED}{'!' * 60}{Style.RESET_ALL}")
        logger.info(f"{Fore.RED}✕ EXCEPTION IN REQUEST {Fore.CYAN}[{request_id}]{Style.RESET_ALL}")
        logger.info(f"{Fore.RED}{'!' * 60}{Style.RESET_ALL}")
        
        logger.info(f"{Fore.MAGENTA}URL:{Style.RESET_ALL} {Fore.CYAN}{request.method} {request.url.path}{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}Duration:{Style.RESET_ALL} {Fore.YELLOW}{duration:.4f}s{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}Exception:{Style.RESET_ALL} {Fore.RED}{type(exception).__name__}{Style.RESET_ALL}")
        logger.info(f"{Fore.MAGENTA}Message:{Style.RESET_ALL} {Fore.RED}{str(exception)}{Style.RESET_ALL}")
        
        logger.info(f"{Fore.RED}{'!' * 60}{Style.RESET_ALL}\n")