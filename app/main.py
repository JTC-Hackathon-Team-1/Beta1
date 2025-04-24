# app/main.py

import os
import sys
import logging
import time
from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from colorama import Fore, Style, init
import uvicorn

# Import routers
from app.routers.pipeline import router as pipeline_router

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua")

# ASCII Art Banner for startup
BANNER = f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗
║ {Fore.YELLOW}  ____                _     _                            {Fore.CYAN}               ║
║ {Fore.YELLOW} / ___|__ _ ___  __ _| |   (_)_ __   __ _ _   _  __ _    {Fore.CYAN}               ║
║ {Fore.YELLOW}| |   / _` / __|/ _` | |   | | '_ \\ / _` | | | |/ _` |   {Fore.CYAN}               ║
║ {Fore.YELLOW}| |__| (_| \\__ \\ (_| | |___| | | | | (_| | |_| | (_| |   {Fore.CYAN}               ║
║ {Fore.YELLOW} \\____\\__,_|___/\\__,_|_____|_|_| |_|\\__, |\\__,_|\\__,_|   {Fore.CYAN}               ║
║ {Fore.YELLOW}                                    |___/                {Fore.CYAN}               ║
║                                                                       ║
║ {Fore.GREEN}Language Processing & Translation Pipeline{Fore.CYAN}                              ║
╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

# Create FastAPI application
app = FastAPI(
    title="CasaLingua API",
    description="A language processing pipeline for translation and simplification",
    version="1.0.0",
)

# Attempt to import the enhanced components, but gracefully handle if they're not available
try:
    # Try to import admin router
    from app.routers.admin import router as admin_router
    has_admin = True
    logger.info(f"{Fore.GREEN}Admin module loaded successfully{Style.RESET_ALL}")
    
    # Try to import services
    try:
        from app.services.hardware_detection import HardwareDetector
        from app.services.model_registry import ModelRegistry
        
        # Initialize the hardware detector and model registry
        hardware_detector = HardwareDetector()
        model_registry = ModelRegistry()
        
        has_services = True
        logger.info(f"{Fore.GREEN}Advanced services loaded successfully{Style.RESET_ALL}")
    except ImportError:
        has_services = False
        logger.warning(f"{Fore.YELLOW}Hardware detection and model registry services not available{Style.RESET_ALL}")
        logger.warning(f"{Fore.YELLOW}Running with basic functionality{Style.RESET_ALL}")
except ImportError:
    has_admin = False
    has_services = False
    logger.warning(f"{Fore.YELLOW}Admin module not available - running in basic mode{Style.RESET_ALL}")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track request processing time and log request details.
    Adds X-Process-Time header to responses with the processing time in seconds.
    """
    # Generate a color for the path based on its hash (for visual distinction)
    path_hash = hash(request.url.path) % 6
    path_colors = [Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    path_color = path_colors[path_hash]
    
    # Log incoming request
    logger.info(f"{Fore.GREEN}→ {request.method}{Style.RESET_ALL} {path_color}{request.url.path}{Style.RESET_ALL}")
    
    # Process the request and track time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Color code the status code
    if response.status_code < 300:
        status_color = Fore.GREEN
    elif response.status_code < 400:
        status_color = Fore.YELLOW
    else:
        status_color = Fore.RED
    
    # Log the response
    logger.info(
        f"{Fore.GREEN}← {status_color}{response.status_code}{Style.RESET_ALL} "
        f"{path_color}{request.url.path}{Style.RESET_ALL} "
        f"({Fore.CYAN}{process_time:.3f}s{Style.RESET_ALL})"
    )
    
    return response


# Health check endpoint
@app.get("/", tags=["health"])
async def health():
    """
    Health check endpoint that returns basic server information.
    Use this to verify the API is running correctly.
    """
    # Basic version of health check
    health_info = {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
    
    # Add advanced information if available
    if has_services:
        # Get hardware info for the response
        hardware_info = hardware_detector.get_summary()
        
        health_info.update({
            "hardware": hardware_info["hardware"],
            "models": {
                "translation": model_registry.get_recommended_model("translation"),
                "simplification": model_registry.get_recommended_model("simplification")
            }
        })
    
    return health_info


# Mount the pipeline router
app.include_router(pipeline_router, prefix="/pipeline")

# Mount the admin router if available
if has_admin:
    app.include_router(admin_router)
    logger.info(f"{Fore.GREEN}Admin API endpoints mounted{Style.RESET_ALL}")

# Serve admin panel static files if they exist
admin_static_path = os.path.join(os.path.dirname(__file__), "static", "admin", "static")
if os.path.exists(admin_static_path):
    app.mount(
        "/static/admin",
        StaticFiles(directory=admin_static_path),
        name="admin_static",
    )
    
    # Serve admin/index.html on /admin/* if it exists
    admin_index_path = os.path.join(os.path.dirname(__file__), "static", "admin", "index.html")
    
    if os.path.exists(admin_index_path):
        @app.get("/admin/{full_path:path}", response_class=HTMLResponse)
        @app.get("/admin", response_class=HTMLResponse)
        async def serve_admin(full_path: str = ""):
            """
            Serve the admin panel single-page application for any path under /admin.
            The frontend routing will handle the specific admin paths.
            """
            logger.info(f"{Fore.MAGENTA}Serving admin panel{Style.RESET_ALL} (path: {full_path})")
            return FileResponse(admin_index_path)
        
        logger.info(f"{Fore.GREEN}Admin frontend mounted{Style.RESET_ALL}")


# Define startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Executed when the application starts.
    Initializes resources and displays startup information.
    """
    # Print the ASCII art banner
    print(BANNER)
    
    # Initialize advanced services if available
    if has_services:
        # Detect hardware capabilities
        logger.info(f"{Fore.GREEN}Detecting hardware capabilities...{Style.RESET_ALL}")
        hardware_info = hardware_detector.detect_hardware()
        
        # Determine optimal configuration
        logger.info(f"{Fore.GREEN}Determining optimal configuration...{Style.RESET_ALL}")
        config = hardware_detector.determine_optimal_configuration()
        
        # Apply the configuration
        logger.info(f"{Fore.GREEN}Applying configuration...{Style.RESET_ALL}")
        hardware_detector.apply_configuration()
        
        # Load model registry
        logger.info(f"{Fore.GREEN}Loading model registry...{Style.RESET_ALL}")
        model_registry.load_registry()
        
        # Preload recommended models if in production
        if os.getenv("ENVIRONMENT") == "production":
            logger.info(f"{Fore.GREEN}Preloading recommended models...{Style.RESET_ALL}")
            
            # Get recommended models
            translation_model = model_registry.get_recommended_model("translation")
            simplification_model = model_registry.get_recommended_model("simplification")
            
            # Preload the models
            if translation_model:
                model_registry.load_model("translation", translation_model)
            
            if simplification_model:
                model_registry.load_model("simplification", simplification_model)
        
        # Log hardware information
        cpu_info = hardware_info["cpu"]
        logger.info(f"{Fore.BLUE}CPU:{Style.RESET_ALL} {cpu_info['model']} ({cpu_info['cores']} cores)")
        
        memory_info = hardware_info["memory"]
        logger.info(f"{Fore.BLUE}Memory:{Style.RESET_ALL} {memory_info['total_gb']:.1f} GB")
        
        gpu_info = hardware_info["gpu"]
        if gpu_info["available"]:
            logger.info(f"{Fore.BLUE}GPU:{Style.RESET_ALL} {gpu_info['model']} ({gpu_info['memory_gb']:.1f} GB)")
        else:
            logger.info(f"{Fore.BLUE}GPU:{Style.RESET_ALL} Not available, using CPU")
        
        # Log configuration summary
        logger.info(f"{Fore.BLUE}Using GPU:{Style.RESET_ALL} {config['use_gpu']}")
        logger.info(f"{Fore.BLUE}Translation model:{Style.RESET_ALL} {config['models']['translation']}")
        logger.info(f"{Fore.BLUE}Simplification model:{Style.RESET_ALL} {config['models']['simplification']}")
        logger.info(f"{Fore.BLUE}Batch size:{Style.RESET_ALL} {config['batch_size']}")
    
    # Log server startup information
    logger.info(f"{Fore.GREEN}Starting CasaLingua API Server{Style.RESET_ALL}")
    logger.info(f"{Fore.BLUE}Environment:{Style.RESET_ALL} {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"{Fore.BLUE}Python version:{Style.RESET_ALL} {sys.version.split()[0]}")
    logger.info(f"{Fore.BLUE}Server time:{Style.RESET_ALL} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log available endpoints
    logger.info(f"{Fore.YELLOW}Available endpoints:{Style.RESET_ALL}")
    logger.info(f"  {Fore.GREEN}GET  /{Style.RESET_ALL} - Health check")
    logger.info(f"  {Fore.YELLOW}POST /pipeline{Style.RESET_ALL} - Language processing pipeline")
    
    if has_admin:
        logger.info(f"  {Fore.BLUE}GET  /admin{Style.RESET_ALL} - Admin panel")
        logger.info(f"  {Fore.MAGENTA}API  /api/admin/*{Style.RESET_ALL} - Admin API endpoints")
    
    # Log documentation URLs
    logger.info(f"{Fore.YELLOW}API Documentation:{Style.RESET_ALL}")
    logger.info(f"  {Fore.CYAN}Swagger UI:{Style.RESET_ALL} http://localhost:8000/docs")
    logger.info(f"  {Fore.CYAN}ReDoc:{Style.RESET_ALL} http://localhost:8000/redoc")
    
    logger.info(f"{Fore.GREEN}Server is ready to accept connections!{Style.RESET_ALL}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Executed when the application shuts down.
    Performs cleanup operations.
    """
    logger.info(f"{Fore.YELLOW}Shutting down CasaLingua API Server...{Style.RESET_ALL}")
    
    # Cleanup for advanced services
    if has_services:
        # Unload all models
        logger.info(f"{Fore.YELLOW}Unloading models...{Style.RESET_ALL}")
        loaded_models = model_registry.get_loaded_models()
        
        for model_key in list(loaded_models.keys()):
            model_type, model_id = model_key.split("/", 1)
            model_registry.unload_model(model_type, model_id)
        
        # Perform memory cleanup
        logger.info(f"{Fore.YELLOW}Cleaning up memory...{Style.RESET_ALL}")
        hardware_detector.manage_memory(clear_cache=True)
    
    logger.info(f"{Fore.GREEN}Shutdown complete. Goodbye!{Style.RESET_ALL}")


# Run the server directly if this script is executed
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Run Uvicorn server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info",
    )