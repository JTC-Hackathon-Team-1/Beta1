# app/routers/admin.py

import os
import platform
import psutil
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Body, Path
from pydantic import BaseModel
import torch
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.admin")

# Create router
router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    responses={404: {"description": "Not found"}},
)

# Models for the API
class SystemInfo(BaseModel):
    """System information including CPU, memory, GPU and environment details"""
    cpuLoad: float
    memoryUsage: float
    memoryTotal: float
    gpuAvailable: bool
    gpuMemoryUsage: Optional[float] = None
    gpuMemoryTotal: Optional[float] = None
    activeModel: str
    translationModel: str
    uptime: int
    requestsProcessed: int
    environment: str
    detectedEnvironment: Dict[str, Any]

class ModelInfo(BaseModel):
    """Information about a specific model"""
    id: str
    name: str
    status: str
    size: str
    description: str
    dateAdded: str
    lastUsed: str
    performance: Dict[str, Any]
    languages: Optional[int] = None

class ModelList(BaseModel):
    """List of models and the currently active model"""
    models: List[ModelInfo]
    activeModel: str

class LogEntry(BaseModel):
    """Individual log entry for veracity or bias checks"""
    id: str
    timestamp: str
    sessionId: str
    text: str
    score: float
    status: str
    details: Dict[str, Any]

class LogList(BaseModel):
    """Paginated list of logs with filtering information"""
    logs: List[LogEntry]
    totalLogs: int
    totalPages: int
    currentPage: int
    filter: str
    search: Optional[str]
    sortBy: str
    sortDirection: str

class TuningRequest(BaseModel):
    """Request to fine-tune a model"""
    modelId: str
    learningRate: float
    epochs: int
    batchSize: int
    trainingDataPath: str
    evaluationDataPath: str
    saveModelPath: str

class TuningStatus(BaseModel):
    """Status of a model tuning job"""
    modelId: str
    inProgress: bool
    progress: float
    logs: List[Dict[str, str]]

# Global state for storing active tuning jobs
active_tuning_jobs: Dict[str, TuningStatus] = {}

# Helper functions for getting system information
def get_system_info() -> SystemInfo:
    """
    Get current system information including CPU, memory, GPU, and environment details.
    """
    # Get CPU information
    cpu_load = psutil.cpu_percent(interval=0.1)
    
    # Get memory information
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)  # Convert bytes to GB
    memory_used_gb = memory.used / (1024 ** 3)
    
    # Check for GPU availability and get GPU info
    gpu_available = torch.cuda.is_available()
    gpu_memory_usage = None
    gpu_memory_total = None
    gpu_info = None
    
    if gpu_available:
        try:
            # Get current device
            current_device = torch.cuda.current_device()
            
            # Get GPU name
            gpu_info = torch.cuda.get_device_name(current_device)
            
            # Get GPU memory in GB
            gpu_memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            
            # Get GPU memory usage (this is an approximation)
            gpu_memory_allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 3)
            gpu_memory_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)
            gpu_memory_usage = max(gpu_memory_allocated, gpu_memory_reserved)
        except Exception as e:
            logger.error(f"{Fore.RED}Error getting GPU information: {e}{Style.RESET_ALL}")
            gpu_available = False
    
    # Get environment variables
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Get uptime (simulated for this example)
    # In a real implementation, you'd use platform-specific methods to get the actual uptime
    uptime_seconds = 86400  # 24 hours
    
    # Get detected environment information
    detected_environment = {
        "os": f"{platform.system()} {platform.release()}",
        "pythonVersion": platform.python_version(),
        "gpuInfo": gpu_info,
        "cpuInfo": platform.processor() or "Unknown",
        "memoryInfo": f"{memory_total_gb:.1f}GB"
    }
    
    # Get model information (simulated)
    active_model = "t5-small"
    translation_model = "m2m100_418M"
    
    # Get request count (simulated)
    requests_processed = 1000
    
    return SystemInfo(
        cpuLoad=cpu_load,
        memoryUsage=memory_used_gb,
        memoryTotal=memory_total_gb,
        gpuAvailable=gpu_available,
        gpuMemoryUsage=gpu_memory_usage,
        gpuMemoryTotal=gpu_memory_total,
        activeModel=active_model,
        translationModel=translation_model,
        uptime=uptime_seconds,
        requestsProcessed=requests_processed,
        environment=environment,
        detectedEnvironment=detected_environment
    )

# Routes for system information
@router.get("/system-info", response_model=SystemInfo, summary="Get system information")
async def get_system_information():
    """
    Get current system information including:
    - CPU and memory usage
    - GPU availability and usage
    - Environment details
    - Active models
    - Uptime and request stats
    """
    try:
        logger.info(f"{Fore.GREEN}Getting system information{Style.RESET_ALL}")
        system_info = get_system_info()
        return system_info
    except Exception as e:
        logger.error(f"{Fore.RED}Error getting system information: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error getting system information: {str(e)}")

# Routes for model management
@router.get("/models/{model_type}", response_model=ModelList, summary="Get models list")
async def get_models(model_type: str = Path(..., description="Model type (translation or simplification)")):
    """
    Get list of available models for a specific type.
    
    - For translation models, returns available languages and other metadata
    - For simplification models, returns performance metrics and other details
    - Includes information about active models
    """
    try:
        logger.info(f"{Fore.GREEN}Getting {model_type} models{Style.RESET_ALL}")
        
        if model_type not in ["translation", "simplification"]:
            raise HTTPException(status_code=400, detail="Model type must be 'translation' or 'simplification'")
        
        # Simulated model data
        if model_type == "translation":
            models = [
                ModelInfo(
                    id="m2m100_418M",
                    name="M2M100 (418M)",
                    status="active",
                    size="418 MB",
                    description="Facebook/Meta multilingual translation model",
                    languages=100,
                    dateAdded="2023-01-15",
                    lastUsed="2023-09-30",
                    performance={"speed": 4.2, "accuracy": 8.1, "memoryUsage": "350 MB"}
                ),
                ModelInfo(
                    id="m2m100_1.2B",
                    name="M2M100 (1.2B)",
                    status="available",
                    size="1.2 GB",
                    description="Larger version of M2M100 with improved quality",
                    languages=100,
                    dateAdded="2023-02-20",
                    lastUsed="2023-07-12",
                    performance={"speed": 2.8, "accuracy": 8.9, "memoryUsage": "1.1 GB"}
                ),
                ModelInfo(
                    id="nllb_1.3B",
                    name="NLLB (1.3B)",
                    status="available",
                    size="1.3 GB",
                    description="Meta 'No Language Left Behind' model",
                    languages=200,
                    dateAdded="2023-03-05",
                    lastUsed="2023-08-22",
                    performance={"speed": 2.5, "accuracy": 9.2, "memoryUsage": "1.2 GB"}
                )
            ]
            active_model = "m2m100_418M"
        else:  # simplification
            models = [
                ModelInfo(
                    id="t5-small",
                    name="T5 Small",
                    status="active",
                    size="242 MB",
                    description="Text-to-Text Transfer Transformer (small version)",
                    dateAdded="2023-01-10",
                    lastUsed="2023-09-28",
                    performance={"speed": 5.1, "accuracy": 7.8, "memoryUsage": "220 MB"}
                ),
                ModelInfo(
                    id="t5-base",
                    name="T5 Base",
                    status="available",
                    size="892 MB",
                    description="T5 base model with better performance",
                    dateAdded="2023-02-15",
                    lastUsed="2023-06-30",
                    performance={"speed": 3.4, "accuracy": 8.5, "memoryUsage": "850 MB"}
                ),
                ModelInfo(
                    id="byt5-small",
                    name="ByT5 Small",
                    status="available",
                    size="300 MB",
                    description="Byte-level T5 model that works at the byte level",
                    dateAdded="2023-04-10",
                    lastUsed="2023-05-15",
                    performance={"speed": 4.8, "accuracy": 8.2, "memoryUsage": "290 MB"}
                ),
                ModelInfo(
                    id="legal-t5",
                    name="Legal-T5 (Custom)",
                    status="available",
                    size="450 MB",
                    description="Custom fine-tuned T5 model for legal document simplification",
                    dateAdded="2023-07-22",
                    lastUsed="2023-09-01",
                    performance={"speed": 3.9, "accuracy": 8.7, "memoryUsage": "420 MB"}
                )
            ]
            active_model = "t5-small"
        
        logger.info(f"{Fore.GREEN}Retrieved {len(models)} {model_type} models{Style.RESET_ALL}")
        return ModelList(models=models, activeModel=active_model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{Fore.RED}Error getting {model_type} models: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@router.post("/models/{model_type}/activate", summary="Activate a model")
async def activate_model(
    model_type: str = Path(..., description="Model type (translation or simplification)"),
    model_id: str = Body(..., embed=True, description="ID of the model to activate")
):
    """
    Activate a specific model for use in the pipeline.
    
    This will:
    1. Load the specified model into memory
    2. Unload the currently active model if different
    3. Update the configuration to use the new model
    4. Return success status
    """
    try:
        logger.info(f"{Fore.GREEN}Activating {model_type} model: {model_id}{Style.RESET_ALL}")
        
        if model_type not in ["translation", "simplification"]:
            raise HTTPException(status_code=400, detail="Model type must be 'translation' or 'simplification'")
        
        # Here you would implement the actual model activation logic
        # For this example, we'll just simulate success
        
        # Update the active model in your model registry or configuration
        
        logger.info(f"{Fore.GREEN}Successfully activated {model_id}{Style.RESET_ALL}")
        return {"status": "success", "message": f"Model {model_id} activated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{Fore.RED}Error activating model {model_id}: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error activating model: {str(e)}")

@router.post("/models/tune", response_model=TuningStatus, summary="Fine-tune a model")
async def tune_model(tuning_request: TuningRequest):
    """
    Start fine-tuning a model with custom data.
    
    This will:
    1. Validate the tuning parameters
    2. Prepare the model and dataset
    3. Start the tuning process in the background
    4. Return a job ID for tracking progress
    """
    try:
        logger.info(f"{Fore.GREEN}Starting model tuning for {tuning_request.modelId}{Style.RESET_ALL}")
        
        # In a real implementation, you would validate paths, check model existence, etc.
        # Then start a background task for the actual tuning process
        
        # Create a new tuning job status
        job_status = TuningStatus(
            modelId=tuning_request.modelId,
            inProgress=True,
            progress=0.0,
            logs=[{"time": datetime.now().strftime("%H:%M:%S"), "message": "Starting tuning process..."}]
        )
        
        # Store the job status for progress tracking
        active_tuning_jobs[tuning_request.modelId] = job_status
        
        logger.info(f"{Fore.GREEN}Tuning job created for {tuning_request.modelId}{Style.RESET_ALL}")
        return job_status
    except Exception as e:
        logger.error(f"{Fore.RED}Error starting model tuning: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error starting tuning: {str(e)}")

@router.get("/models/tune/{model_id}", response_model=TuningStatus, summary="Get tuning status")
async def get_tuning_status(model_id: str = Path(..., description="ID of the model being tuned")):
    """
    Get the status of a model tuning job.
    
    Returns current progress, logs, and whether the job is still in progress.
    """
    try:
        logger.info(f"{Fore.GREEN}Getting tuning status for {model_id}{Style.RESET_ALL}")
        
        if model_id not in active_tuning_jobs:
            raise HTTPException(status_code=404, detail=f"No active tuning job found for model {model_id}")
        
        job_status = active_tuning_jobs[model_id]
        
        # In a real implementation, you would update the progress here by checking the actual tuning job
        # For this example, we'll simulate progress
        if job_status.inProgress and job_status.progress < 100:
            job_status.progress += 5  # Increment progress by 5%
            job_status.logs.append({
                "time": datetime.now().strftime("%H:%M:%S"), 
                "message": f"Training progress: {job_status.progress:.1f}%..."
            })
            
            if job_status.progress >= 100:
                job_status.inProgress = False
                job_status.logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"), 
                    "message": "Training complete!"
                })
        
        logger.info(f"{Fore.GREEN}Tuning progress for {model_id}: {job_status.progress:.1f}%{Style.RESET_ALL}")
        return job_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{Fore.RED}Error getting tuning status: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error getting tuning status: {str(e)}")

# Routes for log viewing
@router.get("/logs/{log_type}", response_model=LogList, summary="Get logs")
async def get_logs(
    log_type: str = Path(..., description="Type of logs to retrieve (veracity or bias)"),
    filter: str = Query("all", description="Filter by status (all, error, warning, success)"),
    search: Optional[str] = Query(None, description="Search text in logs"),
    page: int = Query(1, ge=1, description="Page number"),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_direction: str = Query("desc", description="Sort direction (asc or desc)")
):
    """
    Get logs for veracity or bias checks, with filtering, searching and pagination.
    
    Returns a paginated list of logs with total count and metadata.
    """
    try:
        logger.info(f"{Fore.GREEN}Getting {log_type} logs (page {page}, filter: {filter}){Style.RESET_ALL}")
        
        if log_type not in ["veracity", "bias"]:
            raise HTTPException(status_code=400, detail="Log type must be 'veracity' or 'bias'")
        
        # In a real implementation, you would fetch logs from a database with filtering
        # For this example, we'll generate random logs
        
        # Generate random logs
        all_logs = []
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=i * 30)
            score = round(0.1 + 0.9 * (i % 10 / 10), 2)  # Generate scores between 0.1 and 1.0
            
            status = "success"
            if score < 0.3:
                status = "error"
            elif score < 0.6:
                status = "warning"
            
            all_logs.append(LogEntry(
                id=f"log-{i+1}",
                timestamp=timestamp.isoformat(),
                sessionId=f"session-{1000 + i}",
                text=f"Sample text for {log_type} check #{i+1}",
                score=score,
                status=status,
                details={
                    "model": f"{log_type}-checker-v1",
                    "processingTime": f"{0.1 + i * 0.01:.2f}s",
                    "confidenceLevel": f"{score * 100:.1f}%"
                }
            ))
        
        # Apply filters
        filtered_logs = all_logs
        if filter != "all":
            filtered_logs = [log for log in filtered_logs if log.status == filter]
        
        # Apply search
        if search:
            filtered_logs = [
                log for log in filtered_logs 
                if search.lower() in log.sessionId.lower() or search.lower() in log.text.lower()
            ]
        
        # Apply sorting
        reverse = sort_direction == "desc"
        if sort_by == "timestamp":
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=reverse)
        elif sort_by == "score":
            filtered_logs.sort(key=lambda x: x.score, reverse=reverse)
        elif sort_by == "status":
            status_order = {"error": 0, "warning": 1, "success": 2}
            filtered_logs.sort(key=lambda x: status_order[x.status], reverse=reverse)
        
        # Apply pagination
        page_size = 10
        total_filtered_logs = len(filtered_logs)
        total_pages = (total_filtered_logs + page_size - 1) // page_size
        
        # Adjust page if it's out of bounds
        page = min(max(1, page), max(1, total_pages))
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_logs = filtered_logs[start_idx:end_idx]
        
        logger.info(f"{Fore.GREEN}Retrieved {len(paged_logs)} logs (total: {total_filtered_logs}){Style.RESET_ALL}")
        return LogList(
            logs=paged_logs,
            totalLogs=total_filtered_logs,
            totalPages=total_pages,
            currentPage=page,
            filter=filter,
            search=search,
            sortBy=sort_by,
            sortDirection=sort_direction
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{Fore.RED}Error getting logs: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

# Environment detection and auto-configuration routes
@router.get("/environment", summary="Get detected environment")
async def get_environment():
    """
    Get information about the detected runtime environment.
    
    Returns details about hardware, OS, Python version, GPU availability, etc.
    """
    try:
        logger.info(f"{Fore.GREEN}Getting environment information{Style.RESET_ALL}")
        
        # Get system info which includes environment details
        system_info = get_system_info()
        
        # Extract environment information
        env_info = {
            "os": system_info.detectedEnvironment["os"],
            "pythonVersion": system_info.detectedEnvironment["pythonVersion"],
            "cpuInfo": system_info.detectedEnvironment["cpuInfo"],
            "memoryTotal": system_info.memoryTotal,
            "gpuAvailable": system_info.gpuAvailable,
            "gpuInfo": system_info.detectedEnvironment["gpuInfo"] if system_info.gpuAvailable else None,
            "gpuMemoryTotal": system_info.gpuMemoryTotal if system_info.gpuAvailable else None,
            "environment": system_info.environment
        }
        
        logger.info(f"{Fore.GREEN}Environment detected: {env_info['environment']}, GPU: {env_info['gpuAvailable']}{Style.RESET_ALL}")
        return env_info
    except Exception as e:
        logger.error(f"{Fore.RED}Error getting environment information: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error detecting environment: {str(e)}")

@router.post("/configure-environment", summary="Auto-configure environment")
async def configure_environment():
    """
    Auto-configure the application based on the detected environment.
    
    This will:
    1. Detect available hardware
    2. Select optimal models based on available resources
    3. Configure memory settings
    4. Return the applied configuration
    """
    try:
        logger.info(f"{Fore.GREEN}Auto-configuring environment{Style.RESET_ALL}")
        
        # Get system info
        system_info = get_system_info()
        
        # Determine optimal configuration based on detected environment
        configuration = {
            "useGpu": system_info.gpuAvailable,
            "optimizedModels": {}
        }
        
        # Select models based on available resources
        if system_info.gpuAvailable and system_info.gpuMemoryTotal and system_info.gpuMemoryTotal > 8:
            # High-end GPU with >8GB memory
            configuration["optimizedModels"]["translation"] = "m2m100_1.2B"
            configuration["optimizedModels"]["simplification"] = "t5-base"
            configuration["memorySettings"] = {
                "batchSize": 16,
                "maxSequenceLength": 512
            }
        elif system_info.gpuAvailable:
            # Lower-end GPU
            configuration["optimizedModels"]["translation"] = "m2m100_418M"
            configuration["optimizedModels"]["simplification"] = "t5-small"
            configuration["memorySettings"] = {
                "batchSize": 8,
                "maxSequenceLength": 384
            }
        else:
            # CPU only
            configuration["optimizedModels"]["translation"] = "m2m100_418M"
            configuration["optimizedModels"]["simplification"] = "t5-small"
            configuration["memorySettings"] = {
                "batchSize": 4,
                "maxSequenceLength": 256
            }
        
        # In a real implementation, you would apply this configuration to your models
        
        logger.info(f"{Fore.GREEN}Environment auto-configured: {configuration}{Style.RESET_ALL}")
        return {
            "status": "success",
            "configuration": configuration,
            "message": "Environment auto-configured successfully"
        }
    except Exception as e:
        logger.error(f"{Fore.RED}Error auto-configuring environment: {e}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Error configuring environment: {str(e)}")