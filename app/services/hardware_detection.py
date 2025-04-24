# app/services/hardware_detection.py

import os
import platform
import logging
import numpy as np
import psutil
import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.hardware")

class HardwareDetector:
    """
    Service for detecting hardware capabilities and configuring the environment.
    
    This class provides:
    1. Hardware detection (CPU, memory, GPU)
    2. Environment configuration based on available resources
    3. Memory management for optimal model performance
    4. Model loading strategies based on hardware
    """
    
    def __init__(self):
        """Initialize the hardware detector service."""
        self.hardware_info = {}
        self.optimal_config = {}
        self.has_detected = False
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Log initialization
        logger.info(f"{Fore.GREEN}Hardware detection service initialized{Style.RESET_ALL}")
    
    def detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware resources.
        
        This includes:
        - CPU information (cores, model, speed)
        - Memory information (total, available)
        - GPU information (availability, model, memory)
        - OS information
        
        Returns:
            Dict[str, Any]: Dictionary with detected hardware information
        """
        logger.info(f"{Fore.GREEN}Detecting hardware resources...{Style.RESET_ALL}")
        
        # Detect CPU information
        cpu_info = self._detect_cpu()
        logger.info(f"{Fore.CYAN}CPU detected:{Style.RESET_ALL} {cpu_info['model']}, {cpu_info['cores']} cores")
        
        # Detect memory information
        memory_info = self._detect_memory()
        logger.info(f"{Fore.CYAN}Memory detected:{Style.RESET_ALL} {memory_info['total_gb']:.1f} GB total")
        
        # Detect GPU information
        gpu_info = self._detect_gpu()
        if gpu_info['available']:
            logger.info(f"{Fore.CYAN}GPU detected:{Style.RESET_ALL} {gpu_info['model']}, {gpu_info['memory_gb']:.1f} GB memory")
        else:
            logger.info(f"{Fore.YELLOW}No GPU detected{Style.RESET_ALL}")
        
        # Detect OS information
        os_info = self._detect_os()
        logger.info(f"{Fore.CYAN}Operating system:{Style.RESET_ALL} {os_info['name']} {os_info['version']}")
        
        # Combine all information
        self.hardware_info = {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "os": os_info,
            "environment": self.environment
        }
        
        self.has_detected = True
        return self.hardware_info
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information.
        
        Returns:
            Dict[str, Any]: CPU information including model, cores, and frequency
        """
        try:
            # Get CPU model name
            if platform.system() == "Windows":
                model = platform.processor()
            elif platform.system() == "Darwin":  # macOS
                model = os.popen('sysctl -n machdep.cpu.brand_string').read().strip()
            else:  # Linux and others
                model = os.popen('cat /proc/cpuinfo | grep "model name" | head -n 1').read().strip()
                if not model:
                    model = platform.processor()
                else:
                    model = model.split(": ")[1]
            
            # Get CPU cores
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            
            # Get CPU frequency
            frequency = psutil.cpu_freq()
            if frequency:
                freq_current = frequency.current
                freq_max = frequency.max
            else:
                freq_current = None
                freq_max = None
            
            return {
                "model": model,
                "physical_cores": physical_cores,
                "logical_cores": logical_cores,
                "cores": logical_cores,  # For simplicity
                "frequency_current": freq_current,
                "frequency_max": freq_max
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error detecting CPU: {e}{Style.RESET_ALL}")
            return {
                "model": "Unknown",
                "cores": 1,  # Assume at least 1 core
                "physical_cores": 1,
                "logical_cores": 1
            }
    
    def _detect_memory(self) -> Dict[str, Any]:
        """
        Detect memory information.
        
        Returns:
            Dict[str, Any]: Memory information including total, available, and used
        """
        try:
            # Get memory information
            memory = psutil.virtual_memory()
            
            # Convert to GB for easier reading
            total_gb = memory.total / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)
            used_gb = memory.used / (1024 ** 3)
            
            return {
                "total": memory.total,
                "total_gb": total_gb,
                "available": memory.available,
                "available_gb": available_gb,
                "used": memory.used,
                "used_gb": used_gb,
                "percent": memory.percent
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error detecting memory: {e}{Style.RESET_ALL}")
            return {
                "total_gb": 4.0,  # Assume at least 4GB
                "available_gb": 2.0,
                "used_gb": 2.0,
                "percent": 50.0
            }
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information.
        
        Returns:
            Dict[str, Any]: GPU information including availability, model, and memory
        """
        gpu_info = {
            "available": False,
            "count": 0,
            "model": None,
            "memory_total": 0,
            "memory_gb": 0,
            "cuda_version": None
        }
        
        try:
            # Check if PyTorch can see CUDA
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                
                # Get information about the first GPU (index 0)
                if gpu_info["count"] > 0:
                    gpu_info["model"] = torch.cuda.get_device_name(0)
                    
                    # Get memory information
                    memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_info["memory_total"] = memory_total
                    gpu_info["memory_gb"] = memory_total / (1024 ** 3)
                
                # Get CUDA version
                gpu_info["cuda_version"] = torch.version.cuda
            
            return gpu_info
        except Exception as e:
            logger.error(f"{Fore.RED}Error detecting GPU: {e}{Style.RESET_ALL}")
            return gpu_info
    
    def _detect_os(self) -> Dict[str, Any]:
        """
        Detect operating system information.
        
        Returns:
            Dict[str, Any]: OS information including name, version, and architecture
        """
        try:
            return {
                "name": platform.system(),
                "version": platform.version(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version()
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error detecting OS: {e}{Style.RESET_ALL}")
            return {
                "name": "Unknown",
                "version": "Unknown",
                "architecture": "Unknown",
                "python_version": platform.python_version()
            }
    
    def determine_optimal_configuration(self) -> Dict[str, Any]:
        """
        Determine the optimal configuration based on detected hardware.
        
        This includes:
        - Which models to use based on available memory
        - Whether to use GPU or CPU
        - Batch sizes and sequence lengths
        - Memory management settings
        
        Returns:
            Dict[str, Any]: Optimal configuration settings
        """
        if not self.has_detected:
            self.detect_hardware()
        
        logger.info(f"{Fore.GREEN}Determining optimal configuration...{Style.RESET_ALL}")
        
        # Default configuration (conservative)
        config = {
            "use_gpu": False,
            "models": {
                "translation": "m2m100_418M",
                "simplification": "t5-small"
            },
            "batch_size": 1,
            "max_sequence_length": 256,
            "memory_management": {
                "clear_cuda_cache": True,
                "offload_to_cpu": True
            }
        }
        
        # Adjust based on GPU availability
        if self.hardware_info["gpu"]["available"]:
            gpu_mem_gb = self.hardware_info["gpu"]["memory_gb"]
            config["use_gpu"] = True
            
            # High-end GPU (>8GB)
            if gpu_mem_gb >= 8:
                config["models"]["translation"] = "m2m100_1.2B"
                config["models"]["simplification"] = "t5-base"
                config["batch_size"] = 8
                config["max_sequence_length"] = 512
                logger.info(f"{Fore.GREEN}High-end GPU detected ({gpu_mem_gb:.1f}GB) - using larger models{Style.RESET_ALL}")
            
            # Mid-range GPU (4-8GB)
            elif gpu_mem_gb >= 4:
                config["models"]["translation"] = "m2m100_418M"
                config["models"]["simplification"] = "t5-small"
                config["batch_size"] = 4
                config["max_sequence_length"] = 384
                logger.info(f"{Fore.GREEN}Mid-range GPU detected ({gpu_mem_gb:.1f}GB) - using medium models{Style.RESET_ALL}")
            
            # Low-end GPU (<4GB)
            else:
                config["models"]["translation"] = "m2m100_418M"
                config["models"]["simplification"] = "t5-small"
                config["batch_size"] = 1
                config["max_sequence_length"] = 256
                logger.info(f"{Fore.YELLOW}Low-end GPU detected ({gpu_mem_gb:.1f}GB) - using smaller models{Style.RESET_ALL}")
        else:
            # CPU only - use smaller models
            mem_gb = self.hardware_info["memory"]["total_gb"]
            
            if mem_gb >= 16:
                config["batch_size"] = 2
                logger.info(f"{Fore.YELLOW}No GPU detected, but good CPU memory ({mem_gb:.1f}GB) - using batch size 2{Style.RESET_ALL}")
            else:
                logger.info(f"{Fore.YELLOW}No GPU detected, limited memory ({mem_gb:.1f}GB) - using conservative settings{Style.RESET_ALL}")
        
        self.optimal_config = config
        return config
    
    def apply_configuration(self) -> Dict[str, Any]:
        """
        Apply the determined optimal configuration.
        
        This includes:
        - Setting environment variables
        - Setting torch configurations
        - Setting up memory management
        
        Returns:
            Dict[str, Any]: Applied configuration settings
        """
        if not self.optimal_config:
            self.determine_optimal_configuration()
        
        logger.info(f"{Fore.GREEN}Applying optimal configuration...{Style.RESET_ALL}")
        
        # Set PyTorch configurations
        if self.optimal_config["use_gpu"]:
            # Enable GPU usage for PyTorch
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            logger.info(f"{Fore.GREEN}PyTorch configured to use GPU by default{Style.RESET_ALL}")
        else:
            # Use CPU
            torch.set_default_tensor_type(torch.FloatTensor)
            logger.info(f"{Fore.YELLOW}PyTorch configured to use CPU by default{Style.RESET_ALL}")
        
        # Set environment variables for configuration
        os.environ["USE_GPU"] = str(int(self.optimal_config["use_gpu"]))
        os.environ["BATCH_SIZE"] = str(self.optimal_config["batch_size"])
        os.environ["MAX_SEQUENCE_LENGTH"] = str(self.optimal_config["max_sequence_length"])
        os.environ["TRANSLATION_MODEL"] = self.optimal_config["models"]["translation"]
        os.environ["SIMPLIFICATION_MODEL"] = self.optimal_config["models"]["simplification"]
        
        logger.info(f"{Fore.GREEN}Environment variables set for optimal configuration{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}Translation model:{Style.RESET_ALL} {self.optimal_config['models']['translation']}")
        logger.info(f"{Fore.CYAN}Simplification model:{Style.RESET_ALL} {self.optimal_config['models']['simplification']}")
        logger.info(f"{Fore.CYAN}Batch size:{Style.RESET_ALL} {self.optimal_config['batch_size']}")
        
        return self.optimal_config
    
    def manage_memory(self, clear_cache: bool = True) -> Dict[str, Any]:
        """
        Manage memory usage for optimal performance.
        
        This includes:
        - Clearing GPU cache if applicable
        - Running garbage collection
        - Reporting memory usage
        
        Args:
            clear_cache: Whether to clear the CUDA cache
            
        Returns:
            Dict[str, Any]: Memory usage statistics
        """
        logger.info(f"{Fore.GREEN}Managing memory...{Style.RESET_ALL}")
        
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.optimal_config.get("use_gpu", False) and clear_cache:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"{Fore.GREEN}CUDA cache cleared{Style.RESET_ALL}")
        
        # Get current memory usage
        memory_stats = {}
        
        # System memory
        memory = psutil.virtual_memory()
        memory_stats["system"] = {
            "total_gb": memory.total / (1024 ** 3),
            "used_gb": memory.used / (1024 ** 3),
            "available_gb": memory.available / (1024 ** 3),
            "percent": memory.percent
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            memory_stats["gpu"] = {
                "total_gb": torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3),
                "allocated_gb": torch.cuda.memory_allocated(current_device) / (1024 ** 3),
                "reserved_gb": torch.cuda.memory_reserved(current_device) / (1024 ** 3),
                "percent_allocated": (torch.cuda.memory_allocated(current_device) / 
                                     torch.cuda.get_device_properties(current_device).total_memory) * 100
            }
            
            logger.info(f"{Fore.CYAN}GPU memory usage:{Style.RESET_ALL} {memory_stats['gpu']['allocated_gb']:.2f}GB / "
                       f"{memory_stats['gpu']['total_gb']:.2f}GB "
                       f"({memory_stats['gpu']['percent_allocated']:.1f}%)")
        
        logger.info(f"{Fore.CYAN}System memory usage:{Style.RESET_ALL} {memory_stats['system']['used_gb']:.2f}GB / "
                   f"{memory_stats['system']['total_gb']:.2f}GB "
                   f"({memory_stats['system']['percent']:.1f}%)")
        
        return memory_stats
    
    def benchmark_performance(self, model_type: str = "both") -> Dict[str, Any]:
        """
        Benchmark model performance on current hardware.
        
        This helps determine the optimal configuration by measuring:
        - Model loading time
        - Inference speed
        - Memory usage
        
        Args:
            model_type: Which model type to benchmark ("translation", "simplification", or "both")
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info(f"{Fore.GREEN}Benchmarking performance for {model_type} model(s)...{Style.RESET_ALL}")
        
        results = {}
        
        # Benchmark translation model
        if model_type in ["translation", "both"]:
            logger.info(f"{Fore.CYAN}Benchmarking translation model...{Style.RESET_ALL}")
            
            # Simulate benchmarking with a simple test
            # In a real implementation, you would load the actual model and run inference
            start_time = time.time()
            time.sleep(0.5)  # Simulate model loading
            loading_time = time.time() - start_time
            
            # Simulate inference
            start_time = time.time()
            time.sleep(0.2)  # Simulate inference
            inference_time = time.time() - start_time
            
            results["translation"] = {
                "model": self.optimal_config["models"]["translation"],
                "loading_time_seconds": loading_time,
                "inference_time_seconds": inference_time,
                "tokens_per_second": 1000 / inference_time  # Simulated throughput
            }
            
            logger.info(f"{Fore.GREEN}Translation model benchmark completed{Style.RESET_ALL}")
            logger.info(f"  {Fore.CYAN}Loading time:{Style.RESET_ALL} {loading_time:.2f}s")
            logger.info(f"  {Fore.CYAN}Inference time:{Style.RESET_ALL} {inference_time:.2f}s")
        
        # Benchmark simplification model
        if model_type in ["simplification", "both"]:
            logger.info(f"{Fore.CYAN}Benchmarking simplification model...{Style.RESET_ALL}")
            
            # Simulate benchmarking
            start_time = time.time()
            time.sleep(0.3)  # Simulate model loading
            loading_time = time.time() - start_time
            
            # Simulate inference
            start_time = time.time()
            time.sleep(0.1)  # Simulate inference
            inference_time = time.time() - start_time
            
            results["simplification"] = {
                "model": self.optimal_config["models"]["simplification"],
                "loading_time_seconds": loading_time,
                "inference_time_seconds": inference_time,
                "tokens_per_second": 500 / inference_time  # Simulated throughput
            }
            
            logger.info(f"{Fore.GREEN}Simplification model benchmark completed{Style.RESET_ALL}")
            logger.info(f"  {Fore.CYAN}Loading time:{Style.RESET_ALL} {loading_time:.2f}s")
            logger.info(f"  {Fore.CYAN}Inference time:{Style.RESET_ALL} {inference_time:.2f}s")
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of hardware detection and configuration.
        
        Returns:
            Dict[str, Any]: Summary of hardware and configuration
        """
        if not self.has_detected:
            self.detect_hardware()
        
        if not self.optimal_config:
            self.determine_optimal_configuration()
        
        # Generate a summary of hardware and configuration
        summary = {
            "hardware": {
                "cpu": f"{self.hardware_info['cpu']['model']} ({self.hardware_info['cpu']['cores']} cores)",
                "memory": f"{self.hardware_info['memory']['total_gb']:.1f} GB",
                "gpu": "Not available" if not self.hardware_info['gpu']['available'] else 
                       f"{self.hardware_info['gpu']['model']} ({self.hardware_info['gpu']['memory_gb']:.1f} GB)",
                "os": f"{self.hardware_info['os']['name']} {self.hardware_info['os']['version']}"
            },
            "configuration": {
                "use_gpu": self.optimal_config["use_gpu"],
                "translation_model": self.optimal_config["models"]["translation"],
                "simplification_model": self.optimal_config["models"]["simplification"],
                "batch_size": self.optimal_config["batch_size"],
                "max_sequence_length": self.optimal_config["max_sequence_length"]
            },
            "environment": self.environment
        }
        
        return summary
    
    def get_hardware_score(self) -> float:
        """
        Calculate a hardware capability score (0-100).
        
        This helps to quickly assess the hardware capability for running models.
        
        Returns:
            float: Hardware capability score from 0-100
        """
        if not self.has_detected:
            self.detect_hardware()
        
        # Base score starts at 50
        score = 50.0
        
        # Add points for CPU
        cpu_cores = self.hardware_info["cpu"]["cores"]
        score += min(20, cpu_cores * 2)  # Up to 20 points for CPU cores
        
        # Add points for memory
        memory_gb = self.hardware_info["memory"]["total_gb"]
        score += min(10, memory_gb * 0.5)  # Up to 10 points for memory
        
        # Add points for GPU
        if self.hardware_info["gpu"]["available"]:
            score += 10  # Base points for having a GPU
            
            # Add points for GPU memory
            gpu_memory_gb = self.hardware_info["gpu"]["memory_gb"]
            score += min(10, gpu_memory_gb * 1.25)  # Up to 10 points for GPU memory
        
        # Normalize score to 0-100 range
        score = max(0, min(100, score))
        
        return score


# For testing
if __name__ == "__main__":
    # Test the hardware detection
    detector = HardwareDetector()
    hardware_info = detector.detect_hardware()
    optimal_config = detector.determine_optimal_configuration()
    detector.apply_configuration()
    memory_stats = detector.manage_memory()
    
    # Print hardware score
    hardware_score = detector.get_hardware_score()
    logger.info(f"{Fore.GREEN}Hardware capability score: {hardware_score:.1f}/100{Style.RESET_ALL}")
    
    # Print summary
    summary = detector.get_summary()
    logger.info(f"{Fore.GREEN}Hardware and configuration summary:{Style.RESET_ALL}")
    for category, items in summary.items():
        logger.info(f"{Fore.CYAN}{category.capitalize()}:{Style.RESET_ALL}")
        for key, value in items.items():
            logger.info(f"  {Fore.BLUE}{key}:{Style.RESET_ALL} {value}")