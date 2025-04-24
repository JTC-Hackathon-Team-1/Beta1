# app/services/model_registry.py

import os
import torch
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.model_registry")

class ModelRegistry:
    """
    Central registry for managing models in the CasaLingua application.
    
    This class provides:
    1. Model registration, loading, and unloading
    2. Model versioning and tracking
    3. Hardware-aware model selection
    4. Automatic fallbacks for different hardware configurations
    5. Model performance metrics
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Dictionary to store loaded models
        self.loaded_models = {}
        
        # Dictionary to store model metadata
        self.model_metadata = {}
        
        # Dictionary to track model usage
        self.model_usage = {}
        
        # Flag to track if we've loaded the registry
        self.registry_loaded = False
        
        # Flag to determine if we're using GPU
        self.use_gpu = torch.cuda.is_available()
        
        logger.info(f"{Fore.GREEN}Model registry initialized{Style.RESET_ALL}")
        logger.info(f"{Fore.CYAN}Models directory:{Style.RESET_ALL} {self.models_dir}")
        logger.info(f"{Fore.CYAN}GPU enabled:{Style.RESET_ALL} {self.use_gpu}")
    
    def load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry metadata from disk.
        
        Returns:
            Dict[str, Any]: Dictionary of registered models
        """
        registry_file = self.models_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    self.model_metadata = json.load(f)
                logger.info(f"{Fore.GREEN}Loaded model registry with {len(self.model_metadata)} models{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"{Fore.RED}Error loading model registry: {e}{Style.RESET_ALL}")
                self.model_metadata = {}
        else:
            logger.warning(f"{Fore.YELLOW}Model registry file not found, creating default registry{Style.RESET_ALL}")
            self._create_default_registry()
            self._save_registry()
        
        self.registry_loaded = True
        return self.model_metadata
    
    def _create_default_registry(self):
        """Create a default registry with standard models."""
        self.model_metadata = {
            "translation": {
                "m2m100_418M": {
                    "name": "M2M100 (418M)",
                    "description": "Facebook/Meta multilingual translation model (small)",
                    "path": "facebook/m2m100_418M",
                    "size_mb": 418,
                    "languages": 100,
                    "min_memory_gb": 2,
                    "recommended_memory_gb": 4,
                    "min_gpu_memory_gb": 2,
                    "loading_time_seconds": 5.0,
                    "performance": {
                        "speed": 4.2,
                        "accuracy": 8.1,
                        "memory_usage_mb": 350
                    }
                },
                "m2m100_1.2B": {
                    "name": "M2M100 (1.2B)",
                    "description": "Facebook/Meta multilingual translation model (large)",
                    "path": "facebook/m2m100_1.2B",
                    "size_mb": 1200,
                    "languages": 100,
                    "min_memory_gb": 4,
                    "recommended_memory_gb": 8,
                    "min_gpu_memory_gb": 4,
                    "loading_time_seconds": 10.0,
                    "performance": {
                        "speed": 2.8,
                        "accuracy": 8.9,
                        "memory_usage_mb": 1100
                    }
                },
                "nllb_1.3B": {
                    "name": "NLLB (1.3B)",
                    "description": "Meta 'No Language Left Behind' model",
                    "path": "facebook/nllb-200-1.3B",
                    "size_mb": 1300,
                    "languages": 200,
                    "min_memory_gb": 4,
                    "recommended_memory_gb": 8,
                    "min_gpu_memory_gb": 4,
                    "loading_time_seconds": 12.0,
                    "performance": {
                        "speed": 2.5,
                        "accuracy": 9.2,
                        "memory_usage_mb": 1200
                    }
                }
            },
            "simplification": {
                "t5-small": {
                    "name": "T5 Small",
                    "description": "Text-to-Text Transfer Transformer (small version)",
                    "path": "t5-small",
                    "size_mb": 242,
                    "min_memory_gb": 1,
                    "recommended_memory_gb": 2,
                    "min_gpu_memory_gb": 1,
                    "loading_time_seconds": 3.0,
                    "performance": {
                        "speed": 5.1,
                        "accuracy": 7.8,
                        "memory_usage_mb": 220
                    }
                },
                "t5-base": {
                    "name": "T5 Base",
                    "description": "T5 base model with better performance",
                    "path": "t5-base",
                    "size_mb": 892,
                    "min_memory_gb": 2,
                    "recommended_memory_gb": 4,
                    "min_gpu_memory_gb": 2,
                    "loading_time_seconds": 6.0,
                    "performance": {
                        "speed": 3.4,
                        "accuracy": 8.5,
                        "memory_usage_mb": 850
                    }
                },
                "byt5-small": {
                    "name": "ByT5 Small",
                    "description": "Byte-level T5 model that works at the byte level",
                    "path": "google/byt5-small",
                    "size_mb": 300,
                    "min_memory_gb": 1,
                    "recommended_memory_gb": 2,
                    "min_gpu_memory_gb": 1,
                    "loading_time_seconds": 4.0,
                    "performance": {
                        "speed": 4.8,
                        "accuracy": 8.2,
                        "memory_usage_mb": 290
                    }
                },
                "legal-t5": {
                    "name": "Legal-T5 (Custom)",
                    "description": "Custom fine-tuned T5 model for legal document simplification",
                    "path": "custom/legal-t5",
                    "size_mb": 450,
                    "min_memory_gb": 2,
                    "recommended_memory_gb": 4,
                    "min_gpu_memory_gb": 2,
                    "loading_time_seconds": 5.0,
                    "performance": {
                        "speed": 3.9,
                        "accuracy": 8.7,
                        "memory_usage_mb": 420
                    }
                }
            }
        }
    
    def _save_registry(self):
        """Save the model registry metadata to disk."""
        registry_file = self.models_dir / "registry.json"
        
        try:
            with open(registry_file, "w") as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info(f"{Fore.GREEN}Saved model registry with {len(self.model_metadata)} model types{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving model registry: {e}{Style.RESET_ALL}")
    
    def register_model(self, 
                      model_type: str, 
                      model_id: str, 
                      metadata: Dict[str, Any],
                      save_registry: bool = True) -> bool:
        """
        Register a new model or update an existing model's metadata.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            model_id: Unique identifier for the model
            metadata: Dictionary of model metadata
            save_registry: Whether to save the registry after registering
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_loaded:
            self.load_registry()
        
        # Create model type dictionary if it doesn't exist
        if model_type not in self.model_metadata:
            self.model_metadata[model_type] = {}
        
        # Check if this is a new model or an update
        is_new = model_id not in self.model_metadata[model_type]
        action = "Registered new" if is_new else "Updated"
        
        # Add or update the model metadata
        self.model_metadata[model_type][model_id] = metadata
        
        logger.info(f"{Fore.GREEN}{action} model: {model_type}/{model_id}{Style.RESET_ALL}")
        
        # Save the registry if requested
        if save_registry:
            self._save_registry()
        
        return True
    
    def load_model(self, 
                  model_type: str, 
                  model_id: str,
                  force_reload: bool = False,
                  fallback: bool = True) -> Optional[Any]:
        """
        Load a model into memory.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            model_id: Unique identifier for the model
            force_reload: Whether to force reload if already loaded
            fallback: Whether to try a fallback model if this one fails
            
        Returns:
            Optional[Any]: The loaded model or None if failed
        """
        if not self.registry_loaded:
            self.load_registry()
        
        # Check if model exists in registry
        if model_type not in self.model_metadata or model_id not in self.model_metadata[model_type]:
            logger.error(f"{Fore.RED}Model {model_type}/{model_id} not found in registry{Style.RESET_ALL}")
            return None
        
        # Generate a unique key for the loaded model
        model_key = f"{model_type}/{model_id}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models and not force_reload:
            logger.info(f"{Fore.GREEN}Model {model_key} already loaded{Style.RESET_ALL}")
            
            # Update usage tracking
            if model_key not in self.model_usage:
                self.model_usage[model_key] = {"load_count": 1, "use_count": 0, "last_used": time.time()}
            self.model_usage[model_key]["use_count"] += 1
            self.model_usage[model_key]["last_used"] = time.time()
            
            return self.loaded_models[model_key]
        
        # Get model metadata
        model_metadata = self.model_metadata[model_type][model_id]
        model_path = model_metadata["path"]
        
        # Check if we have enough memory
        if self.use_gpu:
            # Check GPU memory
            min_gpu_memory = model_metadata.get("min_gpu_memory_gb", 0)
            
            if torch.cuda.is_available():
                # Get available GPU memory in GB
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
                free_memory = total_memory - allocated_memory
                
                if free_memory < min_gpu_memory:
                    logger.warning(f"{Fore.YELLOW}Insufficient GPU memory to load {model_key}. "
                                  f"Need {min_gpu_memory:.1f}GB, have {free_memory:.1f}GB free.{Style.RESET_ALL}")
                    
                    if fallback:
                        logger.info(f"{Fore.YELLOW}Attempting to use CPU instead{Style.RESET_ALL}")
                        self.use_gpu = False
                    else:
                        return None
        else:
            # Check system memory
            import psutil
            min_memory = model_metadata.get("min_memory_gb", 0)
            
            memory = psutil.virtual_memory()
            free_memory_gb = memory.available / (1024 ** 3)
            
            if free_memory_gb < min_memory:
                logger.warning(f"{Fore.YELLOW}Low system memory for loading {model_key}. "
                              f"Need {min_memory:.1f}GB, have {free_memory_gb:.1f}GB free.{Style.RESET_ALL}")
                
                if free_memory_gb < min_memory * 0.8 and fallback:
                    # Try to find a smaller model
                    return self._load_fallback_model(model_type, model_id)
        
        # Attempt to load the model
        try:
            logger.info(f"{Fore.GREEN}Loading model {model_key} from {model_path}{Style.RESET_ALL}")
            start_time = time.time()
            
            # Here we would have model type-specific loading logic
            # For this example, we'll simulate model loading
            if model_type == "translation":
                # Simulate loading translation model
                # In a real implementation, you'd use code like:
                # from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
                # tokenizer = M2M100Tokenizer.from_pretrained(model_path)
                # model = M2M100ForConditionalGeneration.from_pretrained(model_path)
                # if self.use_gpu:
                #     model = model.to("cuda")
                time.sleep(1.0)  # Simulate loading time
                model = {"name": model_metadata["name"], "type": "translation", "path": model_path}
            
            elif model_type == "simplification":
                # Simulate loading simplification model
                # In a real implementation, you'd use code like:
                # from transformers import T5ForConditionalGeneration, T5Tokenizer
                # tokenizer = T5

# app/services/model_registry.py (continued)

                # Tokenizer = T5Tokenizer.from_pretrained(model_path)
                # model = T5ForConditionalGeneration.from_pretrained(model_path)
                # if self.use_gpu:
                #     model = model.to("cuda")
                time.sleep(0.5)  # Simulate loading time
                model = {"name": model_metadata["name"], "type": "simplification", "path": model_path}
            
            else:
                logger.warning(f"{Fore.YELLOW}Unknown model type: {model_type}{Style.RESET_ALL}")
                return None
            
            # Calculate loading time
            loading_time = time.time() - start_time
            
            # Store the loaded model
            self.loaded_models[model_key] = model
            
            # Update usage tracking
            if model_key not in self.model_usage:
                self.model_usage[model_key] = {"load_count": 0, "use_count": 0, "last_used": time.time()}
            
            self.model_usage[model_key]["load_count"] += 1
            self.model_usage[model_key]["use_count"] += 1
            self.model_usage[model_key]["last_used"] = time.time()
            self.model_usage[model_key]["loading_time"] = loading_time
            
            logger.info(f"{Fore.GREEN}Successfully loaded model {model_key} in {loading_time:.2f}s{Style.RESET_ALL}")
            
            return model
        
        except Exception as e:
            logger.error(f"{Fore.RED}Error loading model {model_key}: {e}{Style.RESET_ALL}")
            
            # Try a fallback model if requested
            if fallback:
                return self._load_fallback_model(model_type, model_id)
            
            return None
    
    def _load_fallback_model(self, model_type: str, failed_model_id: str) -> Optional[Any]:
        """
        Attempt to load a fallback model if the requested model fails to load.
        
        This tries to find a smaller, more compatible model of the same type.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            failed_model_id: ID of the model that failed to load
            
        Returns:
            Optional[Any]: The fallback model or None if no suitable fallback found
        """
        logger.info(f"{Fore.YELLOW}Attempting to find fallback for {model_type}/{failed_model_id}{Style.RESET_ALL}")
        
        if model_type not in self.model_metadata:
            return None
        
        # Get metadata for the failed model
        failed_metadata = self.model_metadata[model_type].get(failed_model_id)
        if not failed_metadata:
            return None
        
        # Find models of the same type, sorted by size (smallest first)
        candidates = []
        for model_id, metadata in self.model_metadata[model_type].items():
            if model_id != failed_model_id:
                # Calculate a score based on size and performance
                # Lower score = better fallback candidate
                size_mb = metadata.get("size_mb", 1000)
                accuracy = metadata.get("performance", {}).get("accuracy", 5.0)
                
                # Prioritize smaller models with reasonable accuracy
                score = size_mb / (accuracy * 2)
                
                candidates.append((model_id, metadata, score))
        
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[2])
        
        # Try to load each candidate, starting with the best
        for model_id, metadata, score in candidates:
            logger.info(f"{Fore.YELLOW}Trying fallback model {model_type}/{model_id} (score: {score:.2f}){Style.RESET_ALL}")
            
            # Attempt to load this model (without further fallbacks to avoid recursion)
            model = self.load_model(model_type, model_id, force_reload=False, fallback=False)
            
            if model is not None:
                logger.info(f"{Fore.GREEN}Successfully loaded fallback model {model_type}/{model_id}{Style.RESET_ALL}")
                return model
        
        logger.error(f"{Fore.RED}No suitable fallback model found for {model_type}/{failed_model_id}{Style.RESET_ALL}")
        return None
    
    def unload_model(self, model_type: str, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            model_id: Unique identifier for the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_key = f"{model_type}/{model_id}"
        
        if model_key in self.loaded_models:
            # Unload the model
            try:
                # In a real implementation, you might need specific cleanup code
                # For example: del self.loaded_models[model_key]
                # And potentially: torch.cuda.empty_cache() if using GPU
                
                # For this example, simply remove from the dictionary
                del self.loaded_models[model_key]
                
                # Optionally run garbage collection
                import gc
                gc.collect()
                
                # If GPU is being used, clear cache
                if self.use_gpu and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"{Fore.GREEN}Unloaded model {model_key}{Style.RESET_ALL}")
                return True
            
            except Exception as e:
                logger.error(f"{Fore.RED}Error unloading model {model_key}: {e}{Style.RESET_ALL}")
                return False
        else:
            logger.warning(f"{Fore.YELLOW}Model {model_key} not loaded, nothing to unload{Style.RESET_ALL}")
            return False
    
    def get_model_info(self, model_type: str, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            model_id: Unique identifier for the model
            
        Returns:
            Optional[Dict[str, Any]]: Model information or None if not found
        """
        if not self.registry_loaded:
            self.load_registry()
        
        if model_type in self.model_metadata and model_id in self.model_metadata[model_type]:
            model_key = f"{model_type}/{model_id}"
            info = self.model_metadata[model_type][model_id].copy()
            
            # Add usage information if available
            if model_key in self.model_usage:
                info["usage"] = self.model_usage[model_key]
            
            # Add loaded status
            info["is_loaded"] = model_key in self.loaded_models
            
            return info
        
        return None
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered models, optionally filtered by type.
        
        Args:
            model_type: Optional type to filter by
            
        Returns:
            Dict[str, Any]: Dictionary of registered models
        """
        if not self.registry_loaded:
            self.load_registry()
        
        if model_type:
            if model_type in self.model_metadata:
                return {model_type: self.model_metadata[model_type]}
            return {}
        
        return self.model_metadata
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """
        Get a dictionary of currently loaded models.
        
        Returns:
            Dict[str, Any]: Dictionary mapping model keys to loaded models
        """
        return self.loaded_models
    
    def get_recommended_model(self, model_type: str, hardware_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the recommended model ID for a given type and hardware configuration.
        
        Args:
            model_type: Type of model (e.g., 'translation', 'simplification')
            hardware_info: Optional hardware information to use for recommendation
            
        Returns:
            str: Model ID of the recommended model
        """
        if not self.registry_loaded:
            self.load_registry()
        
        if model_type not in self.model_metadata:
            logger.error(f"{Fore.RED}Unknown model type: {model_type}{Style.RESET_ALL}")
            return ""
        
        # If no hardware info provided, detect it
        if hardware_info is None:
            # Use GPU info if available
            if self.use_gpu and torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
                
                hardware_info = {
                    "gpu_memory_gb": gpu_memory_gb,
                    "has_gpu": True
                }
            else:
                # Use system memory
                import psutil
                memory = psutil.virtual_memory()
                system_memory_gb = memory.total / (1024 ** 3)
                
                hardware_info = {
                    "system_memory_gb": system_memory_gb,
                    "has_gpu": False
                }
        
        # Score each model based on hardware compatibility and performance
        model_scores = []
        
        for model_id, metadata in self.model_metadata[model_type].items():
            # Base score starts at the accuracy score
            accuracy = metadata.get("performance", {}).get("accuracy", 5.0)
            base_score = accuracy
            
            # Check memory requirements
            if hardware_info.get("has_gpu", False):
                min_gpu_memory = metadata.get("min_gpu_memory_gb", 0)
                available_gpu_memory = hardware_info.get("gpu_memory_gb", 0)
                
                # Penalize if not enough GPU memory
                if available_gpu_memory < min_gpu_memory:
                    base_score -= 5.0
                # Bonus if plenty of GPU memory
                elif available_gpu_memory > min_gpu_memory * 2:
                    base_score += 1.0
            else:
                min_memory = metadata.get("min_memory_gb", 0)
                available_memory = hardware_info.get("system_memory_gb", 0)
                
                # Penalize if not enough system memory
                if available_memory < min_memory:
                    base_score -= 5.0
                # Bonus if plenty of system memory
                elif available_memory > min_memory * 2:
                    base_score += 1.0
            
            # Add the model and its score
            model_scores.append((model_id, base_score))
        
        # Sort by score (highest first)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        if model_scores:
            recommended_id = model_scores[0][0]
            logger.info(f"{Fore.GREEN}Recommended model for {model_type}: {recommended_id} (score: {model_scores[0][1]:.2f}){Style.RESET_ALL}")
            return recommended_id
        
        return ""
    
    def update_model_stats(self, model_type: str, model_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Update performance statistics for a model based on usage.
        
        Args:
            model_type: Type of model
            model_id: Unique identifier for the model
            performance_data: Dictionary of performance metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_loaded:
            self.load_registry()
        
        if model_type not in self.model_metadata or model_id not in self.model_metadata[model_type]:
            logger.error(f"{Fore.RED}Model {model_type}/{model_id} not found in registry{Style.RESET_ALL}")
            return False
        
        try:
            # Update the model's performance data
            if "performance" not in self.model_metadata[model_type][model_id]:
                self.model_metadata[model_type][model_id]["performance"] = {}
            
            # Update performance metrics
            for key, value in performance_data.items():
                self.model_metadata[model_type][model_id]["performance"][key] = value
            
            logger.info(f"{Fore.GREEN}Updated performance stats for {model_type}/{model_id}{Style.RESET_ALL}")
            
            # Save the registry
            self._save_registry()
            
            return True
        
        except Exception as e:
            logger.error(f"{Fore.RED}Error updating model stats: {e}{Style.RESET_ALL}")
            return False
    
    def clear_unused_models(self, idle_seconds: int = 3600) -> int:
        """
        Unload models that haven't been used recently to free up memory.
        
        Args:
            idle_seconds: Number of seconds of inactivity before unloading
            
        Returns:
            int: Number of models unloaded
        """
        # Get current time
        current_time = time.time()
        unloaded_count = 0
        
        models_to_unload = []
        
        # Find models that haven't been used recently
        for model_key, model in self.loaded_models.items():
            if model_key in self.model_usage:
                last_used = self.model_usage[model_key].get("last_used", 0)
                
                if current_time - last_used > idle_seconds:
                    models_to_unload.append(model_key)
        
        # Unload the identified models
        for model_key in models_to_unload:
            model_type, model_id = model_key.split("/", 1)
            
            if self.unload_model(model_type, model_id):
                unloaded_count += 1
        
        if unloaded_count > 0:
            logger.info(f"{Fore.GREEN}Unloaded {unloaded_count} unused models{Style.RESET_ALL}")
        
        # Run garbage collection to free memory
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return unloaded_count
    
    def get_hardware_usage(self) -> Dict[str, Any]:
        """
        Get current hardware resource usage information.
        
        Returns:
            Dict[str, Any]: Current hardware usage statistics
        """
        usage_stats = {}
        
        # Get system memory usage
        import psutil
        memory = psutil.virtual_memory()
        
        usage_stats["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_gb": memory.total / (1024 ** 3),
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_percent": memory.percent
        }
        
        # Get GPU usage if available
        if self.use_gpu and torch.cuda.is_available():
            device = torch.cuda.current_device()
            
            # Get the device properties
            props = torch.cuda.get_device_properties(device)
            
            # Calculate memory usage
            total_memory = props.total_memory / (1024 ** 3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
            
            usage_stats["gpu"] = {
                "name": props.name,
                "memory_total_gb": total_memory,
                "memory_allocated_gb": allocated_memory,
                "memory_reserved_gb": reserved_memory,
                "memory_percent": (allocated_memory / total_memory) * 100
            }
        
        return usage_stats


# For testing
if __name__ == "__main__":
    # Test the model registry
    registry = ModelRegistry()
    
    # Load the registry
    registry.load_registry()
    
    # List all models
    models = registry.list_models()
    logger.info(f"{Fore.GREEN}Available models:{Style.RESET_ALL}")
    for model_type, model_dict in models.items():
        logger.info(f"{Fore.CYAN}{model_type}:{Style.RESET_ALL}")
        for model_id, metadata in model_dict.items():
            logger.info(f"  {Fore.BLUE}{model_id}:{Style.RESET_ALL} {metadata['name']}")
    
    # Get recommended models
    for model_type in models.keys():
        recommended_id = registry.get_recommended_model(model_type)
        logger.info(f"{Fore.GREEN}Recommended {model_type} model:{Style.RESET_ALL} {recommended_id}")
    
    # Test loading a model
    test_model_type = "simplification"
    test_model_id = "t5-small"
    
    model = registry.load_model(test_model_type, test_model_id)
    
    if model:
        logger.info(f"{Fore.GREEN}Successfully loaded test model{Style.RESET_ALL}")
        
        # Get hardware usage
        usage = registry.get_hardware_usage()
        logger.info(f"{Fore.GREEN}Hardware usage:{Style.RESET_ALL}")
        
        # Display system usage
        system = usage.get("system", {})
        logger.info(f"  {Fore.CYAN}CPU:{Style.RESET_ALL} {system.get('cpu_percent')}%")
        logger.info(f"  {Fore.CYAN}Memory:{Style.RESET_ALL} {system.get('memory_used_gb'):.2f}GB / "
                    f"{system.get('memory_total_gb'):.2f}GB ({system.get('memory_percent')}%)")
        
        # Display GPU usage if available
        if "gpu" in usage:
            gpu = usage["gpu"]
            logger.info(f"  {Fore.CYAN}GPU:{Style.RESET_ALL} {gpu.get('name')}")
            logger.info(f"  {Fore.CYAN}GPU Memory:{Style.RESET_ALL} {gpu.get('memory_allocated_gb'):.2f}GB / "
                        f"{gpu.get('memory_total_gb'):.2f}GB ({gpu.get('memory_percent'):.2f}%)")
        
        # Unload the model
        registry.unload_model(test_model_type, test_model_id)