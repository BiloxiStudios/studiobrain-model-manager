"""
Model Registry for managing AI models
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

# 🚨 CRITICAL: Set GPU configuration BEFORE importing PyTorch
# This ensures proper GPU detection in case registry is imported first
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 4090 only, avoid RTX 5090 compatibility issues

import torch
import gc

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing and loading AI models"""
    
    def __init__(self, settings):
        self.settings = settings
        self.models: Dict[str, Dict[str, Any]] = {}
        self.processors: Dict[str, Any] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the model registry with available models"""
        self.available_models = {
            "image": {
                "qwen_image": {
                    "name": "Qwen-Image",
                    "description": "Qwen's diffusion model for text-to-image generation",
                    "capabilities": ["generate", "text2img"],
                    "size": "4GB",
                    "processor_class": "QwenImageActualProcessor"
                },
                "qwen_image_edit": {
                    "name": "Qwen-Image-Edit",
                    "description": "Qwen's model for image editing and manipulation",
                    "capabilities": ["edit", "inpaint", "outpaint"],
                    "size": "4GB",
                    "processor_class": "QwenImageEditDedicatedProcessor"
                },
                "qwen3_vl": {
                    "name": "Qwen3-VL (LiteLLM)",
                    "description": "Vision-language via LiteLLM proxy (replaces local Qwen3-VL)",
                    "capabilities": ["caption", "describe", "tag", "ocr", "vqa", "analyze"],
                    "size": "0MB (remote)",
                    "processor_class": "LiteLLMVisionProcessor"
                },
                "blip2": {
                    "name": "BLIP-2 (LiteLLM)",
                    "description": "Vision-language via LiteLLM proxy (replaces local BLIP-2)",
                    "capabilities": ["caption", "describe", "vqa", "tag"],
                    "size": "0MB (remote)",
                    "processor_class": "LiteLLMVisionProcessor"
                },
                "florence2": {
                    "name": "Florence-2",
                    "description": "Microsoft's vision-language model for comprehensive image understanding",
                    "capabilities": ["caption", "tag", "detect", "ocr"],
                    "size": "1.5GB",
                    "processor_class": "Florence2Processor"
                },
                "ram": {
                    "name": "Recognize Anything Model",
                    "description": "Multi-label image tagging with 4000+ categories",
                    "capabilities": ["tag"],
                    "size": "1.2GB",
                    "processor_class": "RAMProcessor"
                }
            },
            "audio": {
                "whisper": {
                    "name": "Whisper",
                    "description": "OpenAI's speech recognition model",
                    "capabilities": ["transcribe", "translate"],
                    "size": "1.5GB",
                    "processor_class": "WhisperProcessor"
                }
            },
            "video": {
                "videomae": {
                    "name": "VideoMAE",
                    "description": "Video understanding and action recognition",
                    "capabilities": ["classify", "summarize"],
                    "size": "800MB",
                    "processor_class": "VideoMAEProcessor"
                }
            },
            "model3d": {
                "point_e": {
                    "name": "Point-E",
                    "description": "3D object understanding and classification",
                    "capabilities": ["analyze", "classify"],
                    "size": "1GB",
                    "processor_class": "PointEProcessor"
                }
            },
            "text": {
                "qwen_text": {
                    "name": "Qwen3 Text",
                    "description": "Advanced text generation with thinking/reasoning using Qwen3-8B (16GB bfloat16)",
                    "capabilities": ["completion", "chat", "story_generation", "character_dialogue", "world_description"],
                    "size": "16GB",
                    "processor_class": "QwenTextProcessor"
                }
            },
            "image_edit": {
                "qwen_image_edit": {
                    "name": "Qwen Image Edit (VL)",
                    "description": "AI-guided image editing using Qwen2-VL for analysis",
                    "capabilities": ["analyze", "guided_edit", "filter_apply", "enhancement", "crop_suggestion"],
                    "size": "4GB",
                    "processor_class": "QwenImageEditProcessor"
                },
                "qwen_image_edit_dedicated": {
                    "name": "Qwen-Image-Edit",
                    "description": "Dedicated Qwen-Image-Edit model for advanced image manipulation",
                    "capabilities": ["instruction_edit", "style_transfer", "object_removal", "enhance", "color_adjust"],
                    "size": "6GB",
                    "processor_class": "QwenImageEditDedicatedProcessor"
                }
            },
            "vector": {
                "embedding_service": {
                    "name": "Embedding Service",
                    "description": "Text embedding generation for RAG and similarity search",
                    "capabilities": ["embed", "similarity"],
                    "size": "80MB",
                    "processor_class": "EmbeddingService"
                },
                "chroma_processor": {
                    "name": "ChromaDB Processor",
                    "description": "Vector database operations and similarity search",
                    "capabilities": ["store", "query", "similarity_search"],
                    "size": "50MB",
                    "processor_class": "ChromaProcessor"
                }
            }
        }
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type"""
        result = {}
        for model_type, models in self.available_models.items():
            result[model_type] = list(models.keys())
        return result
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        loaded = []
        for model_type, models in self.models.items():
            for model_name in models:
                loaded.append(f"{model_type}/{model_name}")
        return loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about all models"""
        info = {}
        for model_type, models in self.available_models.items():
            info[model_type] = {}
            for model_name, model_data in models.items():
                is_loaded = model_type in self.models and model_name in self.models.get(model_type, {})
                info[model_type][model_name] = {
                    **model_data,
                    "loaded": is_loaded,
                    "type": model_type
                }
        return info
    
    async def load_model(self, model_type: str, model_name: str) -> bool:
        """Load a specific model into memory"""
        try:
            # Check if model is available
            if model_type not in self.available_models:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            if model_name not in self.available_models[model_type]:
                logger.error(f"Unknown model: {model_name} for type {model_type}")
                return False
            
            # Check if already loaded
            if model_type in self.models and model_name in self.models.get(model_type, {}):
                logger.info(f"Model {model_name} already loaded")
                return True
            
            logger.info(f"Loading model: {model_type}/{model_name}")
            
            # Get processor class
            processor_class_name = self.available_models[model_type][model_name]["processor_class"]
            
            # Dynamically load the processor
            processor = await self._load_processor(model_type, model_name, processor_class_name)
            
            if processor:
                # Load the actual model into the processor
                # Processors use load_model() (BaseProcessor subclasses) or load() (EmbeddingService)
                if hasattr(processor, 'load_model'):
                    load_success = await processor.load_model()
                    if not load_success:
                        logger.error(f"Failed to load model for {model_type}/{model_name}")
                        return False
                elif hasattr(processor, 'load'):
                    load_success = await processor.load()
                    if not load_success:
                        logger.error(f"Failed to load model for {model_type}/{model_name}")
                        return False
                
                # Store the loaded model
                if model_type not in self.models:
                    self.models[model_type] = {}
                self.models[model_type][model_name] = processor
                
                # Store processor reference
                self.processors[f"{model_type}/{model_name}"] = processor
                
                logger.info(f"Successfully loaded {model_type}/{model_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def _load_processor(self, model_type: str, model_name: str, processor_class_name: str):
        """Dynamically load a processor based on model type"""
        try:
            if model_type == "image":
                if model_name == "qwen_image":
                    from studiobrain_model_manager.processors.image.qwen_image_actual import QwenImageActualProcessor
                    return QwenImageActualProcessor(self.settings)
                elif model_name == "qwen_image_edit":
                    from studiobrain_model_manager.processors.image_edit.qwen_image_edit_dedicated import QwenImageEditDedicatedProcessor
                    return QwenImageEditDedicatedProcessor(self.settings)
                elif model_name == "qwen3_vl":
                    from studiobrain_model_manager.processors.image.litellm_vision import LiteLLMVisionProcessor
                    return LiteLLMVisionProcessor(self.settings)
                elif model_name == "blip2":
                    from studiobrain_model_manager.processors.image.litellm_vision import LiteLLMVisionProcessor
                    return LiteLLMVisionProcessor(self.settings)
                elif model_name == "florence2":
                    from studiobrain_model_manager.processors.image.florence2 import Florence2Processor
                    return Florence2Processor(self.settings)
                elif model_name == "ram":
                    from studiobrain_model_manager.processors.image.ram import RAMProcessor
                    return RAMProcessor(self.settings)
            
            elif model_type == "text":
                if model_name == "qwen_text":
                    from studiobrain_model_manager.processors.text.qwen_text import QwenTextProcessor
                    return QwenTextProcessor(self.settings)
            
            elif model_type == "image_edit":
                if model_name == "qwen_image_edit":
                    from studiobrain_model_manager.processors.image_edit.qwen_image_edit import QwenImageEditProcessor
                    return QwenImageEditProcessor(self.settings)
            
            elif model_type == "vector":
                if model_name == "embedding_service":
                    from studiobrain_model_manager.processors.vector.embedding_service import EmbeddingService
                    return EmbeddingService(self.settings)
                elif model_name == "chroma_processor":
                    from studiobrain_model_manager.processors.vector.chroma_processor import ChromaProcessor
                    return ChromaProcessor(self.settings)
            
            # Add other model types as needed
            elif model_type == "audio":
                if model_name == "whisper":
                    # from processors.audio.whisper import WhisperProcessor
                    # return WhisperProcessor(self.settings)
                    logger.info(f"Audio processor {model_name} not yet implemented")
                    return None
            
            logger.warning(f"Processor {processor_class_name} not implemented")
            return None
            
        except ImportError as e:
            logger.error(f"Failed to import processor {processor_class_name}: {e}")
            return None
    
    async def unload_model(self, model_type: str, model_name: str) -> bool:
        """Unload a model from memory. Pass model_type='auto' to search all types."""
        try:
            # Auto-detect model_type by searching all registered types
            if model_type == "auto":
                for mt in list(self.models.keys()):
                    if model_name in self.models.get(mt, {}):
                        model_type = mt
                        break
                else:
                    logger.warning(f"Model {model_name} not found in any type (auto-detect)")
                    return False

            if model_type in self.models and model_name in self.models[model_type]:
                logger.info(f"Unloading model: {model_type}/{model_name}")
                
                # Get the processor
                processor = self.models[model_type][model_name]
                
                # Call cleanup if available
                if hasattr(processor, 'cleanup'):
                    await processor.cleanup()
                
                # Remove from registry
                del self.models[model_type][model_name]
                del self.processors[f"{model_type}/{model_name}"]
                
                # Clean up empty type dict
                if not self.models[model_type]:
                    del self.models[model_type]
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Successfully unloaded {model_type}/{model_name}")
                return True
            
            logger.warning(f"Model {model_type}/{model_name} not loaded")
            return False
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    async def load_default_models(self):
        """Load default models based on configuration"""
        logger.info("Loading default models...")

        # Skip loading - handled by ModelManager
        if hasattr(self.settings, 'auto_load_models') and not self.settings.auto_load_models:
            logger.info("Auto-loading disabled - models handled by ModelManager")
            return

        # Load image models
        if self.settings.enable_image_models:
            for model_name in self.settings.local_image_models:
                if model_name and model_name in self.available_models.get("image", {}):
                    await self.load_model("image", model_name)
        
        # Load audio models
        if self.settings.enable_audio_models:
            for model_name in self.settings.audio_models:
                if model_name and model_name in self.available_models.get("audio", {}):
                    await self.load_model("audio", model_name)
        
        # Load other model types as configured
        # ...
    
    def get_processor(self, model_type: str, model_name: str = None):
        """Get a processor for a specific model type"""
        if model_name:
            # Get specific processor
            if model_type in self.models and model_name in self.models[model_type]:
                return self.models[model_type][model_name]
        else:
            # Get first available processor for type
            if model_type in self.models and self.models[model_type]:
                return list(self.models[model_type].values())[0]
        
        return None
    
    async def cleanup(self):
        """Clean up all loaded models"""
        logger.info("Cleaning up model registry...")
        
        for model_type in list(self.models.keys()):
            for model_name in list(self.models[model_type].keys()):
                await self.unload_model(model_type, model_name)
        
        self.models.clear()
        self.processors.clear()
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()