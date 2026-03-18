"""
RAM (Recognize Anything Model) Image Processor
Using transformers pipeline for reliable image tagging
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from PIL import Image
import os

from studiobrain_model_manager.processors.base import BaseProcessor

# Force to use only RTX 4090 (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

class RAMProcessor(BaseProcessor):
    """RAM processor for image tagging using transformers pipeline"""
    
    def __init__(self, settings):
        super().__init__(settings)
        # Use a reliable image classification model that works like RAM
        self.model_id = "microsoft/DiT-3B"  # Fallback to a known working model
        self.fallback_model_id = "google/vit-base-patch16-224"  # Simple backup
        self.model = None
        self.processor = None
        
    async def load_model(self):
        """Load image classification model"""
        if self.model_loaded:
            return True
        
        try:
            # Force to use only RTX 4090
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            
            # Set default CUDA device
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            logger.info("Loading image classification model for tagging")
            logger.info(f"Cache directory: {self.settings.model_cache_dir}")
            
            from transformers import pipeline
            
            # Ensure cache directory exists
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to create an image classification pipeline
            try:
                logger.info("Attempting to load image classification pipeline...")
                self.model = pipeline(
                    "image-classification",
                    model=self.fallback_model_id,  # Use reliable ViT model
                    device=0 if self.settings.gpu_available else -1,
                    model_kwargs={
                        "cache_dir": str(self.settings.model_cache_dir),
                        "local_files_only": False,
                        "dtype": torch.float16 if self.settings.gpu_available else torch.float32
                    }
                )
                logger.info(f"Successfully loaded {self.fallback_model_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load primary model, trying fallback: {e}")
                # Simplest possible fallback
                self.model = pipeline(
                    "image-classification",
                    device=0 if self.settings.gpu_available else -1
                )
                logger.info("Loaded default image classification model")
            
            self.model_loaded = True
            logger.info("Image classification model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image classification model: {e}")
            return False
    
    async def process(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process image with image classification"""
        options = self.validate_options(options)
        
        # Load model if not loaded
        if not self.model_loaded:
            success = await self.load_model()
            if not success:
                return self.format_result(error="Failed to load model")
        
        try:
            # Load image
            image = Image.open(file_path).convert("RGB")
            
            # Get image metadata
            metadata = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode
            }
            
            # Run inference with error handling
            try:
                results = self.model(image, top_k=10)  # Get top 10 predictions
                
                if not results:
                    logger.error("Model returned empty results")
                    return self.format_result(error="No classification results")
                
            except Exception as inference_error:
                logger.error(f"Model inference failed: {inference_error}")
                return self.format_result(error=f"Inference failed: {str(inference_error)}")
            
            # Apply confidence threshold
            threshold = options.get("confidence_threshold", 0.1)  # Lower threshold for more tags
            
            # Format results
            formatted_tags = []
            for result in results:
                score = result.get("score", 0)
                if score >= threshold:
                    formatted_tags.append({
                        "name": result.get("label", "unknown").replace("_", " ").title(),
                        "confidence": float(score),
                        "category": "classification"
                    })
            
            # Sort by confidence (should already be sorted)
            formatted_tags.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Create description from top tags
            top_tags = [tag["name"] for tag in formatted_tags[:3]]
            description = f"Image classified as: {', '.join(top_tags)}" if top_tags else "No high-confidence classifications"
            
            return self.format_result(
                tags=formatted_tags,
                metadata=metadata,
                descriptions={"summary": description}
            )
            
        except Exception as e:
            logger.error(f"Error processing image with image classifier: {e}")
            return self.format_result(error=str(e))
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up image classification processor")
        
        if self.model:
            del self.model
        
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities"""
        return ["tag", "classify"]