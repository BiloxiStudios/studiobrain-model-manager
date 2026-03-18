"""
BLIP-2 Image Processor
Salesforce's vision-language model for image captioning and VQA
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

class BLIP2Processor(BaseProcessor):
    """BLIP-2 processor for image captioning and visual question answering"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.model_id = "Salesforce/blip2-opt-2.7b"
        self.model = None
        self.processor = None
        
    async def load_model(self):
        """Load BLIP-2 model"""
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
            
            logger.info(f"Loading BLIP-2 model from {self.model_id}")
            logger.info(f"Cache directory: {self.settings.model_cache_dir}")
            
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            # Ensure cache directory exists
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load processor
            logger.info("Loading BLIP-2 processor...")
            self.processor = Blip2Processor.from_pretrained(
                self.model_id,
                cache_dir=self.settings.model_cache_dir,
                local_files_only=False
            )
            
            # Load model
            logger.info("Loading BLIP-2 model...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.settings.model_cache_dir,
                dtype=torch.float16 if self.settings.gpu_available else torch.float32,
                device_map="auto" if self.settings.gpu_available else None,
                local_files_only=False
            )
            
            if not self.settings.gpu_available:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            
            logger.info("BLIP-2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            return False
    
    async def process(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process image with BLIP-2"""
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
            
            results = {}
            tags = []
            descriptions = {}
            features = {}
            
            # Generate caption
            if options.get("enable_caption", True):
                caption = await self._generate_caption(image)
                if caption:
                    descriptions["short"] = caption
                    # Extract basic tags from caption
                    caption_tags = self._extract_tags_from_text(caption)
                    tags.extend(caption_tags)
            
            # Generate detailed description via conditional generation
            if options.get("enable_detailed", True):
                detailed = await self._generate_detailed_description(image)
                if detailed:
                    descriptions["detailed"] = detailed
                    # Extract more tags from detailed description
                    detailed_tags = self._extract_tags_from_text(detailed)
                    tags.extend(detailed_tags)
            
            # Answer specific questions about the image
            if options.get("questions"):
                questions = options["questions"]
                if isinstance(questions, str):
                    questions = [questions]
                
                answers = {}
                for question in questions:
                    answer = await self._answer_question(image, question)
                    if answer:
                        answers[question] = answer
                
                if answers:
                    features["qa_results"] = answers
            
            # Deduplicate tags
            unique_tags = self._deduplicate_tags(tags)
            
            # Apply confidence threshold
            threshold = options.get("confidence_threshold", 0.3)
            filtered_tags = [tag for tag in unique_tags if tag.get("confidence", 1.0) >= threshold]
            
            return self.format_result(
                tags=filtered_tags,
                descriptions=descriptions,
                metadata=metadata,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Error processing image with BLIP-2: {e}")
            return self.format_result(error=str(e))
    
    async def _generate_caption(self, image: Image) -> Optional[str]:
        """Generate a caption for the image"""
        try:
            logger.info("Generating BLIP-2 caption...")
            inputs = self.processor(image, return_tensors="pt")
            
            if self.settings.gpu_available:
                inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
            
            logger.info(f"BLIP-2 input keys: {list(inputs.keys())}")
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for caption")
                    return None
                
                logger.info(f"BLIP-2 generation successful, shape: {generated_ids.shape}")
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"BLIP-2 generated text: '{generated_text}'")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return None
    
    async def _generate_detailed_description(self, image: Image) -> Optional[str]:
        """
        Generate a detailed description.
        NOTE: BLIP-2 doesn't support conditional prompts - it generates captions unconditionally.
        We use longer generation with beam search for more detailed output.
        """
        try:
            logger.info("Generating BLIP-2 detailed description...")

            # BLIP-2 generates captions without prompts
            inputs = self.processor(image, return_tensors="pt")

            if self.settings.gpu_available:
                inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    min_new_tokens=20,  # Ensure longer output for details
                    do_sample=True,     # Enable sampling for variation
                    temperature=0.7,    # Add creativity
                    num_beams=3,        # Beam search for quality
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

                if generated_ids is None:
                    logger.error("Model.generate() returned None for detailed description")
                    return None

                logger.info("BLIP-2 detailed description generated successfully")

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info(f"BLIP-2 generated detailed text: '{generated_text}'")

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating detailed description: {e}")
            return None
    
    async def _answer_question(self, image: Image, question: str) -> Optional[str]:
        """Answer a question about the image"""
        try:
            inputs = self.processor(image, text=question, return_tensors="pt")
            
            if self.settings.gpu_available:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for VQA")
                    return None
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {e}")
            return None
    
    def _extract_tags_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags from generated text"""
        import re
        
        tags = []
        if not text:
            return tags
        
        # Extract objects and nouns
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Common objects and descriptors
        relevant_words = []
        for word in words:
            if len(word) > 2 and word not in ['the', 'and', 'with', 'this', 'that', 'there', 'image', 'picture', 'photo']:
                relevant_words.append(word)
        
        # Get unique words and create tags
        seen = set()
        for word in relevant_words[:10]:  # Limit to 10 tags
            if word not in seen:
                seen.add(word)
                tags.append({
                    "name": word.title(),
                    "confidence": 0.6,
                    "category": "detected"
                })
        
        return tags
    
    def _deduplicate_tags(self, tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tags, keeping highest confidence"""
        unique = {}
        
        for tag in tags:
            name = tag["name"].lower()
            if name not in unique or tag.get("confidence", 0) > unique[name].get("confidence", 0):
                unique[name] = tag
        
        return list(unique.values())
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up BLIP-2 processor")
        
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        
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
        return ["caption", "describe", "vqa", "tag"]