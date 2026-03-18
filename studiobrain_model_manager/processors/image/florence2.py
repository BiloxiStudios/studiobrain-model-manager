"""
Florence-2 Image Processor
Microsoft's vision-language model for comprehensive image understanding
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from PIL import Image
import re
import os

from studiobrain_model_manager.processors.base import BaseProcessor

# Force to use only RTX 4090 (GPU 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def move_inputs_to_device(inputs, device=0, debug=False):
    """Move inputs to device with selective dtype conversion"""
    processed_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            original_dtype = v.dtype
            # Keep integer tensors as their original dtype for embeddings
            if v.dtype in [torch.int32, torch.int64, torch.long, torch.int16, torch.int8]:
                processed_inputs[k] = v.to(device=device)  # Keep original dtype
                if debug:
                    logger.info(f"Input '{k}': {original_dtype} -> cuda:{device} (kept integer type)")
            else:
                # Convert float tensors to float16 to match model
                processed_inputs[k] = v.to(device=device, dtype=torch.float16)
                if debug:
                    logger.info(f"Input '{k}': {original_dtype} -> cuda:{device} float16")
        else:
            processed_inputs[k] = v
            if debug:
                logger.info(f"Input '{k}': non-tensor (kept as-is)")
    return processed_inputs

logger = logging.getLogger(__name__)

class Florence2Processor(BaseProcessor):
    """Florence-2 processor for image analysis"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.model_id = "microsoft/Florence-2-large"
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    async def load_model(self):
        """Load Florence-2 model"""
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
            
            logger.info(f"Loading Florence-2 model from {self.model_id}")
            logger.info(f"Cache directory: {self.settings.model_cache_dir}")
            
            from transformers import AutoProcessor, AutoModelForCausalLM
            from huggingface_hub import snapshot_download
            
            # Ensure cache directory exists
            self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load from cache first, fallback to download if needed
            logger.info("Loading processor from cache...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    cache_dir=self.settings.model_cache_dir,
                    local_files_only=True  # Try cache-only first
                )
                logger.info("Processor loaded from cache successfully")
            except Exception as cache_error:
                logger.warning(f"Cache load failed, downloading: {cache_error}")
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    cache_dir=self.settings.model_cache_dir,
                    local_files_only=False,  # Allow downloading
                    resume_download=True,
                    force_download=False
                )
            
            # Load model with appropriate settings
            try:
                if self.settings.gpu_available:
                    # Try cache-first approach for faster loading
                    try:
                        logger.info("Attempting to load model from cache...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map={"": 0},  # Use device index 0 for RTX 4090
                            cache_dir=self.settings.model_cache_dir,
                            attn_implementation="eager",  # Fix for _supports_sdpa error
                            local_files_only=True  # Cache only
                        )
                        logger.info("Model loaded from cache on cuda:0 (RTX 4090)")
                    except Exception as cache_error:
                        logger.warning(f"Cache load failed, downloading model: {cache_error}")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map={"": 0},  # Use device index 0 for RTX 4090
                            cache_dir=self.settings.model_cache_dir,
                            attn_implementation="eager",  # Fix for _supports_sdpa error
                            local_files_only=False,
                            resume_download=True,
                            force_download=False
                        )
                        logger.info("Model downloaded and loaded on cuda:0 (RTX 4090)")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        cache_dir=self.settings.model_cache_dir,
                        attn_implementation="eager",  # Fix for _supports_sdpa error
                        local_files_only=False,
                        resume_download=True,
                        force_download=False
                    )
                    self.model = self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Failed with eager attention, trying without: {e}")
                # Fallback without attn_implementation
                if self.settings.gpu_available:
                    # Force to use CUDA:0 (RTX 4090) - use device index
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map={"": 0},  # Use device index 0 for RTX 4090
                        cache_dir=self.settings.model_cache_dir,
                        local_files_only=False,
                        resume_download=True,
                        force_download=False
                    )
                    logger.info("Model loaded on cuda:0 (RTX 4090) - fallback mode")
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        cache_dir=self.settings.model_cache_dir,
                        local_files_only=False,
                        resume_download=True,
                        force_download=False
                    )
                    self.model = self.model.to(self.device)
                
                # Monkey patch the _supports_sdpa attribute if needed
                if not hasattr(self.model, '_supports_sdpa'):
                    self.model._supports_sdpa = False
            
            self.model.eval()
            self.model_loaded = True
            
            logger.info("Florence-2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            return False
    
    async def process(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process image with Florence-2"""
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
            
            # Process different tasks based on options
            if options.get("enable_caption", True):
                # Generate caption
                caption = await self._generate_caption(image)
                if caption:
                    descriptions["short"] = caption
            
            if options.get("enable_detailed_caption", True):
                # Generate detailed caption
                detailed = await self._generate_detailed_caption(image)
                if detailed:
                    descriptions["detailed"] = detailed
                    # Extract tags from detailed caption
                    extracted_tags = self._extract_tags_from_text(detailed)
                    tags.extend(extracted_tags)
            
            if options.get("enable_tags", True):
                # Generate more detailed caption for tag extraction
                more_detailed = await self._generate_more_detailed_caption(image)
                if more_detailed:
                    descriptions["technical"] = more_detailed
                    # Extract additional tags
                    additional_tags = self._extract_tags_from_text(more_detailed)
                    tags.extend(additional_tags)
            
            if options.get("enable_detect", False):
                # Object detection
                objects = await self._detect_objects(image)
                if objects:
                    features["detected_objects"] = objects
                    # Add object names as tags
                    for obj in objects:
                        tags.append({
                            "name": obj["label"],
                            "confidence": obj["confidence"],
                            "category": "object"
                        })
            
            if options.get("enable_ocr", False):
                # OCR text extraction
                text = await self._extract_text(image)
                if text:
                    features["extracted_text"] = text
            
            # Deduplicate and format tags
            unique_tags = self._deduplicate_tags(tags)
            
            # Apply confidence threshold
            threshold = options.get("confidence_threshold", 0.5)
            filtered_tags = [tag for tag in unique_tags if tag.get("confidence", 1.0) >= threshold]
            
            return self.format_result(
                tags=filtered_tags,
                descriptions=descriptions,
                metadata=metadata,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Error processing image with Florence-2: {e}")
            return self.format_result(error=str(e))
    
    async def _generate_caption(self, image: Image) -> Optional[str]:
        """Generate a short caption"""
        try:
            task_prompt = "<CAPTION>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            if self.settings.gpu_available:
                # Move inputs to cuda:0 (RTX 4090) with selective dtype conversion
                inputs = move_inputs_to_device(inputs, device=0, debug=True)
            
            with torch.no_grad():
                logger.info(f"Generating caption with inputs: {list(inputs.keys())}")
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                    
                    logger.info(f"Generated IDs type: {type(generated_ids)}")
                    logger.info(f"Generated IDs is None: {generated_ids is None}")
                    
                    if generated_ids is None:
                        logger.error("Model.generate() returned None")
                        return None
                    
                    logger.info(f"Generated IDs shape: {generated_ids.shape}")
                    
                except Exception as gen_error:
                    logger.error(f"Generation failed: {gen_error}")
                    return None
            
            if generated_ids is None:
                logger.error("generated_ids is None after generation")
                return None
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the output
            caption = generated_text.replace(task_prompt, "").strip()
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return None
    
    async def _generate_detailed_caption(self, image: Image) -> Optional[str]:
        """Generate a detailed caption"""
        try:
            task_prompt = "<DETAILED_CAPTION>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            if self.settings.gpu_available:
                # Move inputs to cuda:0 (RTX 4090) with selective dtype conversion
                inputs = move_inputs_to_device(inputs, device=0, debug=True)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for detailed caption")
                    return None
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the output
            detailed = generated_text.replace(task_prompt, "").strip()
            return detailed
            
        except Exception as e:
            logger.error(f"Error generating detailed caption: {e}")
            return None
    
    async def _generate_more_detailed_caption(self, image: Image) -> Optional[str]:
        """Generate an even more detailed caption for tag extraction"""
        try:
            task_prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            if self.settings.gpu_available:
                # Move inputs to cuda:0 (RTX 4090) with selective dtype conversion
                inputs = move_inputs_to_device(inputs, device=0, debug=True)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for more detailed caption")
                    return None
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the output
            more_detailed = generated_text.replace(task_prompt, "").strip()
            return more_detailed
            
        except Exception as e:
            logger.error(f"Error generating more detailed caption: {e}")
            return None
    
    async def _detect_objects(self, image: Image) -> Optional[List[Dict[str, Any]]]:
        """Detect objects in the image"""
        try:
            task_prompt = "<OD>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            if self.settings.gpu_available:
                # Move inputs to cuda:0 (RTX 4090) with selective dtype conversion
                inputs = move_inputs_to_device(inputs, device=0, debug=True)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for object detection")
                    return None
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse the object detection output
            objects = self._parse_detection_output(generated_text.replace(task_prompt, "").strip())
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return None
    
    async def _extract_text(self, image: Image) -> Optional[str]:
        """Extract text from image using OCR"""
        try:
            task_prompt = "<OCR>"
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            
            if self.settings.gpu_available:
                # Move inputs to cuda:0 (RTX 4090) with selective dtype conversion
                inputs = move_inputs_to_device(inputs, device=0, debug=True)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
                
                if generated_ids is None:
                    logger.error("Model.generate() returned None for OCR text extraction")
                    return None
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the output
            text = generated_text.replace(task_prompt, "").strip()
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return None
    
    def _extract_tags_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tags from generated text"""
        tags = []
        
        if not text:
            return tags
        
        # Common patterns to extract
        # Objects and items
        object_words = re.findall(r'\b(?:a |an |the )([a-z]+)\b', text.lower())
        for word in object_words:
            if len(word) > 2:  # Skip very short words
                tags.append({
                    "name": word,
                    "confidence": 0.7,
                    "category": "object"
                })
        
        # Colors
        color_pattern = r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|cyan|magenta)\b'
        colors = re.findall(color_pattern, text.lower())
        for color in set(colors):
            tags.append({
                "name": color,
                "confidence": 0.8,
                "category": "color"
            })
        
        # Descriptive adjectives
        adj_pattern = r'\b(bright|dark|large|small|big|tiny|modern|vintage|futuristic|old|new|shiny|matte|glossy)\b'
        adjectives = re.findall(adj_pattern, text.lower())
        for adj in set(adjectives):
            tags.append({
                "name": adj,
                "confidence": 0.6,
                "category": "style"
            })
        
        # Environment/scene words
        scene_pattern = r'\b(indoor|outdoor|city|urban|rural|nature|forest|desert|ocean|mountain|sky|night|day|sunset|sunrise)\b'
        scenes = re.findall(scene_pattern, text.lower())
        for scene in set(scenes):
            tags.append({
                "name": scene,
                "confidence": 0.7,
                "category": "scene"
            })
        
        return tags
    
    def _parse_detection_output(self, text: str) -> List[Dict[str, Any]]:
        """Parse object detection output"""
        objects = []
        
        # Florence-2 outputs detections in a specific format
        # Parse and extract object labels and bounding boxes
        # This is a simplified parser - adjust based on actual output format
        
        lines = text.split('\n')
        for line in lines:
            if '<' in line and '>' in line:
                # Extract label between < >
                match = re.search(r'<([^>]+)>', line)
                if match:
                    label = match.group(1)
                    objects.append({
                        "label": label.lower(),
                        "confidence": 0.8  # Florence-2 doesn't always provide confidence
                    })
        
        return objects
    
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
        logger.info("Cleaning up Florence-2 processor")
        
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
        return ["caption", "tag", "detect", "ocr", "detailed_caption"]