"""
Embedding Service for generating text embeddings using transformers
"""

import logging
import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Warning: sentence-transformers not available ({type(e).__name__}), using fallback embeddings")
    # Define dummy class for fallback
    SentenceTransformer = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available, using basic embeddings")

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    Handles model loading, caching, and batch processing.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.model: Optional[SentenceTransformer] = None
        self.model_name = settings.embedding_model
        self.device = settings.gpu_device if settings.gpu_available else "cpu"
        self.embedding_dimension = settings.embedding_dimension
        
    async def load(self) -> bool:
        """Load the embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading sentence-transformers model: {self.model_name}")

                loop = asyncio.get_event_loop()
                cache_folder = str(self.settings.model_cache_dir)

                # Try local cache first to avoid network calls on every startup
                try:
                    self.model = await loop.run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            self.model_name,
                            device=self.device,
                            cache_folder=cache_folder,
                            local_files_only=True
                        )
                    )
                    logger.info("Embedding model loaded from local cache")
                except Exception:
                    logger.info("Model not in local cache, downloading from HuggingFace")
                    self.model = await loop.run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            self.model_name,
                            device=self.device,
                            cache_folder=cache_folder
                        )
                    )
                    logger.info("Embedding model downloaded and cached")

                logger.info(f"Embedding model loaded successfully on {self.device}")
                logger.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            else:
                logger.info("Using fallback basic embedding approach")
                self.model = "fallback"  # Signal for fallback mode
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}, falling back to basic embeddings")
            self.model = "fallback"
            return True  # Return True so service can continue with fallback
    
    async def unload(self):
        """Unload the model to free memory"""
        if self.model is not None:
            logger.info("Unloading embedding model")
            del self.model
            self.model = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self.model:
            logger.error("Embedding model not loaded")
            return None
        
        try:
            if self.model == "fallback":
                # Use fallback embedding method
                return self._fallback_embed_text(text)
            
            # Generate embedding in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode([text], convert_to_numpy=True)
            )
            
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        if not self.model:
            logger.error("Embedding model not loaded")
            return None

        if not texts:
            return np.array([])

        # Fallback mode: generate basic embeddings without sentence-transformers
        if self.model == "fallback":
            logger.debug("embed_batch: using fallback embeddings for %d texts", len(texts))
            embeddings = np.array([self._fallback_embed_text(t) for t in texts], dtype=np.float32)
            return embeddings

        try:
            # Process in batches to manage memory
            batch_size = min(self.settings.max_batch_size, 32)  # Reasonable batch size for embeddings
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing embedding batch {i//batch_size + 1}, size: {len(batch)}")

                # Generate embeddings in executor
                loop = asyncio.get_event_loop()
                # Capture batch in a local variable to avoid closure over loop variable
                _batch = batch
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda b=_batch: self.model.encode(b, convert_to_numpy=True, show_progress_bar=False)
                )

                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch results
            result = np.vstack(all_embeddings) if all_embeddings else np.array([])
            logger.info(f"Generated {len(result)} embeddings successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return None
    
    async def similarity_search(self, query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Database of embeddings to search
            top_k: Number of most similar results to return
            
        Returns:
            List of dictionaries with similarity scores and indices
        """
        try:
            # Calculate cosine similarity
            similarities = np.dot(embeddings, query_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k most similar indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "index": int(idx),
                    "similarity": float(similarities[idx]),
                    "score": float(similarities[idx])  # Alias for compatibility
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "not_loaded"}
        
        if self.model == "fallback":
            return {
                "status": "loaded",
                "model_name": "fallback_basic",
                "device": "cpu",
                "embedding_dimension": self.embedding_dimension,
                "max_sequence_length": "unlimited"
            }
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    def _fallback_embed_text(self, text: str) -> np.ndarray:
        """
        Fallback embedding method using basic text features
        Creates a simple embedding based on text characteristics
        """
        # Simple embedding based on text features
        text_lower = text.lower()
        
        # Create feature vector
        features = []
        
        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(text.split()) / 100.0)  # Normalized word count
        
        # Character frequency features (common letters)
        common_chars = 'abcdefghijklmnopqrstuvwxyz'
        for char in common_chars:
            features.append(text_lower.count(char) / max(len(text), 1))
        
        # Common word features
        common_words = [
            'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i',
            'character', 'location', 'brand', 'description', 'personality', 'background',
            'age', 'name', 'voice', 'appearance', 'skills', 'equipment', 'history'
        ]
        
        words = text_lower.split()
        for word in common_words:
            features.append(words.count(word) / max(len(words), 1))
        
        # Text hash features (for uniqueness)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        for i in range(0, min(32, len(text_hash)), 2):
            features.append(int(text_hash[i:i+2], 16) / 255.0)
        
        # Pad or truncate to target dimension
        while len(features) < self.embedding_dimension:
            features.append(0.0)
        
        features = features[:self.embedding_dimension]
        
        return np.array(features, dtype=np.float32)

# Singleton instance for global access
_embedding_service: Optional[EmbeddingService] = None

def get_embedding_service(settings) -> EmbeddingService:
    """Get or create the singleton embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service