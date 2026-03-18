"""
Base processor interface for all AI models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Base class for all asset processors"""

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.processor = None
        # Force CUDA:0 to avoid RTX 5090 compatibility issues
        if settings.gpu_available:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.model_loaded = False

    @abstractmethod
    async def load_model(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    async def process(self, file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process an asset and return results"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this processor supports"""
        return []

    def validate_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize processing options"""
        return options or {}

    def format_result(self,
                     tags: List[Dict[str, Any]] = None,
                     descriptions: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None,
                     features: Dict[str, Any] = None,
                     error: str = None) -> Dict[str, Any]:
        """Format the processing result in standard format"""
        result = {
            "success": error is None,
            "processor": self.__class__.__name__
        }

        if error:
            result["error"] = error
        else:
            if tags is not None:
                result["tags"] = tags
            if descriptions is not None:
                result["descriptions"] = descriptions
            if metadata is not None:
                result["metadata"] = metadata
            if features is not None:
                result["features"] = features

        return result
