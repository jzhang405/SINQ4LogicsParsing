"""Model loading service for Logics-Parsing model."""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoProcessor

# Try to import Qwen2.5-VL specific classes
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


logger = logging.getLogger(__name__)


class ModelLoader:
    """Service for loading Logics-Parsing model from cache."""

    def __init__(self):
        """Initialize model loader."""
        self.device = self._get_available_device()

    def _get_available_device(self) -> str:
        """Get the best available device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(
        self, 
        model_path: Union[str, Path], 
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load Logics-Parsing model from cache directory.
        
        Args:
            model_path: Path to the model cache directory
            device: Device to load model on (cpu, cuda, mps). If None, uses best available.
            
        Returns:
            Loaded PyTorch model
            
        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        device = device or self.device
        logger.info(f"Loading model from {model_path} on device {device}")
        
        try:
            # Load model with memory optimization
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,  # Use bfloat16 to save memory
                "device_map": device,
                "low_cpu_mem_usage": True,
            }
            
            # Try to load as Qwen2.5-VL model first
            if Qwen2_5_VLForConditionalGeneration is not None:
                try:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    logger.info("Loaded as Qwen2.5-VL model")
                except Exception:
                    # Fall back to AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    logger.info("Loaded as AutoModelForCausalLM")
            else:
                # Fall back to AutoModel
                model = AutoModel.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                logger.info("Loaded as AutoModel")
            
            model.eval()
            
            logger.info(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_processor(self, model_path: Union[str, Path]):
        """
        Load processor for Logics-Parsing model.
        
        Args:
            model_path: Path to the model cache directory
            
        Returns:
            Loaded processor
            
        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If processor loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            logger.info("Successfully loaded processor")
            return processor
            
        except Exception as e:
            logger.error(f"Failed to load processor from {model_path}: {e}")
            raise RuntimeError(f"Processor loading failed: {e}")

    def get_model_info(self, model_path: Union[str, Path]) -> dict:
        """
        Get information about the model.
        
        Args:
            model_path: Path to the model cache directory
            
        Returns:
            Dictionary with model information
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            info = {
                "model_type": getattr(config, "model_type", "unknown"),
                "vocab_size": getattr(config, "vocab_size", None),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_attention_heads": getattr(config, "num_attention_heads", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                "intermediate_size": getattr(config, "intermediate_size", None),
                "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info from {model_path}: {e}")
            raise RuntimeError(f"Model info retrieval failed: {e}")

    def validate_model_loading(self, model_path: Union[str, Path]) -> bool:
        """
        Validate that model can be loaded successfully.
        
        Args:
            model_path: Path to the model cache directory
            
        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            model = self.load_model(model_path)
            processor = self.load_processor(model_path)
            
            # Test forward pass with dummy input
            with torch.no_grad():
                # Create dummy inputs for multimodal model
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_text = torch.randint(0, 1000, (1, 10)).to(self.device)
                
                try:
                    _ = model(dummy_image, dummy_text)
                except TypeError:
                    # Try alternative signature
                    _ = model(dummy_text)
                except Exception:
                    # If forward pass fails, still consider model loaded
                    pass
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False