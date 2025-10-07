"""SINQ quantization service for Logics-Parsing model."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


logger = logging.getLogger(__name__)

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))


try:
    from sinq.patch_model import AutoSINQHFModel
    from sinq.sinqlinear import BaseQuantizeConfig
    SINQ_AVAILABLE = True
except ImportError:
    logger.warning("SINQ library not available. Please install SINQ from /Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
    SINQ_AVAILABLE = False


class SINQQuantizer:
    """SINQ (Sinkhorn-Normalized Quantization) service."""

    def __init__(self):
        """Initialize SINQ quantizer."""
        if not SINQ_AVAILABLE:
            raise ImportError("SINQ library not available. Please install SINQ first.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def quantize(
        self, 
        model: torch.nn.Module, 
        tokenizer,
        bits: int = 4,
        group_size: int = 64,
        tiling_mode: str = "1D",
        method: str = "sinq"
    ) -> torch.nn.Module:
        """
        Quantize model using SINQ method.
        
        Args:
            model: PyTorch model to quantize
            tokenizer: Tokenizer for the model
            bits: Number of bits for quantization (2, 3, 4, 5, 6, 8)
            group_size: Group size for quantization
            tiling_mode: Tiling strategy ("1D", "2D")
            method: Quantization method ("sinq", "asinq")
            
        Returns:
            Quantized model
            
        Raises:
            ValueError: If parameters are invalid
        """
        if bits not in [2, 3, 4, 5, 6, 8]:
            raise ValueError(f"Unsupported bits: {bits}. Supported values: 2, 3, 4, 5, 6, 8")
        
        logger.info(f"Starting SINQ quantization with {bits}-bit precision")
        
        # Create quantization configuration
        quant_config = BaseQuantizeConfig(
            nbits=bits,
            group_size=group_size,
            tiling_mode=tiling_mode,
            method=method
        )
        
        # Quantize the model using SINQ
        quantized_model = AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.bfloat16,
            device=self.device
        )
        
        logger.info("SINQ quantization completed")
        return quantized_model

    def save_quantized_model(
        self, 
        model: torch.nn.Module, 
        save_path: str
    ):
        """
        Save quantized model to disk.
        
        Args:
            model: Quantized model
            save_path: Path to save the model
        """
        AutoSINQHFModel.save_quantized(model, save_path, verbose=True)
        logger.info(f"Quantized model saved to {save_path}")

    def load_quantized_model(
        self, 
        model_path: str,
        compute_dtype: torch.dtype = torch.bfloat16
    ) -> torch.nn.Module:
        """
        Load quantized model from disk.
        
        Args:
            model_path: Path to the quantized model
            compute_dtype: Compute data type
            
        Returns:
            Loaded quantized model
        """
        model = AutoSINQHFModel.from_quantized(
            model_path,
            compute_dtype=compute_dtype,
            device=self.device
        )
        logger.info(f"Quantized model loaded from {model_path}")
        return model

    def calculate_compression_ratio(
        self, 
        original_model: torch.nn.Module, 
        quantized_model: torch.nn.Module
    ) -> float:
        """
        Calculate compression ratio between original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Compression ratio (original_size / quantized_size)
        """
        original_size = self._calculate_model_size(original_model)
        quantized_size = self._calculate_model_size(quantized_model)
        
        if quantized_size == 0:
            return float('inf')
        
        return original_size / quantized_size

    def _calculate_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    def validate_quantization(
        self, 
        original_model: torch.nn.Module, 
        quantized_model: torch.nn.Module,
        test_input: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Validate quantization quality.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_input: Optional test input for forward pass validation
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {}
        
        # Calculate compression ratio
        metrics['compression_ratio'] = self.calculate_compression_ratio(
            original_model, quantized_model
        )
        
        # Calculate parameter difference
        original_params = sum(p.numel() for p in original_model.parameters())
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        metrics['parameter_preservation'] = quantized_params / original_params
        
        # Test forward pass if input provided
        if test_input is not None:
            with torch.no_grad():
                try:
                    original_output = original_model(test_input)
                    quantized_output = quantized_model(test_input)
                    
                    if isinstance(original_output, torch.Tensor) and isinstance(quantized_output, torch.Tensor):
                        mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
                        metrics['mse'] = mse
                        
                        # Calculate similarity (1 - normalized MSE)
                        output_range = original_output.max() - original_output.min()
                        if output_range > 1e-8:
                            normalized_mse = mse / (output_range ** 2)
                            metrics['similarity'] = 1.0 - normalized_mse
                        else:
                            metrics['similarity'] = 1.0
                except Exception as e:
                    logger.warning(f"Forward pass validation failed: {e}")
                    metrics['similarity'] = 0.0
        
        return metrics