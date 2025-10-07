"""Quantization pipeline service."""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from .model_loader import ModelLoader
from .quantizer import SINQQuantizer


logger = logging.getLogger(__name__)


class QuantizationPipeline:
    """Pipeline for quantizing Logics-Parsing model using SINQ."""

    def __init__(self):
        """Initialize quantization pipeline."""
        self.model_loader = ModelLoader()
        self.quantizer = SINQQuantizer()

    def quantize_model(
        self,
        model_path: Union[str, Path],
        bits: int = 4,
        output_path: Optional[Union[str, Path]] = None,
        group_size: int = 64,
        tiling_mode: str = "1D",
        method: str = "sinq"
    ) -> torch.nn.Module:
        """
        Run complete quantization pipeline.
        
        Args:
            model_path: Path to original model
            bits: Quantization bits (2, 3, 4, 5, 6, 8)
            output_path: Path to save quantized model
            group_size: Group size for quantization
            tiling_mode: Tiling strategy ("1D", "2D")
            method: Quantization method ("sinq", "asinq")
            
        Returns:
            Quantized model
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If quantization fails
        """
        logger.info(f"Starting quantization pipeline for {model_path}")
        
        # Step 1: Load original model and tokenizer
        logger.info("Step 1: Loading original model and tokenizer")
        original_model = self.model_loader.load_model(model_path)
        tokenizer = self.model_loader.load_processor(model_path)
        
        # Step 2: Quantize model using SINQ
        logger.info("Step 2: Quantizing model with SINQ")
        quantized_model = self.quantizer.quantize(
            original_model, 
            tokenizer=tokenizer,
            bits=bits, 
            group_size=group_size,
            tiling_mode=tiling_mode,
            method=method
        )
        
        # Step 3: Save quantized model
        if output_path:
            logger.info("Step 3: Saving quantized model")
            self.quantizer.save_quantized_model(quantized_model, str(output_path))
        
        # Step 4: Validate quantization
        logger.info("Step 4: Validating quantization")
        validation_results = self.quantizer.validate_quantization(
            original_model, quantized_model
        )
        
        logger.info(f"Quantization pipeline completed: {validation_results}")
        return quantized_model

    def save_quantized_model(
        self,
        model: torch.nn.Module,
        output_path: Union[str, Path],
        original_model_path: Union[str, Path]
    ):
        """
        Save quantized model to disk.
        
        Args:
            model: Quantized model
            output_path: Path to save quantized model
            original_model_path: Path to original model for config
            
        Raises:
            RuntimeError: If saving fails
        """
        output_path = Path(output_path)
        original_model_path = Path(original_model_path)
        
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), output_path / "pytorch_model.bin")
            
            # Copy config from original model
            config_files = ["config.json", "generation_config.json", "preprocessor_config.json"]
            for config_file in config_files:
                src_config = original_model_path / config_file
                if src_config.exists():
                    shutil.copy2(src_config, output_path / config_file)
            
            # Save quantization info
            quantization_info = {
                "quantization_method": "SINQ",
                "model_type": "Logics-Parsing",
                "original_model_path": str(original_model_path),
                "quantization_date": str(torch.datetime.now())
            }
            
            # Save quantization info (you might want to use a proper serialization)
            with open(output_path / "quantization_info.txt", "w") as f:
                for key, value in quantization_info.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Quantized model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save quantized model: {e}")
            raise RuntimeError(f"Model saving failed: {e}")

    def load_quantized_model(
        self, 
        model_path: Union[str, Path],
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load quantized model from disk.
        
        Args:
            model_path: Path to quantized model
            device: Device to load model on
            
        Returns:
            Loaded quantized model
            
        Raises:
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If loading fails
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Quantized model directory not found: {model_path}")
        
        try:
            # Load original model structure first
            original_model_path = self._get_original_model_path(model_path)
            original_model = self.model_loader.load_model(original_model_path)
            
            # Load quantized state dict
            state_dict_path = model_path / "pytorch_model.bin"
            if not state_dict_path.exists():
                raise FileNotFoundError(f"Model weights not found: {state_dict_path}")
            
            state_dict = torch.load(state_dict_path, map_location='cpu')
            
            # Load state dict into model structure
            original_model.load_state_dict(state_dict)
            
            # Move to device
            device = device or self.model_loader.device
            original_model = original_model.to(device)
            original_model.eval()
            
            logger.info(f"Quantized model loaded from {model_path}")
            return original_model
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            raise RuntimeError(f"Quantized model loading failed: {e}")

    def _get_original_model_path(self, quantized_model_path: Path) -> Path:
        """
        Get original model path from quantization info.
        
        Args:
            quantized_model_path: Path to quantized model
            
        Returns:
            Path to original model
            
        Raises:
            FileNotFoundError: If original model path cannot be determined
        """
        # Try to read from quantization info
        info_path = quantized_model_path / "quantization_info.txt"
        if info_path.exists():
            with open(info_path, 'r') as f:
                for line in f:
                    if line.startswith("original_model_path:"):
                        original_path = line.split(":", 1)[1].strip()
                        if Path(original_path).exists():
                            return Path(original_path)
        
        # Fallback: try to infer from naming convention
        if "Logics-Parsing-SINQ" in str(quantized_model_path):
            original_path = str(quantized_model_path).replace("-SINQ", "")
            if Path(original_path).exists():
                return Path(original_path)
        
        raise FileNotFoundError(
            f"Cannot determine original model path for {quantized_model_path}"
        )

    def run_comprehensive_pipeline(
        self,
        model_path: Union[str, Path],
        bits: int = 4,
        output_path: Optional[Union[str, Path]] = None,
        validate: bool = True
    ) -> Dict:
        """
        Run comprehensive quantization pipeline with validation.
        
        Args:
            model_path: Path to original model
            bits: Quantization bits
            output_path: Path to save quantized model
            validate: Whether to run validation
            
        Returns:
            Dictionary with pipeline results
        """
        results = {}
        
        try:
            # Load original model
            original_model = self.model_loader.load_model(model_path)
            results['original_model_info'] = self.model_loader.get_model_info(model_path)
            
            # Quantize model
            quantized_model = self.quantize_model(model_path, bits, output_path)
            
            # Run validation if requested
            if validate:
                validation_results = self.quantizer.validate_quantization(
                    original_model, quantized_model
                )
                results['validation_results'] = validation_results
                
                # Calculate compression metrics
                original_size = self._calculate_model_size(original_model)
                quantized_size = self._calculate_model_size(quantized_model)
                results['compression_metrics'] = {
                    'original_size_mb': original_size / (1024 * 1024),
                    'quantized_size_mb': quantized_size / (1024 * 1024),
                    'size_reduction': (original_size - quantized_size) / original_size,
                    'compression_ratio': original_size / quantized_size
                }
            
            results['success'] = True
            results['message'] = "Quantization pipeline completed successfully"
            
        except Exception as e:
            logger.error(f"Comprehensive pipeline failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            results['message'] = "Quantization pipeline failed"
        
        return results

    def _calculate_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size