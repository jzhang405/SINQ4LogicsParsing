#!/usr/bin/env python3
"""
Low-memory CLI script for quantizing Logics-Parsing model using SINQ.
This version uses CPU for model loading and quantization to avoid MPS memory issues.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.pipeline import QuantizationPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization_low_memory.log')
        ]
    )


def main():
    """Main function for CLI script."""
    parser = argparse.ArgumentParser(
        description="Quantize Logics-Parsing model using SINQ (low memory version)"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str,
        required=True,
        help="Path to the original model directory"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str,
        default="/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/",
        help="Path to save the quantized model"
    )
    
    parser.add_argument(
        "--bits", 
        type=int,
        choices=[2, 3, 4, 5, 6, 8],
        default=4,
        help="Quantization bit precision"
    )
    
    parser.add_argument(
        "--group-size", 
        type=int,
        choices=[64, 128],
        default=64,
        help="Group size for quantization"
    )
    
    parser.add_argument(
        "--tiling-mode", 
        type=str,
        choices=["1D", "2D"],
        default="1D",
        help="Tiling strategy"
    )
    
    parser.add_argument(
        "--method", 
        type=str,
        choices=["sinq", "asinq"],
        default="sinq",
        help="Quantization method"
    )
    
    parser.add_argument(
        "--device", 
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device to use for quantization (use 'cpu' for low memory)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline with specified device
        logger.info(f"Using device: {args.device}")
        
        # Import and override device settings
        import torch
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        # Create custom model loader with specified device
        from services.model_loader import ModelLoader
        
        class LowMemoryModelLoader(ModelLoader):
            def __init__(self, device="cpu"):
                self.device = device
                
            def _get_available_device(self) -> str:
                return self.device
        
        # Create custom pipeline
        from services.quantizer import SINQQuantizer
        
        class LowMemoryQuantizationPipeline:
            def __init__(self, device="cpu"):
                self.model_loader = LowMemoryModelLoader(device)
                self.quantizer = SINQQuantizer()
                # Override quantizer device
                self.quantizer.device = torch.device(device)
            
            def quantize_model(
                self,
                model_path,
                bits=4,
                output_path=None,
                group_size=64,
                tiling_mode="1D",
                method="sinq"
            ):
                logger.info(f"Starting low-memory quantization pipeline for {model_path}")
                
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
                
                logger.info("Low-memory quantization pipeline completed")
                return quantized_model
        
        # Initialize pipeline
        pipeline = LowMemoryQuantizationPipeline(device=args.device)
        
        # Run quantization
        logger.info("Starting low-memory quantization...")
        
        quantized_model = pipeline.quantize_model(
            model_path=Path(args.model_path),
            bits=args.bits,
            output_path=Path(args.output_path),
            group_size=args.group_size,
            tiling_mode=args.tiling_mode,
            method=args.method
        )
        
        logger.info("Low-memory quantization completed successfully!")
        logger.info(f"Quantized model saved to: {args.output_path}")
        
        print("\n" + "="*50)
        print("LOW-MEMORY QUANTIZATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Quantized model saved to: {args.output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Low-memory quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()