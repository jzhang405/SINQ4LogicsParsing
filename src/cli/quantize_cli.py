#!/usr/bin/env python3
"""
Integrated CLI for SINQ quantization with progress monitoring and error handling.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization_cli.log')
        ]
    )
    return logging.getLogger(__name__)


def get_quantization_configs() -> list:
    """Get available quantization configurations."""
    return [
        # Primary configurations (most stable) - SINQ recommended defaults
        {"name": "4bit_group64", "nbits": 4, "group_size": 64, "tiling_mode": "1D", "method": "sinq"},
        {"name": "8bit_group64", "nbits": 8, "group_size": 64, "tiling_mode": "1D", "method": "sinq"},
        
        # Alternative methods
        {"name": "4bit_asinq", "nbits": 4, "group_size": 64, "tiling_mode": "1D", "method": "asinq"},
        {"name": "8bit_asinq", "nbits": 8, "group_size": 64, "tiling_mode": "1D", "method": "asinq"},
        
        # Other grouped configurations
        {"name": "4bit_group128", "nbits": 4, "group_size": 128, "tiling_mode": "1D", "method": "sinq"},
        {"name": "8bit_group128", "nbits": 8, "group_size": 128, "tiling_mode": "1D", "method": "sinq"},
        
        # 2D tiling
        {"name": "4bit_2d", "nbits": 4, "group_size": 64, "tiling_mode": "2D", "method": "sinq"},
    ]


def load_model_and_processor(model_path: str, device: str = "cpu") -> tuple:
    """Load model and processor with memory optimization."""
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": device,
    }
    
    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, processor


def quantize_with_progress(
    model, 
    processor, 
    config: Dict[str, Any], 
    device: str = "cpu"
) -> bool:
    """Quantize model with progress monitoring."""
    try:
        print(f"\nStarting quantization with config: {config['name']}")
        print(f"Parameters: nbits={config['nbits']}, group_size={config['group_size']}, "
              f"tiling_mode={config['tiling_mode']}, method={config['method']}")
        
        quant_config = BaseQuantizeConfig(
            nbits=config["nbits"],
            group_size=config["group_size"],
            tiling_mode=config["tiling_mode"],
            method=config["method"]
        )
        
        print("Quantizing model layers...")
        start_time = time.time()
        
        # For multimodal models, we need to use the text tokenizer part of the processor
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=processor.tokenizer,  # Use the text tokenizer component
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )
        
        end_time = time.time()
        print(f"✓ Quantization completed in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False


def save_quantized_model(model, output_path: str) -> bool:
    """Save quantized model."""
    try:
        print(f"\nSaving quantized model to {output_path}...")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        print("✓ Quantized model saved successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to save quantized model: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SINQ Quantization CLI for Logics-Parsing Model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/",
        help="Path to the original model directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/",
        help="Path to save the quantized model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        choices=[c["name"] for c in get_quantization_configs()],
        default="8bit_asinq",
        help="Quantization configuration to use"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "auto"],
        default="auto",
        help="Device to use for quantization (auto will use MPS if available)"
    )
    
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available quantization configurations"
    )
    
    parser.add_argument(
        "--logics-parsing",
        action="store_true",
        help="Use Logics-Parsing optimized configuration (CPU-only, 8-bit A-SINQ)"
    )
    
    parser.add_argument(
        "--auto-group",
        action="store_true",
        help="Automatically try different group sizes to find compatible one"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # List configurations if requested
    if args.list_configs:
        print("\nAvailable quantization configurations:")
        for config in get_quantization_configs():
            print(f"  {config['name']}: nbits={config['nbits']}, "
                  f"group_size={config['group_size']}, "
                  f"tiling_mode={config['tiling_mode']}, "
                  f"method={config['method']}")
        return
    
    try:
        print("=" * 60)
        print("SINQ Quantization CLI")
        print("=" * 60)
        
        # Handle Logics-Parsing specific configuration
        if args.logics_parsing:
            print("Using Logics-Parsing optimized configuration:")
            print("  - 8-bit A-SINQ")
            print("  - Group size: 64 (SINQ recommended)")
            print("  - 1D tiling")
            
            # Force CPU for Logics-Parsing configuration to avoid MPS memory issues
            args.device = "cpu"
            print(f"  - Device: {args.device} (forced for memory safety)")
            
            selected_config = {
                "name": "logics_parsing_8bit_asinq",
                "nbits": 8,
                "group_size": 64,
                "tiling_mode": "1D",
                "method": "asinq"
            }
        else:
            # Get selected configuration
            configs = get_quantization_configs()
            selected_config = next(c for c in configs if c["name"] == args.config)
        
        # Load model and processor
        model, processor = load_model_and_processor(args.model_path, args.device)
        
        # Quantize model
        success = quantize_with_progress(model, processor, selected_config, args.device)
        
        if success:
            # Save quantized model
            save_success = save_quantized_model(model, args.output_path)
            
            if save_success:
                print("\n" + "=" * 60)
                print("QUANTIZATION COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print(f"Original model: {args.model_path}")
                print(f"Quantized model: {args.output_path}")
                if args.logics_parsing:
                    print(f"Configuration: Logics-Parsing optimized (8-bit A-SINQ)")
                else:
                    print(f"Configuration: {args.config}")
                print("=" * 60)
            else:
                print("\n✗ Quantization completed but saving failed")
                sys.exit(1)
        else:
            print("\n✗ Quantization failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nQuantization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()