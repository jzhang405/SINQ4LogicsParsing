#!/usr/bin/env python3
"""
Test SINQ quantization on a small standard model first.
"""

import os
import sys
import logging
from pathlib import Path

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_sinq_small.log')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Use a small standard model for testing
    model_name = "microsoft/DialoGPT-small"  # Small model for quick testing
    output_path = "./test_quantized_model/"
    
    logger.info(f"Testing SINQ with small model: {model_name}")
    
    try:
        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "cpu",
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        logger.info("Loaded model successfully")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Loaded tokenizer successfully")
        
        # 4-bit quantization
        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=None,  # No grouping
            tiling_mode="1D",
            method="sinq"
        )
        
        logger.info("Starting SINQ quantization...")
        
        # Quantize the model
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device="cpu"
        )
        
        logger.info("SINQ quantization completed successfully!")
        
        # Save quantized model
        logger.info(f"Saving quantized model to {output_path}")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        logger.info("Quantized model saved successfully!")
        
        print("\n" + "="*50)
        print("SINQ TEST SUCCESSFUL!")
        print("="*50)
        print(f"Original model: {model_name}")
        print(f"Quantized model: {output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"SINQ test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        print("\n" + "="*50)
        print("SINQ TEST FAILED")
        print("="*50)
        print(f"Error: {e}")
        print("="*50)
        
        sys.exit(1)


if __name__ == "__main__":
    main()