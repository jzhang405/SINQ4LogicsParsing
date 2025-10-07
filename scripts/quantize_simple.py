#!/usr/bin/env python3
"""
Simple quantization script using SINQ directly.
Based on SINQ's test code pattern.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization_simple.log')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model - try different approaches
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "cpu",  # Use CPU to avoid MPS memory issues
        }
        
        # Try to load as multimodal model first
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                **model_kwargs
            )
            logger.info("Loaded as Qwen2.5-VL model")
        except Exception as e:
            logger.warning(f"Failed to load as Qwen2.5-VL: {e}")
            # Fall back to AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info("Loaded as AutoModelForCausalLM")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully")
        
        # Quantization configuration
        quant_config = BaseQuantizeConfig(
            nbits=4, 
            group_size=64, 
            axis=1, 
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
            device="cpu"  # Use CPU for quantization
        )
        
        logger.info("SINQ quantization completed")
        
        # Save quantized model
        logger.info(f"Saving quantized model to {output_path}")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        logger.info("Quantized model saved successfully!")
        
        print("\n" + "="*50)
        print("QUANTIZATION SUCCESSFUL!")
        print("="*50)
        print(f"Original model: {model_path}")
        print(f"Quantized model: {output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()