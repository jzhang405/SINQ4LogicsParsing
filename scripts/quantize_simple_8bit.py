#!/usr/bin/env python3
"""
Simple 8-bit quantization script using A-SINQ method.
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization_simple_8bit.log')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    logger.info(f"Loading Qwen2.5-VL model from {model_path}")
    
    try:
        # Load Qwen2.5-VL model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "cpu",
        }
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )
        logger.info("Loaded Qwen2.5-VL model successfully")
        
        # Load tokenizer directly (not processor)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Loaded tokenizer successfully")
        
        # 8-bit A-SINQ quantization
        quant_config = BaseQuantizeConfig(
            nbits=8,
            group_size=None,  # No grouping to avoid dimension issues
            tiling_mode="1D",
            method="asinq"
        )
        
        logger.info("Starting 8-bit A-SINQ quantization...")
        
        # Quantize the model
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device="cpu"
        )
        
        logger.info("8-bit A-SINQ quantization completed successfully!")
        
        # Save quantized model
        logger.info(f"Saving quantized model to {output_path}")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        logger.info("Quantized model saved successfully!")
        
        print("\n" + "="*50)
        print("8-BIT A-SINQ QUANTIZATION SUCCESSFUL!")
        print("="*50)
        print(f"Original model: {model_path}")
        print(f"Quantized model: {output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        print("\n" + "="*50)
        print("QUANTIZATION FAILED")
        print("="*50)
        print(f"Error: {e}")
        print("="*50)
        
        sys.exit(1)


if __name__ == "__main__":
    main()