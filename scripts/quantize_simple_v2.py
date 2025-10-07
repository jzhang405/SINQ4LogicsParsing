#!/usr/bin/env python3
"""
Simple quantization script using SINQ with different configuration.
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
            logging.FileHandler('quantization_simple_v2.log')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model with simpler config
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "cpu",
        }
        
        # Try AutoModelForCausalLM first (avoid multimodal complexities)
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
        
        # Try different quantization configurations
        configs_to_try = [
            {"nbits": 4, "group_size": 128, "tiling_mode": "1D", "method": "sinq"},
            {"nbits": 4, "group_size": 64, "tiling_mode": "1D", "method": "sinq"},
            {"nbits": 4, "group_size": None, "tiling_mode": "1D", "method": "sinq"},
        ]
        
        for i, config in enumerate(configs_to_try):
            logger.info(f"Trying config {i+1}: {config}")
            
            try:
                # Reload model for each attempt
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                quant_config = BaseQuantizeConfig(**config)
                
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
                print("QUANTIZATION SUCCESSFUL!")
                print("="*50)
                print(f"Original model: {model_path}")
                print(f"Quantized model: {output_path}")
                print(f"Config used: {config}")
                print("="*50)
                
                # Success - break out of loop
                break
                
            except Exception as e:
                logger.warning(f"Config {i+1} failed: {e}")
                if i == len(configs_to_try) - 1:
                    # Last config failed
                    raise e
                else:
                    continue
        
    except Exception as e:
        logger.error(f"All quantization attempts failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()