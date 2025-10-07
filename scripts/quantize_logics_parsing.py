#!/usr/bin/env python3
"""
Quantize Logics-Parsing model according to design/01.md specifications.
CPU-only implementation optimized for memory constraints.
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
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantize_logics_parsing.log')
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Model paths
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    logger.info(f"Quantizing Logics-Parsing model from: {model_path}")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Load model with CPU-only configuration (design/01.md)
        logger.info("Loading model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Use CPU instead of GPU
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
        
        # Load tokenizer (use tokenizer component, not processor for quantization)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # Optimized quantization config for CPU and memory constraints (design/01.md)
        quant_config = BaseQuantizeConfig(
            nbits=8,            # 8-bit quantization as requested
            group_size=None,    # No grouping to avoid dimension issues
            tiling_mode="1D",   # tiling strategy
            method="asinq"      # A-SINQ method as requested
        )
        
        logger.info("Starting A-SINQ 8-bit quantization...")
        
        # Quantize on CPU (design/01.md)
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,  # Use float16 for better CPU compatibility
            device="cpu"  # Use CPU for quantization
        )
        
        logger.info("A-SINQ 8-bit quantization completed successfully!")
        
        # Save quantized model
        logger.info(f"Saving quantized model to {output_path}")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        logger.info("Quantized model saved successfully!")
        
        # Test loading the quantized model
        logger.info("Testing quantized model loading...")
        qmodel = AutoSINQHFModel.from_quantized(
            output_path,
            device="cpu",  # Load on CPU
            compute_dtype=torch.float16,  # Use float16 for CPU
        )
        logger.info("Quantized model loaded successfully!")
        
        # Quick smoke test
        logger.info("Running quick smoke test...")
        prompt = "What is logical reasoning?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Move to CPU
        with torch.inference_mode():
            out_ids = qmodel.generate(**inputs, max_new_tokens=32, do_sample=False)
        result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        logger.info(f"Test generation: {result}")
        
        print("\n" + "="*60)
        print("LOGICS-PARSING QUANTIZATION SUCCESSFUL!")
        print("="*60)
        print(f"Original model: {model_path}")
        print(f"Quantized model: {output_path}")
        print(f"Quantization: A-SINQ 8-bit")
        print(f"Device: CPU")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        print("\n" + "="*60)
        print("LOGICS-PARSING QUANTIZATION FAILED")
        print("="*60)
        print(f"Error: {e}")
        print("="*60)
        
        sys.exit(1)


if __name__ == "__main__":
    main()