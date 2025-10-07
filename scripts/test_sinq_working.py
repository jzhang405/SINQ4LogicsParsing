#!/usr/bin/env python3
"""
测试SINQ是否正常工作 - 使用SINQ示例中的配置
"""

import os
import sys
from pathlib import Path

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def main():
    print("=" * 50)
    print("SINQ 功能测试")
    print("=" * 50)
    
    # 使用SINQ README中的示例配置
    model_name = "Qwen/Qwen3-1.7B"
    output_path = "./test_qwen_quantized/"
    
    try:
        print(f"1. 加载模型: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("2. 加载tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("3. 配置量化参数 (SINQ推荐配置)")
        quant_config = BaseQuantizeConfig(
            nbits=4,            # 4-bit quantization
            group_size=64,      # SINQ推荐值
            tiling_mode="1D",   # tiling strategy
            method="sinq"       # quantization method
        )
        
        print("4. 开始量化...")
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device="cpu"
        )
        
        print("✓ 量化完成!")
        
        print("5. 保存量化模型...")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        print("✓ 模型保存完成!")
        
        print("6. 测试加载量化模型...")
        qmodel = AutoSINQHFModel.from_quantized(
            output_path,
            device="cpu",
            compute_dtype=torch.float16,
        )
        
        print("✓ 量化模型加载成功!")
        
        print("7. 运行测试...")
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            out_ids = qmodel.generate(**inputs, max_new_tokens=20, do_sample=False)
        result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(f"测试结果: {result}")
        
        print("\n" + "=" * 50)
        print("SINQ测试成功!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()