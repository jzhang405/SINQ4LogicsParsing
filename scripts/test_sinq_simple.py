#!/usr/bin/env python3
"""
简单测试SINQ量化 - 使用小模型验证功能
"""

import os
import sys
from pathlib import Path

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def main():
    print("=" * 50)
    print("简单SINQ量化测试")
    print("=" * 50)
    
    # 使用小模型测试
    model_name = "microsoft/DialoGPT-small"
    output_path = "./test_quantized/"
    
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
        
        print("3. 配置量化参数")
        quant_config = BaseQuantizeConfig(
            nbits=8,
            group_size=None,
            tiling_mode="1D",
            method="asinq"
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