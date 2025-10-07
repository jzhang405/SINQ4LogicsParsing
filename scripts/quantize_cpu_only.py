#!/usr/bin/env python3
"""
CPU-only 量化脚本 - 针对16GB内存限制
使用最节省内存的配置
"""

import os
import sys
import logging
import gc
from pathlib import Path

# Add SINQ to Python path
sinq_path = Path("/Users/zhangcz/ws/python/src/github.com/huawei-csl/SINQ")
if sinq_path.exists():
    sys.path.insert(0, str(sinq_path))

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantize_cpu_only.log')
        ]
    )


def load_model_cpu_only(model_path: str):
    """CPU-only 加载模型，最节省内存"""
    print("使用CPU-only模式加载模型...")
    
    # 最节省内存的配置
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
        "offload_folder": "./offload",
        "offload_state_dict": True,  # 卸载状态字典
    }
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    return model


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 模型路径
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    print("=" * 60)
    print("CPU-only 量化 - 16GB内存限制")
    print("=" * 60)
    
    try:
        # 1. 加载处理器
        print("1. 加载处理器...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. CPU-only 加载模型
        print("2. CPU-only 加载模型...")
        model = load_model_cpu_only(model_path)
        
        # 3. 量化配置
        config = {
            "nbits": 8,           # 8-bit量化
            "group_size": None,   # 无分组避免维度问题
            "tiling_mode": "1D",  # 1D平铺
            "method": "asinq"     # A-SINQ方法
        }
        
        print(f"3. 使用配置: {config}")
        
        # 4. 创建量化配置
        quant_config = BaseQuantizeConfig(
            nbits=config["nbits"],
            group_size=config["group_size"],
            tiling_mode=config["tiling_mode"],
            method=config["method"]
        )
        
        # 5. 量化
        print("4. 开始量化...")
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=processor.tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device="cpu"
        )
        
        print("✓ 量化完成!")
        
        # 6. 保存量化模型
        print("5. 保存量化模型...")
        AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
        
        print("✓ 模型保存完成!")
        
        # 7. 清理内存
        print("6. 清理内存...")
        del model
        gc.collect()
        
        # 8. 测试加载量化模型
        print("7. 测试加载量化模型...")
        qmodel = AutoSINQHFModel.from_quantized(
            output_path,
            device="cpu",
            compute_dtype=torch.float16,
        )
        
        print("✓ 量化模型加载成功!")
        
        # 9. 简单测试
        print("8. 运行简单测试...")
        prompt = "What is logical reasoning?"
        inputs = processor.tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            out_ids = qmodel.generate(**inputs, max_new_tokens=20, do_sample=False)
        result = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(f"测试结果: {result}")
        
        print("\n" + "=" * 60)
        print("CPU-only 量化成功完成!")
        print("=" * 60)
        print(f"原始模型: {model_path}")
        print(f"量化模型: {output_path}")
        print(f"量化配置: 8-bit A-SINQ")
        print(f"内存使用: CPU-only 模式")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()