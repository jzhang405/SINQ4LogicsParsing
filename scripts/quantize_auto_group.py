#!/usr/bin/env python3
"""
自动选择group_size的量化脚本 - 解决维度兼容性问题
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantize_auto_group.log')
        ]
    )


def try_quantize_with_group_sizes(model, processor, nbits=8, method="asinq"):
    """尝试不同的group_size值"""
    group_sizes = [64, 128, 256, 512, 1024, 2048]
    
    for group_size in group_sizes:
        try:
            print(f"尝试 group_size={group_size}...")
            
            quant_config = BaseQuantizeConfig(
                nbits=nbits,
                group_size=group_size,
                tiling_mode="1D",
                method=method
            )
            
            AutoSINQHFModel.quantize_model(
                model,
                tokenizer=processor.tokenizer,
                quant_config=quant_config,
                compute_dtype=torch.float16,
                device="cpu"
            )
            
            print(f"✓ 成功使用 group_size={group_size}")
            return True
            
        except Exception as e:
            print(f"✗ group_size={group_size} 失败: {e}")
            continue
    
    print("所有group_size都失败了")
    return False


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 模型路径
    model_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/"
    output_path = "/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/"
    
    print("=" * 60)
    print("自动group_size量化")
    print("=" * 60)
    
    try:
        # 1. 加载处理器
        print("1. 加载处理器...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. 加载模型
        print("2. 加载模型...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # 3. 尝试不同group_size
        print("3. 尝试不同group_size...")
        success = try_quantize_with_group_sizes(model, processor, nbits=8, method="asinq")
        
        if success:
            print("✓ 量化完成!")
            
            # 4. 保存量化模型
            print("4. 保存量化模型...")
            AutoSINQHFModel.save_quantized(model, output_path, verbose=True)
            
            print("✓ 模型保存完成!")
            
            print("\n" + "=" * 60)
            print("量化成功完成!")
            print("=" * 60)
        else:
            print("\n✗ 所有group_size都失败了")
            
    except Exception as e:
        print(f"\n✗ 量化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()