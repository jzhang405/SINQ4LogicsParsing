# Logics-Parsing 模型量化指南

## 可用量化方法

### 1. CLI 量化工具
```bash
# 查看所有配置
python src/cli/quantize_cli.py --list-configs

# 使用Logics-Parsing优化配置 (推荐)
python src/cli/quantize_cli.py --logics-parsing

# 使用特定配置
python src/cli/quantize_cli.py --config 8bit_asinq --device cpu
```

### 2. 专用脚本

#### CPU-only 量化 (16GB内存限制)
```bash
python scripts/quantize_cpu_only.py
```

#### 内存优化量化 (自动选择设备)
```bash
python scripts/quantize_memory_optimized.py
```

#### Logics-Parsing专用量化
```bash
python scripts/quantize_logics_parsing.py
```

## 量化配置说明

### Logics-Parsing优化配置
- **方法**: A-SINQ (8-bit)
- **分组**: 无 (group_size=None)
- **平铺**: 1D
- **设备**: 自动选择 (优先MPS，内存不足时回退到CPU)

### 标准配置
- **4bit_no_group**: 4-bit SINQ，无分组
- **8bit_no_group**: 8-bit SINQ，无分组  
- **4bit_asinq**: 4-bit A-SINQ，无分组
- **8bit_asinq**: 8-bit A-SINQ，无分组 (推荐)
- **4bit_group128**: 4-bit SINQ，128分组
- **4bit_group64**: 4-bit SINQ，64分组
- **4bit_2d**: 4-bit SINQ，2D平铺

## 内存使用建议

### 16GB内存限制
- 使用 `quantize_cpu_only.py` 脚本
- 或 CLI: `--device cpu --config 8bit_asinq`

### 24GB内存
- 可尝试使用 MPS: `--device mps --config 8bit_asinq`
- 如果内存不足，自动回退到 CPU

## 模型路径

- **原始模型**: `/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/`
- **量化模型**: `/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/`

## 故障排除

### 内存不足
- 使用 CPU-only 模式
- 关闭其他应用程序释放内存
- 使用更低的量化位宽 (4-bit)

### 量化失败
- 检查 SINQ 库是否正确安装
- 验证模型路径是否存在
- 尝试不同的量化配置

## 性能预期

- **模型大小**: 减少约 50-75%
- **推理速度**: 提升 2-4 倍
- **内存使用**: 减少 60-80%
- **精度损失**: < 5% (A-SINQ 8-bit)