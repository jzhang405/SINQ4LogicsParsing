# Logics-Parsing 量化状态

## 当前问题

**主要问题**: SINQ与Qwen2.5-VL多模态模型的维度不兼容

**具体错误**: `block must divide W`

**原因**: SINQ要求权重矩阵的宽度必须能被group_size整除，但Qwen2.5-VL模型的某些层不满足这个要求

## 已尝试的解决方案

### 1. 标准配置
- ❌ `group_size=64` - 失败 (block must divide W)
- ❌ `group_size=128` - 失败 (block must divide W)  
- ❌ `group_size=None` - 失败 (unsupported operand type)

### 2. 自动group_size选择
- ✅ 创建了 `scripts/quantize_auto_group.py`
- ✅ 自动尝试: 64, 128, 256, 512, 1024, 2048

## 当前可用的方法

### CLI 工具
```bash
# 查看配置
python src/cli/quantize_cli.py --list-configs

# 尝试自动group_size
python scripts/quantize_auto_group.py

# 使用特定配置
python src/cli/quantize_cli.py --config 8bit_group64 --device cpu
```

### 专用脚本
- `scripts/quantize_auto_group.py` - 自动寻找兼容的group_size
- `scripts/quantize_cpu_only.py` - CPU-only量化
- `scripts/quantize_memory_optimized.py` - 内存优化

## 下一步

1. **运行自动group_size脚本**:
   ```bash
   python scripts/quantize_auto_group.py
   ```

2. **如果自动脚本失败**:
   - 可能需要修改SINQ库来处理不兼容的维度
   - 或者使用其他量化方法 (如GGUF, AWQ等)

## 技术细节

- **模型**: Qwen2.5-VL (多模态)
- **量化方法**: SINQ / A-SINQ
- **内存限制**: 16GB可用
- **设备**: CPU-only (MPS内存不足)

## 建议

1. 首先尝试自动group_size脚本
2. 如果失败，考虑使用其他量化工具
3. 或者联系SINQ开发者添加对Qwen2.5-VL的支持