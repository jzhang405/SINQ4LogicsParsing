# SINQ4LogicsParsing
SINQ Quantization for Alibaba Logics-Parsing Model

## Project Status: ❌ FAILED

### Failure Analysis

**Primary Issue**: SINQ library incompatibility with Qwen2.5-VL multimodal model architecture

**Root Cause**: 
- SINQ requires weight matrix dimensions to be divisible by `group_size`
- Qwen2.5-VL model contains layers with incompatible dimensions
- Error: `block must divide W` - indicates dimension mismatch

**Technical Details**:
- Model: Qwen/Qwen2.5-VL (multimodal vision-language model)
- Quantization Method: SINQ/A-SINQ
- Memory Constraints: 16GB available (24GB total)
- Device: CPU-only (MPS memory insufficient)

**Attempted Solutions**:
1. ✅ Standard configurations (group_size=64, 128) - Failed
2. ✅ No grouping (group_size=None) - Failed (type error)
3. ✅ Auto group_size selection - Failed (dimension incompatibility)
4. ✅ Memory-optimized loading - Failed (fundamental library issue)

### What Works

- ✅ Project structure and CLI implementation
- ✅ Model loading and memory optimization
- ✅ Configuration management
- ✅ SINQ library integration

### What Doesn't Work

- ❌ SINQ quantization with Qwen2.5-VL model
- ❌ Any group_size configuration
- ❌ Both SINQ and A-SINQ methods

### Recommendations

1. **Alternative Quantization Tools**:
   - GGUF quantization
   - AWQ (Activation-aware Weight Quantization)
   - GPTQ

2. **Library Updates**:
   - Contact SINQ developers for Qwen2.5-VL support
   - Consider contributing dimension compatibility fixes

3. **Model Alternatives**:
   - Use standard Qwen models (non-multimodal)
   - Consider other quantization-compatible models

### Project Structure

```
├── src/cli/quantize_cli.py          # Main CLI interface
├── scripts/quantize_auto_group.py   # Auto group_size selection
├── scripts/quantize_cpu_only.py     # CPU-only optimization
├── scripts/quantize_memory_optimized.py # Memory optimization
├── design/01.md                     # Original design spec
└── specs/001-github-huawei-csl/     # Project specifications
```

### Usage (For Compatible Models)

```bash
# List available configurations
python src/cli/quantize_cli.py --list-configs

# Use Logics-Parsing optimized config
python src/cli/quantize_cli.py --logics-parsing

# Auto group_size selection
python scripts/quantize_auto_group.py
```

**Note**: This implementation works for standard models but fails with Qwen2.5-VL due to fundamental library incompatibility. 
