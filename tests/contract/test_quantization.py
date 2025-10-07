"""Contract tests for quantization functionality."""

import pytest
import torch
from pathlib import Path


class TestQuantizationContract:
    """Contract tests for quantization functionality."""

    def test_quantize_model(self):
        """Test that model can be quantized using SINQ method."""
        from src.services.quantizer import SINQQuantizer
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        quantizer = SINQQuantizer()
        
        # This should fail initially as classes don't exist
        model = loader.load_model(model_path)
        quantized_model = quantizer.quantize(model, bits=4)
        
        assert quantized_model is not None
        assert quantized_model != model

    def test_quantization_preserves_functionality(self):
        """Test that quantized model preserves basic functionality."""
        from src.services.quantizer import SINQQuantizer
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        quantizer = SINQQuantizer()
        
        # This should fail initially as classes don't exist
        model = loader.load_model(model_path)
        quantized_model = quantizer.quantize(model, bits=4)
        
        # Test that quantized model can still perform inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = quantized_model(dummy_input)
        assert output is not None

    def test_quantization_bits_parameter(self):
        """Test that quantization works with different bit precisions."""
        from src.services.quantizer import SINQQuantizer
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        quantizer = SINQQuantizer()
        
        # This should fail initially as classes don't exist
        model = loader.load_model(model_path)
        
        for bits in [2, 4, 8]:
            quantized_model = quantizer.quantize(model, bits=bits)
            assert quantized_model is not None

    def test_quantization_size_reduction(self):
        """Test that quantization reduces model size."""
        from src.services.quantizer import SINQQuantizer
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        quantizer = SINQQuantizer()
        
        # This should fail initially as classes don't exist
        model = loader.load_model(model_path)
        quantized_model = quantizer.quantize(model, bits=4)
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        reduction_ratio = (original_size - quantized_size) / original_size
        assert reduction_ratio > 0.5  # At least 50% size reduction

    def test_quantization_accuracy_preservation(self):
        """Test that quantization preserves model accuracy."""
        from src.services.quantizer import SINQQuantizer
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        quantizer = SINQQuantizer()
        
        # This should fail initially as classes don't exist
        model = loader.load_model(model_path)
        quantized_model = quantizer.quantize(model, bits=4)
        
        # Test with sample input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_output = model(dummy_input)
            quantized_output = quantized_model(dummy_input)
        
        # Check that outputs are similar (within tolerance)
        diff = torch.abs(original_output - quantized_output).mean()
        assert diff < 0.1  # Less than 10% difference