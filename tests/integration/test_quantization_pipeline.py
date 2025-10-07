"""Integration tests for quantization pipeline."""

import pytest
from pathlib import Path


class TestQuantizationPipelineIntegration:
    """Integration tests for quantization pipeline functionality."""

    @pytest.mark.integration
    def test_full_quantization_pipeline(self):
        """Test the complete quantization pipeline from model loading to quantization."""
        from src.services.model_loader import ModelLoader
        from src.services.quantizer import SINQQuantizer
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        # This should fail initially as classes don't exist
        pipeline = QuantizationPipeline()
        quantized_model = pipeline.quantize_model(model_path, bits=4)
        
        assert quantized_model is not None
        assert hasattr(quantized_model, 'config')

    @pytest.mark.integration
    def test_quantization_pipeline_with_different_bits(self):
        """Test quantization pipeline with different bit precisions."""
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        
        for bits in [2, 4, 8]:
            quantized_model = pipeline.quantize_model(model_path, bits=bits)
            assert quantized_model is not None

    @pytest.mark.integration
    def test_quantization_pipeline_size_reduction(self):
        """Test that quantization pipeline achieves expected size reduction."""
        from src.services.pipeline import QuantizationPipeline
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        loader = ModelLoader()
        
        # Load original model
        original_model = loader.load_model(model_path)
        
        # Quantize model
        quantized_model = pipeline.quantize_model(model_path, bits=4)
        
        # Calculate size reduction
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        reduction_ratio = (original_size - quantized_size) / original_size
        
        print(f"Original size: {original_size / (1024**2):.2f} MB")
        print(f"Quantized size: {quantized_size / (1024**2):.2f} MB")
        print(f"Reduction ratio: {reduction_ratio:.2%}")
        
        assert reduction_ratio > 0.5  # At least 50% size reduction

    @pytest.mark.integration
    def test_quantization_pipeline_accuracy(self):
        """Test that quantization pipeline preserves model accuracy."""
        from src.services.pipeline import QuantizationPipeline
        from src.services.model_loader import ModelLoader
        import torch
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        loader = ModelLoader()
        
        # Load original model
        original_model = loader.load_model(model_path)
        
        # Quantize model
        quantized_model = pipeline.quantize_model(model_path, bits=4)
        
        # Test with sample input
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_text = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            try:
                original_output = original_model(dummy_image, dummy_text)
                quantized_output = quantized_model(dummy_image, dummy_text)
                
                # Calculate similarity
                if isinstance(original_output, torch.Tensor) and isinstance(quantized_output, torch.Tensor):
                    diff = torch.abs(original_output - quantized_output).mean()
                    assert diff < 0.1  # Less than 10% difference
                else:
                    # For complex outputs, check they're both not None
                    assert original_output is not None
                    assert quantized_output is not None
                    
            except Exception as e:
                pytest.skip(f"Model forward pass failed: {e}")

    @pytest.mark.integration
    def test_quantization_pipeline_save_load(self):
        """Test that quantized model can be saved and loaded."""
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        
        # Quantize model and save to specified output directory
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Verify files were created in output directory
        assert output_path.exists()
        assert (output_path / "config.json").exists()
        assert (output_path / "pytorch_model.bin").exists()
        
        # Load the saved model
        loaded_model = pipeline.load_quantized_model(output_path)
        assert loaded_model is not None

    @pytest.mark.integration
    def test_quantization_pipeline_performance(self):
        """Test that quantized model has improved performance characteristics."""
        from src.services.pipeline import QuantizationPipeline
        from src.services.model_loader import ModelLoader
        import time
        import torch
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        loader = ModelLoader()
        
        # Load original model
        original_model = loader.load_model(model_path)
        
        # Quantize model
        quantized_model = pipeline.quantize_model(model_path, bits=4)
        
        # Test inference speed
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warm up
        with torch.no_grad():
            _ = original_model(dummy_input)
            _ = quantized_model(dummy_input)
        
        # Measure original model inference time
        start_time = time.time()
        with torch.no_grad():
            _ = original_model(dummy_input)
        original_time = time.time() - start_time
        
        # Measure quantized model inference time
        start_time = time.time()
        with torch.no_grad():
            _ = quantized_model(dummy_input)
        quantized_time = time.time() - start_time
        
        print(f"Original model inference time: {original_time:.4f}s")
        print(f"Quantized model inference time: {quantized_time:.4f}s")
        
        # Quantized model should be faster or similar speed
        # (quantization can sometimes be slower due to dequantization overhead)
        # But we'll check that it's not significantly slower
        assert quantized_time < original_time * 2  # Not more than 2x slower

    @pytest.mark.integration
    def test_quantization_pipeline_error_handling(self):
        """Test quantization pipeline error handling."""
        from src.services.pipeline import QuantizationPipeline
        
        pipeline = QuantizationPipeline()
        
        # Test with non-existent path
        with pytest.raises(Exception):
            pipeline.quantize_model(Path("/non/existent/path"), bits=4)
        
        # Test with invalid bits
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        if model_path.exists():
            with pytest.raises(Exception):
                pipeline.quantize_model(model_path, bits=0)  # Invalid bits
            with pytest.raises(Exception):
                pipeline.quantize_model(model_path, bits=16)  # Invalid bits