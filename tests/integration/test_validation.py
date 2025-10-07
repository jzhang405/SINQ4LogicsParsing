"""Integration tests for model validation."""

import pytest
from pathlib import Path


class TestModelValidationIntegration:
    """Integration tests for model validation functionality."""

    @pytest.mark.integration
    def test_model_validation_accuracy(self):
        """Test that quantized model maintains acceptable accuracy."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Validate accuracy
        accuracy_score = validator.validate_accuracy(quantized_model, model_path)
        
        assert accuracy_score > 0.95  # At least 95% accuracy preservation

    @pytest.mark.integration
    def test_model_validation_performance(self):
        """Test that quantized model meets performance requirements."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Validate performance
        performance_score = validator.validate_performance(quantized_model)
        
        assert performance_score > 0.8  # At least 80% performance preservation

    @pytest.mark.integration
    def test_model_validation_size_reduction(self):
        """Test that quantized model achieves target size reduction."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Validate size reduction
        size_reduction = validator.validate_size_reduction(quantized_model, model_path)
        
        assert size_reduction > 0.5  # At least 50% size reduction

    @pytest.mark.integration
    def test_model_validation_functionality(self):
        """Test that quantized model preserves core functionality."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Validate functionality
        functionality_score = validator.validate_functionality(quantized_model)
        
        assert functionality_score > 0.9  # At least 90% functionality preservation

    @pytest.mark.integration
    def test_model_validation_comprehensive(self):
        """Test comprehensive validation of quantized model."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Run comprehensive validation
        validation_results = validator.comprehensive_validation(quantized_model, model_path)
        
        # Check all validation criteria
        assert validation_results['accuracy'] > 0.95
        assert validation_results['performance'] > 0.8
        assert validation_results['size_reduction'] > 0.5
        assert validation_results['functionality'] > 0.9
        assert validation_results['overall_score'] > 0.85

    @pytest.mark.integration
    def test_model_validation_with_different_bits(self):
        """Test validation with different quantization bit precisions."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        for bits in [2, 4, 8]:
            # Quantize model
            quantized_model = pipeline.quantize_model(model_path, bits=bits)
            
            # Validate
            validation_results = validator.comprehensive_validation(quantized_model, model_path)
            
            # Lower bits should have better size reduction but potentially lower accuracy
            if bits == 2:
                assert validation_results['size_reduction'] > 0.7  # Higher reduction for 2-bit
            elif bits == 4:
                assert validation_results['size_reduction'] > 0.5  # Medium reduction for 4-bit
            elif bits == 8:
                assert validation_results['size_reduction'] > 0.25  # Lower reduction for 8-bit

    @pytest.mark.integration
    def test_model_validation_error_handling(self):
        """Test validation error handling."""
        from src.services.validator import ModelValidator
        
        validator = ModelValidator()
        
        # Test with None model
        with pytest.raises(Exception):
            validator.validate_accuracy(None, Path("/dummy/path"))
        
        # Test with invalid model path
        with pytest.raises(Exception):
            validator.validate_accuracy("dummy_model", Path("/non/existent/path"))

    @pytest.mark.integration
    def test_model_validation_benchmark(self):
        """Test validation with benchmark data."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Run benchmark validation
        benchmark_results = validator.benchmark_validation(quantized_model, model_path)
        
        # Check benchmark metrics
        assert 'inference_time' in benchmark_results
        assert 'memory_usage' in benchmark_results
        assert 'throughput' in benchmark_results
        assert 'latency' in benchmark_results
        
        # Verify reasonable values
        assert benchmark_results['inference_time'] > 0
        assert benchmark_results['memory_usage'] > 0
        assert benchmark_results['throughput'] > 0
        assert benchmark_results['latency'] > 0

    @pytest.mark.integration
    def test_model_validation_comparison(self):
        """Test comparison between original and quantized models."""
        from src.services.validator import ModelValidator
        from src.services.pipeline import QuantizationPipeline
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        pipeline = QuantizationPipeline()
        validator = ModelValidator()
        loader = ModelLoader()
        
        # Load original model
        original_model = loader.load_model(model_path)
        
        # Quantize model
        output_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/")
        quantized_model = pipeline.quantize_model(model_path, bits=4, output_path=output_path)
        
        # Compare models
        comparison_results = validator.compare_models(original_model, quantized_model)
        
        # Check comparison metrics
        assert 'accuracy_difference' in comparison_results
        assert 'size_difference' in comparison_results
        assert 'performance_difference' in comparison_results
        assert 'overall_comparison' in comparison_results
        
        # Verify quantized model meets requirements
        assert comparison_results['accuracy_difference'] < 0.05  # Less than 5% accuracy loss
        assert comparison_results['size_difference'] > 0.5  # More than 50% size reduction
        assert comparison_results['overall_comparison'] > 0.8  # Good overall comparison score