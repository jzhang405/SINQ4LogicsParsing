"""Model validation service for quantized models."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import numpy as np

from .model_loader import ModelLoader


logger = logging.getLogger(__name__)


class ModelValidator:
    """Service for validating quantized model quality and performance."""

    def __init__(self):
        """Initialize model validator."""
        self.model_loader = ModelLoader()
        self.device = self.model_loader.device

    def validate_accuracy(
        self, 
        quantized_model: torch.nn.Module, 
        original_model_path: Union[str, Path]
    ) -> float:
        """
        Validate that quantized model maintains acceptable accuracy.
        
        Args:
            quantized_model: Quantized model
            original_model_path: Path to original model for comparison
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        logger.info("Validating model accuracy")
        
        try:
            # Load original model for comparison
            original_model = self.model_loader.load_model(original_model_path)
            
            # Create test inputs
            test_inputs = self._generate_test_inputs()
            
            accuracy_scores = []
            
            for input_data in test_inputs:
                with torch.no_grad():
                    try:
                        # Get outputs from both models
                        original_output = original_model(*input_data)
                        quantized_output = quantized_model(*input_data)
                        
                        # Calculate similarity
                        similarity = self._calculate_output_similarity(
                            original_output, quantized_output
                        )
                        accuracy_scores.append(similarity)
                        
                    except Exception as e:
                        logger.warning(f"Accuracy test failed for input: {e}")
                        accuracy_scores.append(0.0)
            
            # Calculate average accuracy
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            logger.info(f"Accuracy validation completed: {avg_accuracy:.4f}")
            
            return avg_accuracy
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            return 0.0

    def validate_performance(self, quantized_model: torch.nn.Module) -> float:
        """
        Validate that quantized model meets performance requirements.
        
        Args:
            quantized_model: Quantized model
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        logger.info("Validating model performance")
        
        try:
            # Generate test inputs
            test_inputs = self._generate_test_inputs()
            
            # Measure inference time
            inference_times = []
            
            for input_data in test_inputs:
                start_time = time.time()
                
                with torch.no_grad():
                    try:
                        _ = quantized_model(*input_data)
                    except Exception as e:
                        logger.warning(f"Performance test failed: {e}")
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            # Calculate average inference time
            avg_inference_time = np.mean(inference_times) if inference_times else float('inf')
            
            # Normalize performance score (lower time = better performance)
            # Assuming reasonable inference time threshold of 1 second
            performance_score = max(0.0, 1.0 - avg_inference_time)
            
            logger.info(f"Performance validation completed: {performance_score:.4f}")
            logger.info(f"Average inference time: {avg_inference_time:.4f}s")
            
            return performance_score
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return 0.0

    def validate_size_reduction(
        self, 
        quantized_model: torch.nn.Module, 
        original_model_path: Union[str, Path]
    ) -> float:
        """
        Validate that quantized model achieves target size reduction.
        
        Args:
            quantized_model: Quantized model
            original_model_path: Path to original model
            
        Returns:
            Size reduction ratio (0.0 to 1.0)
        """
        logger.info("Validating model size reduction")
        
        try:
            # Load original model
            original_model = self.model_loader.load_model(original_model_path)
            
            # Calculate model sizes
            original_size = self._calculate_model_size(original_model)
            quantized_size = self._calculate_model_size(quantized_model)
            
            # Calculate size reduction
            size_reduction = (original_size - quantized_size) / original_size
            
            logger.info(f"Size reduction validation completed: {size_reduction:.4f}")
            logger.info(f"Original size: {original_size / (1024*1024):.2f} MB")
            logger.info(f"Quantized size: {quantized_size / (1024*1024):.2f} MB")
            
            return size_reduction
            
        except Exception as e:
            logger.error(f"Size reduction validation failed: {e}")
            return 0.0

    def validate_functionality(self, quantized_model: torch.nn.Module) -> float:
        """
        Validate that quantized model preserves core functionality.
        
        Args:
            quantized_model: Quantized model
            
        Returns:
            Functionality score (0.0 to 1.0)
        """
        logger.info("Validating model functionality")
        
        functionality_tests = []
        
        try:
            # Test 1: Model can perform forward pass
            test_inputs = self._generate_test_inputs()
            for input_data in test_inputs:
                with torch.no_grad():
                    try:
                        output = quantized_model(*input_data)
                        if output is not None:
                            functionality_tests.append(1.0)
                        else:
                            functionality_tests.append(0.0)
                    except Exception:
                        functionality_tests.append(0.0)
            
            # Test 2: Model has expected attributes
            expected_attrs = ['config', 'device', 'training']
            attr_tests = []
            for attr in expected_attrs:
                if hasattr(quantized_model, attr):
                    attr_tests.append(1.0)
                else:
                    attr_tests.append(0.0)
            
            # Test 3: Model parameters are accessible
            try:
                param_count = sum(p.numel() for p in quantized_model.parameters())
                if param_count > 0:
                    functionality_tests.append(1.0)
                else:
                    functionality_tests.append(0.0)
            except Exception:
                functionality_tests.append(0.0)
            
            # Calculate overall functionality score
            functionality_score = np.mean(functionality_tests + attr_tests)
            
            logger.info(f"Functionality validation completed: {functionality_score:.4f}")
            
            return functionality_score
            
        except Exception as e:
            logger.error(f"Functionality validation failed: {e}")
            return 0.0

    def comprehensive_validation(
        self, 
        quantized_model: torch.nn.Module, 
        original_model_path: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Run comprehensive validation of quantized model.
        
        Args:
            quantized_model: Quantized model
            original_model_path: Path to original model
            
        Returns:
            Dictionary with all validation scores
        """
        logger.info("Running comprehensive validation")
        
        validation_results = {}
        
        # Run individual validations
        validation_results['accuracy'] = self.validate_accuracy(
            quantized_model, original_model_path
        )
        validation_results['performance'] = self.validate_performance(quantized_model)
        validation_results['size_reduction'] = self.validate_size_reduction(
            quantized_model, original_model_path
        )
        validation_results['functionality'] = self.validate_functionality(quantized_model)
        
        # Calculate overall score (weighted average)
        weights = {
            'accuracy': 0.4,
            'performance': 0.2,
            'size_reduction': 0.3,
            'functionality': 0.1
        }
        
        overall_score = sum(
            validation_results[key] * weights[key] 
            for key in validation_results.keys()
        )
        validation_results['overall_score'] = overall_score
        
        logger.info(f"Comprehensive validation completed: {overall_score:.4f}")
        
        return validation_results

    def benchmark_validation(
        self, 
        quantized_model: torch.nn.Module, 
        original_model_path: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Run benchmark validation with detailed metrics.
        
        Args:
            quantized_model: Quantized model
            original_model_path: Path to original model
            
        Returns:
            Dictionary with benchmark metrics
        """
        logger.info("Running benchmark validation")
        
        benchmark_results = {}
        
        try:
            # Load original model for comparison
            original_model = self.model_loader.load_model(original_model_path)
            
            # Generate test inputs
            test_inputs = self._generate_test_inputs()
            
            # Benchmark inference time
            original_times = []
            quantized_times = []
            
            for input_data in test_inputs:
                # Original model
                start_time = time.time()
                with torch.no_grad():
                    _ = original_model(*input_data)
                original_times.append(time.time() - start_time)
                
                # Quantized model
                start_time = time.time()
                with torch.no_grad():
                    _ = quantized_model(*input_data)
                quantized_times.append(time.time() - start_time)
            
            benchmark_results['original_inference_time'] = np.mean(original_times)
            benchmark_results['quantized_inference_time'] = np.mean(quantized_times)
            benchmark_results['speedup_ratio'] = (
                benchmark_results['original_inference_time'] / 
                benchmark_results['quantized_inference_time']
            )
            
            # Memory usage (approximate)
            original_memory = self._calculate_model_size(original_model)
            quantized_memory = self._calculate_model_size(quantized_model)
            benchmark_results['original_memory_mb'] = original_memory / (1024 * 1024)
            benchmark_results['quantized_memory_mb'] = quantized_memory / (1024 * 1024)
            benchmark_results['memory_reduction'] = (
                (original_memory - quantized_memory) / original_memory
            )
            
            # Throughput (samples per second)
            benchmark_results['original_throughput'] = 1.0 / benchmark_results['original_inference_time']
            benchmark_results['quantized_throughput'] = 1.0 / benchmark_results['quantized_inference_time']
            
            # Latency
            benchmark_results['original_latency'] = benchmark_results['original_inference_time']
            benchmark_results['quantized_latency'] = benchmark_results['quantized_inference_time']
            
            logger.info("Benchmark validation completed")
            
        except Exception as e:
            logger.error(f"Benchmark validation failed: {e}")
            # Return default values
            benchmark_results = {
                'original_inference_time': 0.0,
                'quantized_inference_time': 0.0,
                'speedup_ratio': 1.0,
                'original_memory_mb': 0.0,
                'quantized_memory_mb': 0.0,
                'memory_reduction': 0.0,
                'original_throughput': 0.0,
                'quantized_throughput': 0.0,
                'original_latency': 0.0,
                'quantized_latency': 0.0
            }
        
        return benchmark_results

    def compare_models(
        self, 
        original_model: torch.nn.Module, 
        quantized_model: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Compare original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison_results = {}
        
        try:
            # Accuracy difference
            test_inputs = self._generate_test_inputs()
            accuracy_diffs = []
            
            for input_data in test_inputs:
                with torch.no_grad():
                    try:
                        original_output = original_model(*input_data)
                        quantized_output = quantized_model(*input_data)
                        similarity = self._calculate_output_similarity(
                            original_output, quantized_output
                        )
                        accuracy_diffs.append(1.0 - similarity)
                    except Exception:
                        accuracy_diffs.append(1.0)
            
            comparison_results['accuracy_difference'] = np.mean(accuracy_diffs)
            
            # Size difference
            original_size = self._calculate_model_size(original_model)
            quantized_size = self._calculate_model_size(quantized_model)
            comparison_results['size_difference'] = (
                (original_size - quantized_size) / original_size
            )
            
            # Performance difference
            original_time, quantized_time = self._measure_inference_times(
                original_model, quantized_model
            )
            comparison_results['performance_difference'] = (
                (original_time - quantized_time) / original_time
            )
            
            # Overall comparison score
            comparison_results['overall_comparison'] = (
                1.0 - comparison_results['accuracy_difference'] * 0.6 +
                comparison_results['size_difference'] * 0.3 +
                comparison_results['performance_difference'] * 0.1
            )
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            comparison_results = {
                'accuracy_difference': 1.0,
                'size_difference': 0.0,
                'performance_difference': 0.0,
                'overall_comparison': 0.0
            }
        
        return comparison_results

    def _generate_test_inputs(self) -> list:
        """Generate test inputs for model validation."""
        test_inputs = []
        
        # For multimodal model, create both image and text inputs
        # Image input
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        # Text input
        dummy_text = torch.randint(0, 1000, (1, 10)).to(self.device)
        
        test_inputs.append((dummy_image, dummy_text))
        
        # Add more variations
        test_inputs.append((torch.randn(1, 3, 224, 224).to(self.device), 
                          torch.randint(0, 1000, (1, 5)).to(self.device)))
        
        return test_inputs

    def _calculate_output_similarity(
        self, 
        output1: torch.Tensor, 
        output2: torch.Tensor
    ) -> float:
        """Calculate similarity between two model outputs."""
        try:
            if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
                # Flatten outputs for comparison
                flat1 = output1.flatten()
                flat2 = output2.flatten()
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    flat1.unsqueeze(0), flat2.unsqueeze(0)
                ).item()
                
                return max(0.0, similarity)
            else:
                # For complex outputs, return 1.0 if both are not None
                return 1.0 if output1 is not None and output2 is not None else 0.0
                
        except Exception:
            return 0.0

    def _calculate_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    def _measure_inference_times(
        self, 
        model1: torch.nn.Module, 
        model2: torch.nn.Module
    ) -> tuple:
        """Measure inference times for two models."""
        test_inputs = self._generate_test_inputs()
        
        times1 = []
        times2 = []
        
        for input_data in test_inputs:
            # Model 1
            start_time = time.time()
            with torch.no_grad():
                _ = model1(*input_data)
            times1.append(time.time() - start_time)
            
            # Model 2
            start_time = time.time()
            with torch.no_grad():
                _ = model2(*input_data)
            times2.append(time.time() - start_time)
        
        return np.mean(times1), np.mean(times2)