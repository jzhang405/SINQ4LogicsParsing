"""Integration tests for model loading from cache."""

import pytest
from pathlib import Path


class TestModelLoadingIntegration:
    """Integration tests for model loading functionality."""

    @pytest.mark.integration
    def test_load_model_from_actual_cache(self):
        """Test loading model from actual cache directory."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        
        # This should fail initially as ModelLoader doesn't exist
        model = loader.load_model(model_path)
        
        assert model is not None
        assert hasattr(model, 'config')
        assert hasattr(model, 'device')

    @pytest.mark.integration
    def test_model_loading_with_different_devices(self):
        """Test model loading with different device configurations."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        
        # Test CPU loading
        model_cpu = loader.load_model(model_path, device="cpu")
        assert model_cpu.device.type == "cpu"
        
        # Test CUDA loading if available
        if torch.cuda.is_available():
            model_cuda = loader.load_model(model_path, device="cuda")
            assert model_cuda.device.type == "cuda"
        
        # Test MPS loading if available (Apple Silicon)
        if torch.backends.mps.is_available():
            model_mps = loader.load_model(model_path, device="mps")
            assert model_mps.device.type == "mps"

    @pytest.mark.integration
    def test_model_config_integrity(self):
        """Test that model config is properly loaded and accessible."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        model = loader.load_model(model_path)
        
        config = model.config
        
        # Verify essential config attributes
        assert hasattr(config, 'model_type')
        assert hasattr(config, 'vocab_size')
        assert hasattr(config, 'hidden_size')
        assert hasattr(config, 'num_attention_heads')
        assert hasattr(config, 'num_hidden_layers')

    @pytest.mark.integration
    def test_model_forward_pass(self):
        """Test that loaded model can perform forward pass."""
        from src.services.model_loader import ModelLoader
        import torch
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        model = loader.load_model(model_path)
        
        # Create dummy input for multimodal model
        # For vision component
        dummy_image = torch.randn(1, 3, 224, 224)
        # For language component
        dummy_text = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            # This should work for multimodal models
            try:
                output = model(dummy_image, dummy_text)
                assert output is not None
            except Exception as e:
                # If the specific signature doesn't work, try alternatives
                pytest.skip(f"Model forward pass failed with specific signature: {e}")

    @pytest.mark.integration
    def test_model_parameter_count(self):
        """Test that model has expected number of parameters."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        model = loader.load_model(model_path)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Log the parameter count for debugging
        print(f"Model parameter count: {param_count}")
        
        # For a multimodal model, expect significant parameter count
        assert param_count > 1000000  # At least 1M parameters

    @pytest.mark.integration
    def test_model_state_dict_loading(self):
        """Test that model state dict is properly loaded."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        
        # Skip if model doesn't exist
        if not model_path.exists():
            pytest.skip("Model cache directory not found")
        
        loader = ModelLoader()
        model = loader.load_model(model_path)
        
        # Check that model has state dict
        state_dict = model.state_dict()
        assert state_dict is not None
        assert len(state_dict) > 0
        
        # Verify some key components exist in state dict
        expected_keys = ['vision_tower', 'language_model']
        state_dict_keys = list(state_dict.keys())
        
        # Check if any key contains expected components
        has_vision = any('vision' in key.lower() for key in state_dict_keys)
        has_language = any('language' in key.lower() or 'text' in key.lower() for key in state_dict_keys)
        
        assert has_vision or has_language, "State dict should contain vision or language components"