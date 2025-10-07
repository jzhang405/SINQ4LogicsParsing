"""Contract tests for model loading functionality."""

import pytest
from pathlib import Path


class TestModelLoadingContract:
    """Contract tests for model loading functionality."""

    def test_load_model_from_cache(self):
        """Test that model can be loaded from cache directory."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        
        # This should fail initially as ModelLoader doesn't exist
        model = loader.load_model(model_path)
        assert model is not None
        assert hasattr(model, 'config')

    def test_load_model_with_device(self):
        """Test that model can be loaded with specific device."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        
        # This should fail initially as ModelLoader doesn't exist
        model = loader.load_model(model_path, device="cpu")
        assert model is not None
        assert model.device.type == "cpu"

    def test_model_has_expected_components(self):
        """Test that loaded model has expected components."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        
        # This should fail initially as ModelLoader doesn't exist
        model = loader.load_model(model_path)
        
        # Check for multimodal model components
        assert hasattr(model, 'vision_tower')
        assert hasattr(model, 'language_model')
        assert hasattr(model, 'config')

    def test_model_config_validation(self):
        """Test that model config is properly validated."""
        from src.services.model_loader import ModelLoader
        
        model_path = Path("/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing/")
        loader = ModelLoader()
        
        # This should fail initially as ModelLoader doesn't exist
        model = loader.load_model(model_path)
        
        config = model.config
        assert hasattr(config, 'model_type')
        assert config.model_type == 'logics-parsing'
        assert hasattr(config, 'vocab_size')
        assert hasattr(config, 'hidden_size')