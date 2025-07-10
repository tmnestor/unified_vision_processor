"""
Unit Tests for Model Factory

Tests the model factory that creates and manages vision models
in the unified architecture with standardized interfaces.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.model_factory import (
    ModelCreationError,
    ModelFactory,
)
from vision_processor.config.unified_config import ModelType, UnifiedConfig
from vision_processor.models.internvl_model import InternVLModel
from vision_processor.models.llama_model import LlamaVisionModel


class TestModelFactory:
    """Test suite for ModelFactory class."""

    def test_model_factory_internvl_creation(self, test_config):
        """Test InternVL model creation through factory."""
        test_config.model_type = ModelType.INTERNVL3
        test_config.model_path = "mock_internvl_path"

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_internvl:
            # Setup mock
            mock_instance = MagicMock(spec=InternVLModel)
            mock_internvl.return_value = mock_instance
            mock_instance.model_type = "internvl3"

            # Create model
            model = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_internvl_path", test_config
            )

            # Verify creation
            mock_internvl.assert_called_once_with("mock_internvl_path", test_config)
            assert model == mock_instance
            assert model.model_type == "internvl3"

    def test_model_factory_llama_creation(self, test_config):
        """Test Llama model creation through factory."""
        test_config.model_type = ModelType.LLAMA32_VISION
        test_config.model_path = "mock_llama_path"

        with patch(
            "vision_processor.models.llama_model.LlamaVisionModel"
        ) as mock_llama:
            # Setup mock
            mock_instance = MagicMock(spec=LlamaVisionModel)
            mock_llama.return_value = mock_instance
            mock_instance.model_type = "llama32_vision"

            # Create model
            model = ModelFactory.create_model(
                ModelType.LLAMA32_VISION, "mock_llama_path", test_config
            )

            # Verify creation
            mock_llama.assert_called_once_with("mock_llama_path", test_config)
            assert model == mock_instance
            assert model.model_type == "llama32_vision"

    def test_model_factory_invalid_type(self, test_config):
        """Test factory error handling for invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelFactory.create_model("invalid_model_type", "mock_path", test_config)

    def test_model_factory_config_validation(self):
        """Test factory configuration validation."""
        config = UnifiedConfig()
        config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance

            # Test valid configuration
            model = ModelFactory.create_model(ModelType.INTERNVL3, "mock_path", config)

            assert model is not None
            mock_model.assert_called_once_with("mock_path", config)

    def test_model_factory_device_configuration(self, test_config):
        """Test model factory device configuration handling."""
        test_config.device_config = "cuda:0"
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance
            mock_instance.device = "cuda:0"

            model = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_path", test_config
            )

            assert model.device == "cuda:0"
            mock_model.assert_called_once_with("mock_path", test_config)

    def test_model_factory_memory_optimization(self, test_config):
        """Test model factory memory optimization features."""
        test_config.enable_8bit_quantization = True
        test_config.gpu_memory_fraction = 0.6
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance

            ModelFactory.create_model(ModelType.INTERNVL3, "mock_path", test_config)

            # Verify configuration was passed
            mock_model.assert_called_once_with("mock_path", test_config)
            assert test_config.enable_8bit_quantization is True
            assert test_config.gpu_memory_fraction == 0.6

    def test_model_factory_multi_gpu_configuration(self, test_config):
        """Test model factory multi-GPU configuration."""
        test_config.multi_gpu_dev = True
        test_config.device_config = "auto"
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance
            mock_instance.supports_multi_gpu = True

            model = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_path", test_config
            )

            assert model.supports_multi_gpu is True
            mock_model.assert_called_once_with("mock_path", test_config)

    def test_model_factory_auto_device_selection(self, test_config):
        """Test automatic device selection in factory."""
        test_config.device_config = "auto"
        test_config.model_type = ModelType.LLAMA32_VISION

        with patch(
            "vision_processor.models.llama_model.LlamaVisionModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=LlamaVisionModel)
            mock_model.return_value = mock_instance

            # Mock device detection
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=2):
                    ModelFactory.create_model(
                        ModelType.LLAMA32_VISION, "mock_path", test_config
                    )

                    mock_model.assert_called_once_with("mock_path", test_config)

    def test_model_factory_error_handling(self, test_config):
        """Test model factory error handling."""
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            # Simulate model creation failure
            mock_model.side_effect = Exception("Model loading failed")

            with pytest.raises(ModelCreationError, match="Failed to create model"):
                ModelFactory.create_model(
                    ModelType.INTERNVL3, "invalid_path", test_config
                )

    def test_model_factory_path_validation(self, test_config):
        """Test model factory path validation."""
        test_config.model_type = ModelType.INTERNVL3

        # Test with None path
        with pytest.raises(ValueError, match="Model path cannot be None or empty"):
            ModelFactory.create_model(ModelType.INTERNVL3, None, test_config)

        # Test with empty path
        with pytest.raises(ValueError, match="Model path cannot be None or empty"):
            ModelFactory.create_model(ModelType.INTERNVL3, "", test_config)

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        supported_models = ModelFactory.get_supported_models()

        assert isinstance(supported_models, list)
        assert "internvl3" in supported_models
        assert "llama32_vision" in supported_models
        assert len(supported_models) >= 2

    def test_is_model_supported(self):
        """Test model support checking."""
        assert ModelFactory.is_model_supported(ModelType.INTERNVL3) is True
        assert ModelFactory.is_model_supported(ModelType.LLAMA32_VISION) is True
        assert ModelFactory.is_model_supported("invalid_model") is False

    def test_model_factory_with_custom_config(self):
        """Test model factory with custom configuration parameters."""
        config = UnifiedConfig()
        config.model_type = ModelType.INTERNVL3
        config.custom_parameter = "custom_value"  # Custom configuration

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance

            ModelFactory.create_model(ModelType.INTERNVL3, "mock_path", config)

            # Verify custom config was passed
            mock_model.assert_called_once_with("mock_path", config)
            assert hasattr(config, "custom_parameter")
            assert config.custom_parameter == "custom_value"

    def test_model_factory_singleton_behavior(self, test_config):
        """Test that factory creates fresh instances (not singleton)."""
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_model.side_effect = lambda *_args, **_kwargs: MagicMock(
                spec=InternVLModel
            )

            # Create two models
            model1 = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_path", test_config
            )

            model2 = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_path", test_config
            )

            # Should be different instances
            assert model1 is not model2
            assert mock_model.call_count == 2

    def test_model_factory_response_standardization(self, test_config):
        """Test that factory ensures response standardization."""
        test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.models.internvl_model.InternVLModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=InternVLModel)
            mock_model.return_value = mock_instance

            # Mock standardized response
            mock_response = Mock()
            mock_response.raw_text = "Mock response"
            mock_response.confidence = 0.85
            mock_response.processing_time = 1.5
            mock_instance.process_image.return_value = mock_response

            model = ModelFactory.create_model(
                ModelType.INTERNVL3, "mock_path", test_config
            )

            # Test response standardization
            response = model.process_image("mock_image.jpg", "mock_prompt")
            assert hasattr(response, "raw_text")
            assert hasattr(response, "confidence")
            assert hasattr(response, "processing_time")

    def test_model_factory_cross_platform_compatibility(self, test_config):
        """Test model factory cross-platform compatibility."""
        test_config.model_type = ModelType.LLAMA32_VISION
        test_config.cross_platform = True

        with patch(
            "vision_processor.models.llama_model.LlamaVisionModel"
        ) as mock_model:
            mock_instance = MagicMock(spec=LlamaVisionModel)
            mock_model.return_value = mock_instance
            mock_instance.cross_platform_compatible = True

            model = ModelFactory.create_model(
                ModelType.LLAMA32_VISION, "mock_path", test_config
            )

            assert model.cross_platform_compatible is True
            mock_model.assert_called_once_with("mock_path", test_config)
