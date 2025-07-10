"""
Unit Tests for Unified Configuration

Tests the unified configuration system that manages settings for both
InternVL and Llama models within the unified architecture.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vision_processor.config.unified_config import (
    ExtractionMethod,
    ModelType,
    ProcessingPipeline,
    ProductionAssessment,
    UnifiedConfig,
)
from vision_processor.models.base_model import DeviceConfig


class TestUnifiedConfig:
    """Test suite for UnifiedConfig class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = UnifiedConfig()

        # Model defaults
        assert config.model_type == ModelType.INTERNVL3
        assert config.processing_pipeline == ProcessingPipeline.SEVEN_STEP
        assert config.extraction_method == ExtractionMethod.HYBRID

        # Threshold defaults
        assert config.quality_threshold == 0.6
        assert config.confidence_threshold == 0.8

        # Feature defaults
        assert config.highlight_detection is True
        assert config.awk_fallback is True
        assert config.graceful_degradation is True

        # Performance defaults
        assert config.batch_size == 1
        assert config.max_workers == 4
        assert config.gpu_memory_fraction == 0.8

    def test_from_env_configuration(self):
        """Test configuration from environment variables."""
        env_vars = {
            "VISION_MODEL_TYPE": "llama32_vision",
            "VISION_PROCESSING_PIPELINE": "7step",
            "VISION_EXTRACTION_METHOD": "awk_only",
            "VISION_QUALITY_THRESHOLD": "0.8",
            "VISION_CONFIDENCE_THRESHOLD": "0.9",
            "VISION_HIGHLIGHT_DETECTION": "false",
            "VISION_AWK_FALLBACK": "true",
            "VISION_GRACEFUL_DEGRADATION": "false",
            "VISION_BATCH_SIZE": "4",
            "VISION_MAX_WORKERS": "8",
            "VISION_GPU_MEMORY_FRACTION": "0.6",
        }

        with patch.dict(os.environ, env_vars):
            config = UnifiedConfig.from_env()

            assert config.model_type == ModelType.LLAMA32_VISION
            assert config.processing_pipeline == ProcessingPipeline.SEVEN_STEP
            assert config.extraction_method == ExtractionMethod.AWK_ONLY
            assert config.quality_threshold == 0.8
            assert config.confidence_threshold == 0.9
            assert config.highlight_detection is False
            assert config.awk_fallback is True
            assert config.graceful_degradation is False
            assert config.batch_size == 4
            assert config.max_workers == 8
            assert config.gpu_memory_fraction == 0.6

    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        with pytest.raises(ValueError):
            config = UnifiedConfig()
            config.model_type = "invalid_model"

    def test_model_type_enum_validation(self):
        """Test ModelType enum validation."""
        assert ModelType.INTERNVL3.value == "internvl3"
        assert ModelType.LLAMA32_VISION.value == "llama32_vision"

        # Test string to enum conversion
        assert ModelType("internvl3") == ModelType.INTERNVL3
        assert ModelType("llama32_vision") == ModelType.LLAMA32_VISION

    def test_configuration_validation(self):
        """Test configuration validation rules."""
        config = UnifiedConfig()

        # Test valid threshold ranges
        config.confidence_threshold = 0.5
        config.quality_threshold = 0.8
        assert config.confidence_threshold == 0.5
        assert config.quality_threshold == 0.8

        # Test invalid threshold ranges
        with pytest.raises(ValueError):
            config.confidence_threshold = 1.5  # > 1.0

        with pytest.raises(ValueError):
            config.quality_threshold = -0.1  # < 0.0

    def test_processing_pipeline_validation(self):
        """Test processing pipeline validation."""
        config = UnifiedConfig()

        # Valid pipeline
        config.processing_pipeline = ProcessingPipeline.SEVEN_STEP
        assert config.processing_pipeline == ProcessingPipeline.SEVEN_STEP

        # Test string assignment (should be converted to enum)
        config.processing_pipeline = ProcessingPipeline.SIMPLE
        assert config.processing_pipeline == ProcessingPipeline.SIMPLE

    def test_extraction_method_validation(self):
        """Test extraction method validation."""
        config = UnifiedConfig()

        valid_methods = [
            ExtractionMethod.HYBRID,
            ExtractionMethod.KEY_VALUE,
            ExtractionMethod.AWK_ONLY,
        ]
        for method in valid_methods:
            config.extraction_method = method
            assert config.extraction_method == method

    def test_device_configuration(self):
        """Test device configuration handling."""
        config = UnifiedConfig()

        # Test auto device selection
        config.device_config = DeviceConfig.AUTO
        assert config.device_config == DeviceConfig.AUTO

        # Test specific device
        config.device_config = DeviceConfig.SINGLE_GPU
        assert config.device_config == DeviceConfig.SINGLE_GPU

        # Test CPU fallback
        config.device_config = DeviceConfig.CPU
        assert config.device_config == DeviceConfig.CPU

    def test_performance_settings(self):
        """Test performance-related settings."""
        config = UnifiedConfig()

        # Test batch size limits
        config.batch_size = 1
        assert config.batch_size == 1

        config.batch_size = 16
        assert config.batch_size == 16

        # Test worker limits
        config.max_workers = 1
        assert config.max_workers == 1

        config.max_workers = 16
        assert config.max_workers == 16

        # Test memory fraction
        config.gpu_memory_fraction = 0.5
        assert config.gpu_memory_fraction == 0.5

        config.gpu_memory_fraction = 1.0
        assert config.gpu_memory_fraction == 1.0

    def test_feature_flags(self):
        """Test feature flag configuration."""
        config = UnifiedConfig()

        # Test all feature flags
        config.highlight_detection = True
        config.awk_fallback = False
        config.computer_vision = True
        config.graceful_degradation = False

        assert config.highlight_detection is True
        assert config.awk_fallback is False
        assert config.computer_vision is True
        assert config.graceful_degradation is False

    def test_path_configuration(self):
        """Test path configuration handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config = UnifiedConfig()
            config.dataset_path = temp_path / "dataset"
            config.ground_truth_path = temp_path / "ground_truth"
            config.output_path = temp_path / "output"

            assert config.dataset_path == temp_path / "dataset"
            assert config.ground_truth_path == temp_path / "ground_truth"
            assert config.output_path == temp_path / "output"

    def test_environment_variable_types(self):
        """Test proper type conversion from environment variables."""
        env_vars = {
            "VISION_CONFIDENCE_THRESHOLD": "0.85",
            "VISION_BATCH_SIZE": "8",
            "VISION_HIGHLIGHT_DETECTION": "true",
            "VISION_AWK_FALLBACK": "false",
            "VISION_GPU_MEMORY_FRACTION": "0.75",
        }

        with patch.dict(os.environ, env_vars):
            config = UnifiedConfig.from_env()

            # Test float conversion
            assert isinstance(config.confidence_threshold, float)
            assert config.confidence_threshold == 0.85

            # Test int conversion
            assert isinstance(config.batch_size, int)
            assert config.batch_size == 8

            # Test bool conversion
            assert isinstance(config.highlight_detection, bool)
            assert config.highlight_detection is True
            assert isinstance(config.awk_fallback, bool)
            assert config.awk_fallback is False

            # Test float conversion
            assert isinstance(config.gpu_memory_fraction, float)
            assert config.gpu_memory_fraction == 0.75

    def test_configuration_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = UnifiedConfig()
        config.model_type = ModelType.LLAMA32_VISION
        config.confidence_threshold = 0.8
        config.highlight_detection = False

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "llama32_vision"
        assert config_dict["confidence_threshold"] == 0.8
        assert config_dict["highlight_detection"] is False

    def test_configuration_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_type": "internvl3",
            "processing_pipeline": "7step",
            "confidence_threshold": 0.9,
            "highlight_detection": True,
            "awk_fallback": False,
            "batch_size": 2,
        }

        config = UnifiedConfig.from_dict(config_dict)

        assert config.model_type == ModelType.INTERNVL3
        assert config.processing_pipeline == ProcessingPipeline.SEVEN_STEP
        assert config.confidence_threshold == 0.9
        assert config.highlight_detection is True
        assert config.awk_fallback is False
        assert config.batch_size == 2

    def test_config_update(self):
        """Test configuration update method."""
        config = UnifiedConfig()

        updates = {
            "model_type": ModelType.LLAMA32_VISION,
            "confidence_threshold": 0.85,
            "highlight_detection": False,
        }

        config.update(updates)

        assert config.model_type == ModelType.LLAMA32_VISION
        assert config.confidence_threshold == 0.85
        assert config.highlight_detection is False

    def test_cross_platform_compatibility(self):
        """Test cross-platform path handling."""
        config = UnifiedConfig()

        # Test Windows-style path
        windows_path = "C:\\Users\\test\\dataset"
        config.dataset_path = windows_path

        # Test Unix-style path
        unix_path = "/home/test/dataset"
        config.dataset_path = unix_path

        # Should handle both without errors
        assert config.dataset_path is not None

    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        config = UnifiedConfig()

        # Test invalid confidence threshold
        with pytest.raises(
            ValueError, match="Confidence threshold must be between 0.0 and 1.0"
        ):
            config.confidence_threshold = 2.0

        # Test invalid quality threshold
        with pytest.raises(
            ValueError, match="Quality threshold must be between 0.0 and 1.0"
        ):
            config.quality_threshold = -0.5

        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            config.batch_size = 0

        # Test invalid max workers
        with pytest.raises(ValueError, match="Max workers must be positive"):
            config.max_workers = -1

    def test_llama_specific_settings(self):
        """Test Llama-specific configuration settings."""
        config = UnifiedConfig()
        config.model_type = ModelType.LLAMA32_VISION

        # Llama-specific settings
        config.graceful_degradation = True
        config.confidence_components = 4
        config.production_assessment = ProductionAssessment.FIVE_LEVEL

        assert config.graceful_degradation is True
        assert config.confidence_components == 4
        assert config.production_assessment == ProductionAssessment.FIVE_LEVEL

    def test_internvl_specific_settings(self):
        """Test InternVL-specific configuration settings."""
        config = UnifiedConfig()
        config.model_type = ModelType.INTERNVL3

        # InternVL-specific settings
        config.multi_gpu_dev = True
        config.single_gpu_prod = True
        config.enable_8bit_quantization = True
        config.computer_vision = True

        assert config.multi_gpu_dev is True
        assert config.single_gpu_prod is True
        assert config.enable_8bit_quantization is True
        assert config.computer_vision is True
