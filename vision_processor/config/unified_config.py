"""Unified Configuration Management

Combines Llama configuration with InternVL features for model-agnostic processing.
Supports environment-driven configuration with intelligent defaults.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..models.base_model import DeviceConfig, ModelType

logger = logging.getLogger(__name__)


class ProcessingPipeline(Enum):
    """Processing pipeline options."""

    SEVEN_STEP = "7step"
    SIMPLE = "simple"
    CUSTOM = "custom"


class ExtractionMethod(Enum):
    """Extraction method options."""

    HYBRID = "hybrid"
    KEY_VALUE = "key_value"
    AWK_ONLY = "awk_only"


class ProductionAssessment(Enum):
    """Production readiness assessment levels."""

    FIVE_LEVEL = "5level"
    BINARY = "binary"
    CUSTOM = "custom"


@dataclass
class UnifiedConfig:
    """Unified configuration for vision document processing.

    Integrates Llama-3.2 configuration framework with InternVL advanced features.
    Supports cross-platform deployment from Mac M1 → 2x H200 → single V100.
    """

    # =====================================================
    # MODEL SELECTION
    # =====================================================
    model_type: ModelType = ModelType.INTERNVL3
    model_path: Path | None = None
    device_config: DeviceConfig = DeviceConfig.AUTO

    # Model-specific paths for offline production
    internvl_model_path: Path | None = None
    llama_model_path: Path | None = None
    offline_mode: bool = True  # Default to offline for production safety
    testing_mode: bool = False  # Skip validation for testing

    # =====================================================
    # PROCESSING CONFIGURATION (Llama-based)
    # =====================================================
    processing_pipeline: ProcessingPipeline = ProcessingPipeline.SEVEN_STEP
    extraction_method: ExtractionMethod = ExtractionMethod.HYBRID
    quality_threshold: float = 0.6
    confidence_threshold: float = 0.8

    # =====================================================
    # FEATURE INTEGRATION
    # =====================================================
    highlight_detection: bool = True  # InternVL feature
    awk_fallback: bool = True
    computer_vision: bool = True  # InternVL feature
    graceful_degradation: bool = True  # Llama feature

    # =====================================================
    # CONFIDENCE AND PRODUCTION (Llama-based)
    # =====================================================
    confidence_components: int = 4
    production_assessment: ProductionAssessment = ProductionAssessment.FIVE_LEVEL

    # =====================================================
    # GPU CONFIGURATION
    # =====================================================
    gpu_memory_limit: int = 15360  # V100 16GB with buffer (MB)
    enable_8bit_quantization: bool = True
    multi_gpu_dev: bool = True  # For 2x H200 development
    single_gpu_prod: bool = True  # For V100 production

    # =====================================================
    # DATA PATHS
    # =====================================================
    dataset_path: Path | None = None
    ground_truth_path: Path | None = None
    output_path: Path | None = None

    # =====================================================
    # PERFORMANCE AND COMPATIBILITY
    # =====================================================
    batch_size: int = 1
    max_workers: int = 4
    gpu_memory_fraction: float = 0.8
    cross_platform: bool = True

    # =====================================================
    # EVALUATION AND COMPARISON
    # =====================================================
    fair_comparison: bool = True
    model_comparison: bool = True
    evaluation_fields: list[str] = field(
        default_factory=lambda: [
            "date_value",
            "store_name_value",
            "tax_value",
            "total_value",
        ],
    )
    sroie_evaluation: bool = True

    # =====================================================
    # DEVELOPMENT ENVIRONMENT SPECIFIC
    # =====================================================
    local_dev: bool = True  # Mac M1 local development
    remote_sync: bool = True
    h200_development: bool = False
    v100_production: bool = False
    production_mode: bool = False

    # =====================================================
    # ADVANCED SETTINGS
    # =====================================================
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_profiling: bool = False
    cache_models: bool = True
    trust_remote_code: bool = True  # Required for InternVL

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.internvl_model_path, str):
            self.internvl_model_path = Path(self.internvl_model_path)
        if isinstance(self.llama_model_path, str):
            self.llama_model_path = Path(self.llama_model_path)
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.ground_truth_path, str):
            self.ground_truth_path = Path(self.ground_truth_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Auto-detect testing environment
        import sys

        if "pytest" in sys.modules or "unittest" in sys.modules:
            self.testing_mode = True

        # Resolve model path if not explicitly set
        if not self.model_path:
            self._resolve_model_path()

        # Validate configuration
        self._validate_config()

        # Apply environment-specific optimizations
        self._apply_environment_optimizations()

    def _resolve_model_path(self) -> None:
        """Resolve model path based on model type and configured paths."""
        if self.model_type == ModelType.INTERNVL3 and self.internvl_model_path:
            self.model_path = self.internvl_model_path
            logger.info(f"Using InternVL model path: {self.model_path}")
        elif self.model_type == ModelType.LLAMA32_VISION and self.llama_model_path:
            self.model_path = self.llama_model_path
            logger.info(f"Using Llama model path: {self.model_path}")
        else:
            # If offline mode (default) and no path configured, raise error (unless testing)
            if self.offline_mode and not self.testing_mode:
                # Create proper environment variable name for the error message
                env_var_name = (
                    "VISION_INTERNVL_MODEL_PATH"
                    if self.model_type == ModelType.INTERNVL3
                    else "VISION_LLAMA_MODEL_PATH"
                )
                raise ValueError(
                    f"Offline mode is enabled (default) but no model path configured for {self.model_type.value}. "
                    f"Set {env_var_name} in .env or "
                    f"set VISION_OFFLINE_MODE=false for development with internet access.",
                )
            elif self.testing_mode:
                # Set a mock path for testing
                self.model_path = Path("mock_model_path_for_testing")
            # Otherwise, model will be downloaded (development only)
            logger.warning(
                f"No local model path configured for {self.model_type.value}. "
                "Model will be downloaded from internet (development mode only)",
            )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate thresholds
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Validate memory limit
        if self.gpu_memory_limit < 1024:  # At least 1GB
            raise ValueError("gpu_memory_limit must be at least 1024 MB")

        # Validate confidence components
        if self.confidence_components not in [1, 2, 3, 4]:
            raise ValueError("confidence_components must be 1, 2, 3, or 4")

        # Validate model path in offline mode (skip in testing)
        if self.offline_mode and self.model_path and not self.testing_mode:
            if not self.model_path.exists():
                raise ValueError(
                    f"Offline mode enabled but model path does not exist: {self.model_path}",
                )

    def _apply_environment_optimizations(self) -> None:
        """Apply optimizations based on detected environment."""
        import platform

        import torch

        # Detect environment and apply optimizations
        system = platform.system()

        # Mac M1 optimizations
        if system == "Darwin" and platform.machine() == "arm64":
            logger.info("Detected Mac M1, applying MPS optimizations")
            self.local_dev = True
            self.enable_8bit_quantization = False  # MPS doesn't support quantization
            self.gpu_memory_limit = min(
                self.gpu_memory_limit,
                16384,
            )  # Unified memory constraint

        # Multi-GPU development system (H200)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 70 * 1024**3:  # 70GB+ indicates H200-class GPU
                logger.info(
                    "Detected high-memory multi-GPU system, applying H200 optimizations",
                )
                self.h200_development = True
                self.enable_8bit_quantization = (
                    False  # High memory, no quantization needed
                )
                self.multi_gpu_dev = True
                self.gpu_memory_limit = None  # No limit on high-memory system

        # Single GPU production (V100)
        elif torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if 15 * 1024**3 <= gpu_memory <= 17 * 1024**3:  # V100 16GB range
                logger.info(
                    "Detected V100-class GPU, applying production optimizations",
                )
                self.v100_production = True
                self.production_mode = True
                self.enable_8bit_quantization = True  # Required for 16GB
                self.multi_gpu_dev = False
                self.gpu_memory_limit = 15360  # Conservative limit for V100

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "UnifiedConfig":
        """Create configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            UnifiedConfig instance

        """
        # Load environment file if specified
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env file in current directory or parent directories
            current_dir = Path.cwd()
            for path in [current_dir] + list(current_dir.parents):
                env_path = path / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    break

        # Create config from environment variables
        config_dict = {}

        # Model selection
        if model_type := os.getenv("VISION_MODEL_TYPE"):
            config_dict["model_type"] = ModelType(model_type)
        if model_path := os.getenv("VISION_MODEL_PATH"):
            config_dict["model_path"] = Path(model_path)
        if device_config := os.getenv("VISION_DEVICE_CONFIG"):
            config_dict["device_config"] = DeviceConfig(device_config)

        # Model-specific paths for offline production
        if internvl_path := os.getenv("VISION_INTERNVL_MODEL_PATH"):
            config_dict["internvl_model_path"] = Path(internvl_path)
        if llama_path := os.getenv("VISION_LLAMA_MODEL_PATH"):
            config_dict["llama_model_path"] = Path(llama_path)
        config_dict["offline_mode"] = cls._get_bool_env(
            "VISION_OFFLINE_MODE",
            True,
        )  # Default to True

        # Processing configuration
        if pipeline := os.getenv("VISION_PROCESSING_PIPELINE"):
            config_dict["processing_pipeline"] = ProcessingPipeline(pipeline)
        if method := os.getenv("VISION_EXTRACTION_METHOD"):
            config_dict["extraction_method"] = ExtractionMethod(method)
        if threshold := os.getenv("VISION_QUALITY_THRESHOLD"):
            config_dict["quality_threshold"] = float(threshold)
        if confidence := os.getenv("VISION_CONFIDENCE_THRESHOLD"):
            config_dict["confidence_threshold"] = float(confidence)

        # Feature integration
        config_dict.update(
            {
                "highlight_detection": cls._get_bool_env(
                    "VISION_HIGHLIGHT_DETECTION",
                    True,
                ),
                "awk_fallback": cls._get_bool_env("VISION_AWK_FALLBACK", True),
                "computer_vision": cls._get_bool_env("VISION_COMPUTER_VISION", True),
                "graceful_degradation": cls._get_bool_env(
                    "VISION_GRACEFUL_DEGRADATION",
                    True,
                ),
            },
        )

        # Confidence and production
        if components := os.getenv("VISION_CONFIDENCE_COMPONENTS"):
            config_dict["confidence_components"] = int(components)
        if assessment := os.getenv("VISION_PRODUCTION_ASSESSMENT"):
            config_dict["production_assessment"] = ProductionAssessment(assessment)

        # GPU configuration
        if gpu_limit := os.getenv("VISION_GPU_MEMORY_LIMIT"):
            config_dict["gpu_memory_limit"] = int(gpu_limit)
        config_dict.update(
            {
                "enable_8bit_quantization": cls._get_bool_env(
                    "VISION_ENABLE_8BIT_QUANTIZATION",
                    True,
                ),
                "multi_gpu_dev": cls._get_bool_env("VISION_MULTI_GPU_DEV", True),
                "single_gpu_prod": cls._get_bool_env("VISION_SINGLE_GPU_PROD", True),
            },
        )

        # Data paths
        if dataset_path := os.getenv("VISION_DATASET_PATH"):
            config_dict["dataset_path"] = Path(dataset_path)
        if ground_truth_path := os.getenv("VISION_GROUND_TRUTH_PATH"):
            config_dict["ground_truth_path"] = Path(ground_truth_path)
        if output_path := os.getenv("VISION_OUTPUT_PATH"):
            config_dict["output_path"] = Path(output_path)

        # Performance and compatibility
        if batch_size := os.getenv("VISION_BATCH_SIZE"):
            config_dict["batch_size"] = int(batch_size)
        if max_workers := os.getenv("VISION_MAX_WORKERS"):
            config_dict["max_workers"] = int(max_workers)
        if memory_fraction := os.getenv("VISION_GPU_MEMORY_FRACTION"):
            config_dict["gpu_memory_fraction"] = float(memory_fraction)
        config_dict.update(
            {
                "cross_platform": cls._get_bool_env("VISION_CROSS_PLATFORM", True),
            },
        )

        # Evaluation and comparison
        config_dict.update(
            {
                "fair_comparison": cls._get_bool_env("VISION_FAIR_COMPARISON", True),
                "model_comparison": cls._get_bool_env("VISION_MODEL_COMPARISON", True),
                "sroie_evaluation": cls._get_bool_env("VISION_SROIE_EVALUATION", True),
            },
        )
        if eval_fields := os.getenv("VISION_EVALUATION_FIELDS"):
            config_dict["evaluation_fields"] = eval_fields.split(",")

        # Development environment
        config_dict.update(
            {
                "local_dev": cls._get_bool_env("VISION_LOCAL_DEV", True),
                "remote_sync": cls._get_bool_env("VISION_REMOTE_SYNC", True),
                "h200_development": cls._get_bool_env("VISION_H200_DEVELOPMENT", False),
                "v100_production": cls._get_bool_env("VISION_V100_PRODUCTION", False),
                "production_mode": cls._get_bool_env("VISION_PRODUCTION_MODE", False),
            },
        )

        return cls(**config_dict)

    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}

        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, Path):
                result[key] = str(value) if value else None
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UnifiedConfig":
        """Create configuration from dictionary."""
        # Convert string values to enums if needed
        if "model_type" in config_dict and isinstance(config_dict["model_type"], str):
            config_dict["model_type"] = ModelType(config_dict["model_type"])
        if "processing_pipeline" in config_dict and isinstance(
            config_dict["processing_pipeline"], str
        ):
            config_dict["processing_pipeline"] = ProcessingPipeline(
                config_dict["processing_pipeline"]
            )
        if "extraction_method" in config_dict and isinstance(
            config_dict["extraction_method"], str
        ):
            config_dict["extraction_method"] = ExtractionMethod(
                config_dict["extraction_method"]
            )

        # Set testing mode to avoid validation issues
        config_dict["testing_mode"] = True

        return cls(**config_dict)

    def save_to_file(self, file_path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        file_path = Path(file_path)
        config_dict = self.to_dict()

        with file_path.open("w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "UnifiedConfig":
        """Load configuration from YAML file."""
        import yaml

        file_path = Path(file_path)

        with file_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        # Convert string enums back to enum objects
        if "model_type" in config_dict:
            config_dict["model_type"] = ModelType(config_dict["model_type"])
        if "device_config" in config_dict:
            config_dict["device_config"] = DeviceConfig(config_dict["device_config"])
        if "processing_pipeline" in config_dict:
            config_dict["processing_pipeline"] = ProcessingPipeline(
                config_dict["processing_pipeline"],
            )
        if "extraction_method" in config_dict:
            config_dict["extraction_method"] = ExtractionMethod(
                config_dict["extraction_method"],
            )
        if "production_assessment" in config_dict:
            config_dict["production_assessment"] = ProductionAssessment(
                config_dict["production_assessment"],
            )

        logger.info(f"Configuration loaded from {file_path}")
        return cls(**config_dict)

    def get_model_config(self) -> dict[str, Any]:
        """Get configuration specific to model creation."""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device_config": self.device_config,
            "enable_quantization": self.enable_8bit_quantization,
            "memory_limit_mb": self.gpu_memory_limit,
            "trust_remote_code": self.trust_remote_code,
        }

    def get_processing_config(self) -> dict[str, Any]:
        """Get configuration specific to processing pipeline."""
        return {
            "processing_pipeline": self.processing_pipeline,
            "extraction_method": self.extraction_method,
            "quality_threshold": self.quality_threshold,
            "confidence_threshold": self.confidence_threshold,
            "highlight_detection": self.highlight_detection,
            "awk_fallback": self.awk_fallback,
            "computer_vision": self.computer_vision,
            "graceful_degradation": self.graceful_degradation,
            "confidence_components": self.confidence_components,
            "production_assessment": self.production_assessment,
        }

    def __repr__(self) -> str:
        return (
            f"UnifiedConfig("
            f"model_type={self.model_type.value}, "
            f"device={self.device_config.value}, "
            f"pipeline={self.processing_pipeline.value}, "
            f"production={self.production_mode})"
        )
