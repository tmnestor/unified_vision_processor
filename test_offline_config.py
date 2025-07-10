#!/usr/bin/env python3
"""
Test offline configuration and model path loading.
"""

from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.models.base_model import ModelType


def test_offline_config():
    """Test offline configuration loading."""

    print("\n=== Testing Offline Configuration ===\n")

    # Load configuration from .env
    config = UnifiedConfig.from_env()

    print(f"Configuration loaded: {config}")
    print(f"\nOffline mode: {config.offline_mode}")
    print(f"Model type: {config.model_type.value}")
    print(f"Model path: {config.model_path}")
    print(f"InternVL path: {config.internvl_model_path}")
    print(f"Llama path: {config.llama_model_path}")

    # Check if paths exist
    if config.model_path:
        exists = config.model_path.exists()
        print(f"\nResolved model path exists: {exists}")
        if exists:
            print(
                f"Model directory contents: {list(config.model_path.iterdir())[:5]}..."
            )

    # Test model path resolution for both types
    print("\n--- Testing Model Path Resolution ---")

    # Test InternVL
    internvl_config = UnifiedConfig.from_env()
    internvl_config.model_type = ModelType.INTERNVL3
    internvl_config.model_path = None  # Force resolution
    internvl_config._resolve_model_path()
    print(f"\nInternVL resolved path: {internvl_config.model_path}")

    # Test Llama
    llama_config = UnifiedConfig.from_env()
    llama_config.model_type = ModelType.LLAMA32_VISION
    llama_config.model_path = None  # Force resolution
    llama_config._resolve_model_path()
    print(f"Llama resolved path: {llama_config.model_path}")

    print("\n=== Offline Configuration Test Complete ===")


if __name__ == "__main__":
    test_offline_config()
