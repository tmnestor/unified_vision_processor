#!/usr/bin/env python3
"""
Test that offline mode is true by default and loads from .env.
"""

from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.models.base_model import ModelType


def test_offline_default():
    """Test offline mode default behavior with .env configuration."""

    print("\n=== Testing Offline Mode Default with .env ===\n")

    # Load config from .env file
    config = UnifiedConfig.from_env()

    print("Loaded from .env:")
    print(f"  offline_mode: {config.offline_mode} (default: True)")
    print(f"  model_type: {config.model_type.value}")
    print(f"  internvl_model_path: {config.internvl_model_path}")
    print(f"  llama_model_path: {config.llama_model_path}")
    print(f"  resolved model_path: {config.model_path}")

    # Verify offline mode is True by default
    assert config.offline_mode, "offline_mode should default to True"

    # Verify paths are loaded from .env
    assert config.internvl_model_path.exists(), (
        f"InternVL path should exist: {config.internvl_model_path}"
    )
    assert config.llama_model_path.exists(), (
        f"Llama path should exist: {config.llama_model_path}"
    )
    assert config.model_path.exists(), (
        f"Resolved model path should exist: {config.model_path}"
    )

    print("\n✓ All paths loaded correctly from .env")
    print("✓ Offline mode is True by default")

    # Test that it raises error without model paths
    print("\nTesting error when model paths not set...")
    try:
        config.model_type = ModelType.INTERNVL3
        config.model_path = None
        config._resolve_model_path()
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test with offline_mode explicitly set to False
    print("\n--- Testing with offline_mode=False ---")
    config_online = UnifiedConfig(offline_mode=False)
    print(f"offline_mode: {config_online.offline_mode}")

    # Should not raise error
    config_online.model_type = ModelType.INTERNVL3
    config_online.model_path = None
    config_online._resolve_model_path()
    print("✓ No error raised in online mode without model path")

    print("\n=== Offline Mode Default Test Complete ===")


if __name__ == "__main__":
    test_offline_default()
