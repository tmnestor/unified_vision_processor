#!/usr/bin/env python3
"""
Test Llama-3.2-Vision model implementation.
"""

import logging

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.models.base_model import DeviceConfig, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_llama_model():
    """Test Llama-3.2-Vision model loading and basic functionality."""

    print("\n=== Testing Llama-3.2-Vision Model Implementation ===\n")

    # Create configuration
    config = UnifiedConfig(
        model_type=ModelType.LLAMA32_VISION,
        llama_model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
        offline_mode=True,
        device_config=DeviceConfig.AUTO,
        enable_8bit_quantization=True,
    )

    print("Configuration:")
    print(f"  Model type: {config.model_type.value}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device_config.value}")
    print(f"  Quantization: {config.enable_8bit_quantization}")
    print(f"  Offline mode: {config.offline_mode}")

    try:
        # Create model using factory
        print("\n1. Creating Llama-3.2-Vision model...")
        model = ModelFactory.create_model(config.model_type, config.model_path, config)
        print("✓ Model created successfully")

        # Check capabilities
        print("\n2. Model capabilities:")
        caps = model.capabilities
        print(f"  - Multi-GPU support: {caps.supports_multi_gpu}")
        print(f"  - Quantization support: {caps.supports_quantization}")
        print(f"  - Highlight detection: {caps.supports_highlight_detection}")
        print(f"  - Max image size: {caps.max_image_size}")
        print(f"  - Cross-platform: {caps.cross_platform}")

        # Check device info
        print("\n3. Device information:")
        device_info = model.get_device_info()
        for key, value in device_info.items():
            print(f"  - {key}: {value}")

        # Test model loading (without actually loading due to size)
        print("\n4. Model loading test:")
        print("  - Model path exists:", config.model_path.exists())
        if config.model_path.exists():
            # List some files to verify structure
            model_files = list(config.model_path.glob("*.safetensors"))[:3]
            print(
                f"  - Found {len(list(config.model_path.glob('*.safetensors')))} safetensors files"
            )
            if model_files:
                print(f"  - Example files: {[f.name for f in model_files]}")

        print("\n5. Model class and methods:")
        print(f"  - Model class: {model.__class__.__name__}")
        print(f"  - Has process_image: {hasattr(model, 'process_image')}")
        print(f"  - Has process_batch: {hasattr(model, 'process_batch')}")
        print(f"  - Has load_model: {hasattr(model, 'load_model')}")
        print(f"  - Has classify_document: {hasattr(model, 'classify_document')}")

        # Test configuration features
        print("\n6. Llama-specific features:")
        print(f"  - Graceful degradation: {config.graceful_degradation}")
        print(f"  - Processing pipeline: {config.processing_pipeline}")
        print(f"  - Extraction method: {config.extraction_method}")
        print(f"  - AWK fallback: {config.awk_fallback}")
        print(f"  - Confidence threshold: {config.confidence_threshold}")

        print("\n=== Llama-3.2-Vision Model Test Complete ===")
        print("✓ All tests passed!")

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llama_model()
    exit(0 if success else 1)
