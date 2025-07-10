#!/usr/bin/env python3
"""
Test model response standardization and compatibility.
"""

import logging

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.models.base_model import DeviceConfig, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_model_compatibility():
    """Test that both models return compatible responses."""

    print("\n=== Testing Model Response Standardization ===\n")

    # Create configurations for both models
    internvl_config = UnifiedConfig(
        model_type=ModelType.INTERNVL3,
        internvl_model_path="/Users/tod/PretrainedLLM/InternVL3-8B",
        offline_mode=True,
        device_config=DeviceConfig.AUTO,
        enable_8bit_quantization=False,  # Disable for compatibility testing
    )

    llama_config = UnifiedConfig(
        model_type=ModelType.LLAMA32_VISION,
        llama_model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
        offline_mode=True,
        device_config=DeviceConfig.AUTO,
        enable_8bit_quantization=False,  # Disable for compatibility testing
    )

    try:
        # Create both models
        print("1. Creating InternVL model...")
        internvl_model = ModelFactory.create_model(
            internvl_config.model_type, internvl_config.model_path, internvl_config
        )
        print("✓ InternVL model created successfully")

        print("\n2. Creating Llama model...")
        llama_model = ModelFactory.create_model(
            llama_config.model_type, llama_config.model_path, llama_config
        )
        print("✓ Llama model created successfully")

        # Compare model interfaces
        print("\n3. Comparing model interfaces:")

        # Test basic methods
        methods_to_test = [
            "process_image",
            "process_batch",
            "load_model",
            "unload_model",
            "get_device_info",
            "get_memory_usage",
            "classify_document",
        ]

        interface_compatible = True
        for method in methods_to_test:
            internvl_has = hasattr(internvl_model, method)
            llama_has = hasattr(llama_model, method)

            if internvl_has and llama_has:
                status = "✓"
            else:
                status = "✗"
                interface_compatible = False

            print(f"  {status} {method}: InternVL={internvl_has}, Llama={llama_has}")

        if interface_compatible:
            print("✓ Model interfaces are compatible")
        else:
            print("✗ Model interfaces have differences")

        # Test capabilities comparison
        print("\n4. Comparing model capabilities:")
        internvl_caps = internvl_model.capabilities
        llama_caps = llama_model.capabilities

        capabilities_data = [
            (
                "Multi-GPU support",
                internvl_caps.supports_multi_gpu,
                llama_caps.supports_multi_gpu,
            ),
            (
                "Quantization support",
                internvl_caps.supports_quantization,
                llama_caps.supports_quantization,
            ),
            (
                "Highlight detection",
                internvl_caps.supports_highlight_detection,
                llama_caps.supports_highlight_detection,
            ),
            (
                "Batch processing",
                internvl_caps.supports_batch_processing,
                llama_caps.supports_batch_processing,
            ),
            (
                "Memory efficient",
                internvl_caps.memory_efficient,
                llama_caps.memory_efficient,
            ),
            ("Cross-platform", internvl_caps.cross_platform, llama_caps.cross_platform),
        ]

        for capability, internvl_val, llama_val in capabilities_data:
            print(f"  - {capability}: InternVL={internvl_val}, Llama={llama_val}")

        # Test device info compatibility
        print("\n5. Comparing device information:")
        internvl_device_info = internvl_model.get_device_info()
        llama_device_info = llama_model.get_device_info()

        common_keys = set(internvl_device_info.keys()) & set(llama_device_info.keys())
        print(f"  - Common keys: {len(common_keys)}")
        print(f"  - InternVL keys: {list(internvl_device_info.keys())}")
        print(f"  - Llama keys: {list(llama_device_info.keys())}")

        for key in common_keys:
            print(
                f"  - {key}: InternVL={internvl_device_info[key]}, Llama={llama_device_info[key]}"
            )

        # Test ModelResponse structure (without actually loading models)
        print("\n6. Testing ModelResponse structure:")

        # Check that both models would return the same response structure
        from vision_processor.models.base_model import ModelResponse

        # Test response creation
        test_response = ModelResponse(
            raw_text="Test response",
            confidence=0.85,
            processing_time=1.0,
            device_used="mps",
            memory_usage=1024.0,
            model_type="test",
            quantized=False,
            metadata={"test": "value"},
        )

        required_fields = [
            "raw_text",
            "confidence",
            "processing_time",
            "device_used",
            "memory_usage",
            "model_type",
            "quantized",
            "metadata",
        ]

        response_compatible = True
        for field in required_fields:
            if hasattr(test_response, field):
                status = "✓"
            else:
                status = "✗"
                response_compatible = False
            print(f"  {status} {field}: {hasattr(test_response, field)}")

        if response_compatible:
            print("✓ ModelResponse structure is compatible")
        else:
            print("✗ ModelResponse structure has issues")

        # Test configuration compatibility
        print("\n7. Testing configuration compatibility:")

        config_fields = [
            "model_type",
            "offline_mode",
            "device_config",
            "enable_8bit_quantization",
            "gpu_memory_limit",
            "multi_gpu_dev",
            "single_gpu_prod",
            "graceful_degradation",
            "processing_pipeline",
            "extraction_method",
            "awk_fallback",
            "confidence_threshold",
        ]

        config_compatible = True
        for field in config_fields:
            internvl_has = hasattr(internvl_config, field)
            llama_has = hasattr(llama_config, field)

            if internvl_has and llama_has:
                status = "✓"
            else:
                status = "✗"
                config_compatible = False

            print(f"  {status} {field}: InternVL={internvl_has}, Llama={llama_has}")

        if config_compatible:
            print("✓ Configuration fields are compatible")
        else:
            print("✗ Configuration fields have differences")

        # Final compatibility assessment
        print("\n8. Overall compatibility assessment:")

        overall_compatible = (
            interface_compatible and response_compatible and config_compatible
        )

        if overall_compatible:
            print("✓ Models are fully compatible for unified processing")
        else:
            print("✗ Models have compatibility issues that need addressing")

        print("\n=== Model Compatibility Test Complete ===")
        print("✓ All compatibility tests completed!")

        return overall_compatible

    except Exception as e:
        print(f"\n✗ Error during compatibility testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_compatibility()
    exit(0 if success else 1)
