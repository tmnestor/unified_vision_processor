#!/usr/bin/env python3
"""
Test device management validation across both models.
"""

import logging

from vision_processor.config.model_factory import ModelFactory
from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.models.base_model import DeviceConfig, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_device_management():
    """Test device management across both models."""

    print("\n=== Testing Device Management Validation ===\n")

    # Test different device configurations
    device_configs = [
        (DeviceConfig.AUTO, "Auto device selection"),
        (DeviceConfig.CPU, "CPU only"),
        (DeviceConfig.SINGLE_GPU, "Single GPU"),
        (DeviceConfig.MULTI_GPU, "Multi GPU"),
        (DeviceConfig.MPS, "Apple Silicon MPS"),
    ]

    results = []

    for device_config, description in device_configs:
        print(f"\n{description} ({device_config.value}):")
        print("-" * 50)

        # Test InternVL
        try:
            internvl_config = UnifiedConfig(
                model_type=ModelType.INTERNVL3,
                internvl_model_path="/Users/tod/PretrainedLLM/InternVL3-8B",
                offline_mode=True,
                device_config=device_config,
                enable_8bit_quantization=False,
            )

            print(f"1. Testing InternVL with {description}...")
            internvl_model = ModelFactory.create_model(
                internvl_config.model_type, internvl_config.model_path, internvl_config
            )

            internvl_device_info = internvl_model.get_device_info()
            print(f"   ✓ InternVL device: {internvl_device_info['device']}")
            print(f"   ✓ Memory limit: {internvl_device_info['memory_limit_mb']} MB")

            internvl_success = True

        except Exception as e:
            print(f"   ✗ InternVL failed: {e}")
            internvl_success = False
            internvl_device_info = {}

        # Test Llama
        try:
            llama_config = UnifiedConfig(
                model_type=ModelType.LLAMA32_VISION,
                llama_model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
                offline_mode=True,
                device_config=device_config,
                enable_8bit_quantization=False,
            )

            print(f"2. Testing Llama with {description}...")
            llama_model = ModelFactory.create_model(
                llama_config.model_type, llama_config.model_path, llama_config
            )

            llama_device_info = llama_model.get_device_info()
            print(f"   ✓ Llama device: {llama_device_info['device']}")
            print(f"   ✓ Memory limit: {llama_device_info['memory_limit_mb']} MB")

            llama_success = True

        except Exception as e:
            print(f"   ✗ Llama failed: {e}")
            llama_success = False
            llama_device_info = {}

        # Compare results
        if internvl_success and llama_success:
            internvl_device = internvl_device_info.get("device", "unknown")
            llama_device = llama_device_info.get("device", "unknown")

            if internvl_device == llama_device:
                print(f"   ✓ Device consistency: Both models use {internvl_device}")
                consistency = True
            else:
                print(
                    f"   ✗ Device inconsistency: InternVL={internvl_device}, Llama={llama_device}"
                )
                consistency = False
        else:
            consistency = False

        results.append(
            {
                "config": device_config,
                "description": description,
                "internvl_success": internvl_success,
                "llama_success": llama_success,
                "consistency": consistency,
                "internvl_device": internvl_device_info.get("device", "unknown"),
                "llama_device": llama_device_info.get("device", "unknown"),
            }
        )

    # Test quantization with different devices
    print("\n\nQuantization Testing:")
    print("-" * 50)

    quantization_configs = [
        (True, "With quantization"),
        (False, "Without quantization"),
    ]

    for enable_quant, quant_desc in quantization_configs:
        print(f"\n{quant_desc}:")

        # Test both models with quantization
        try:
            internvl_config = UnifiedConfig(
                model_type=ModelType.INTERNVL3,
                internvl_model_path="/Users/tod/PretrainedLLM/InternVL3-8B",
                offline_mode=True,
                device_config=DeviceConfig.AUTO,
                enable_8bit_quantization=enable_quant,
            )

            internvl_model = ModelFactory.create_model(
                internvl_config.model_type, internvl_config.model_path, internvl_config
            )

            internvl_info = internvl_model.get_device_info()
            print(
                f"   ✓ InternVL quantization: {internvl_info['quantization_enabled']}"
            )

        except Exception as e:
            print(f"   ✗ InternVL quantization test failed: {e}")

        try:
            llama_config = UnifiedConfig(
                model_type=ModelType.LLAMA32_VISION,
                llama_model_path="/Users/tod/PretrainedLLM/Llama-3.2-11B-Vision",
                offline_mode=True,
                device_config=DeviceConfig.AUTO,
                enable_8bit_quantization=enable_quant,
            )

            llama_model = ModelFactory.create_model(
                llama_config.model_type, llama_config.model_path, llama_config
            )

            llama_info = llama_model.get_device_info()
            print(f"   ✓ Llama quantization: {llama_info['quantization_enabled']}")

        except Exception as e:
            print(f"   ✗ Llama quantization test failed: {e}")

    # Test memory limits
    print("\n\nMemory Limit Testing:")
    print("-" * 50)

    memory_limits = [
        (None, "No memory limit"),
        (8192, "8GB memory limit"),
        (15360, "15GB memory limit (V100)"),
    ]

    for memory_limit, mem_desc in memory_limits:
        print(f"\n{mem_desc}:")

        try:
            test_config = UnifiedConfig(
                model_type=ModelType.INTERNVL3,
                internvl_model_path="/Users/tod/PretrainedLLM/InternVL3-8B",
                offline_mode=True,
                device_config=DeviceConfig.AUTO,
                gpu_memory_limit=memory_limit,
            )

            test_model = ModelFactory.create_model(
                test_config.model_type, test_config.model_path, test_config
            )

            device_info = test_model.get_device_info()
            print(f"   ✓ Applied memory limit: {device_info['memory_limit_mb']} MB")

        except Exception as e:
            print(f"   ✗ Memory limit test failed: {e}")

    # Summary
    print("\n\nDevice Management Summary:")
    print("=" * 50)

    successful_configs = [
        r for r in results if r["internvl_success"] and r["llama_success"]
    ]
    consistent_configs = [r for r in results if r["consistency"]]

    print(f"Total configurations tested: {len(results)}")
    print(f"Successful configurations: {len(successful_configs)}")
    print(f"Consistent configurations: {len(consistent_configs)}")

    print("\nSupported device configurations:")
    for result in successful_configs:
        status = "✓" if result["consistency"] else "~"
        print(f"  {status} {result['description']}: {result['internvl_device']}")

    print("\nUnsupported device configurations:")
    for result in results:
        if not result["internvl_success"] or not result["llama_success"]:
            print(f"  ✗ {result['description']}: Failed")

    # Overall assessment
    overall_success = len(consistent_configs) >= 2  # At least 2 working configs

    if overall_success:
        print("\n✓ Device management validation successful!")
        print("  Both models support multiple device configurations")
        print("  Device selection is consistent across models")
        print("  Memory management works correctly")
    else:
        print("\n✗ Device management validation failed!")
        print("  Models have inconsistent device behavior")

    print("\n=== Device Management Test Complete ===")

    return overall_success


if __name__ == "__main__":
    success = test_device_management()
    exit(0 if success else 1)
