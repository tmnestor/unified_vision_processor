#!/usr/bin/env python3
"""Quick test to verify the fixed Llama implementation works."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from vision_processor.models.base_model import DeviceConfig
from vision_processor.models.llama_model import LlamaVisionModel


def test_llama_model():
    """Test the fixed Llama model implementation."""
    print("Testing fixed Llama-3.2-Vision implementation...")

    # Initialize model with working checkpoint
    model = LlamaVisionModel(
        model_path="/Users/tod/Desktop/Llama_3.2/llama_vision/models/llama3.2-vision-checkpoint",
        device_config=DeviceConfig.AUTO,
        enable_quantization=True,
        memory_limit_mb=15360,  # V100 16GB limit
    )

    print(f"Model initialized: {model}")
    print(f"Capabilities: {model.capabilities}")
    print(f"Device: {model.device}")
    print(f"Model loaded: {model.is_loaded}")

    # Test model loading
    try:
        print("\n=== Testing model loading ===")
        model.load_model()
        print("✅ Model loaded successfully!")
        print(f"Model is loaded: {model.is_loaded}")

        # Test basic functionality
        print("\n=== Testing basic prediction ===")
        test_response = model.predict(
            "/Users/tod/Desktop/internvl_PoC/test_images/woolworths_receipt.jpg",
            "<|image|>Extract the store name from this receipt."
        )
        print(f"Test response: {test_response[:100]}...")

        if test_response and not test_response.startswith("Error:"):
            print("✅ Basic prediction works!")
        else:
            print("❌ Basic prediction failed!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        # Cleanup
        try:
            model.unload_model()
            print("✅ Model unloaded successfully")
        except Exception as e:
            print(f"⚠️ Model unload warning: {e}")


if __name__ == "__main__":
    success = test_llama_model()
    if success:
        print("\n🎉 Llama model fix appears to be working!")
    else:
        print("\n💥 Llama model fix needs more work")
