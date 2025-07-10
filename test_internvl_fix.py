#!/usr/bin/env python3
"""
Test script for InternVL device mismatch fixes.
Run this on your H200 development machine to verify the fixes work.
"""

import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging to see device info
logging.basicConfig(level=logging.INFO)


def test_internvl_device_handling():
    """Test InternVL model loading and inference with device handling fixes."""

    try:
        from vision_processor.config.model_factory import ModelFactory
        from vision_processor.config.unified_config import UnifiedConfig

        print("✅ Successfully imported vision_processor modules")

        # Create config for InternVL3
        config = UnifiedConfig.from_env()
        print(f"✅ Config loaded: {config.model_type}")

        # Create model
        model_config = config.get_model_config()
        model = ModelFactory.create_model(**model_config)
        print("✅ Model created successfully")

        # Load model
        model.load_model()
        print("✅ Model loaded successfully")

        # Test processing with a simple image
        test_image_path = "datasets/image25.png"
        if not Path(test_image_path).exists():
            print(f"⚠️ Test image {test_image_path} not found, skipping inference test")
            return

        # Test inference
        print("🔍 Testing inference with device fixes...")
        prompt = "What type of document is this?"

        response = model.process_image(test_image_path, prompt)
        print("✅ Inference successful!")
        print(f"   Response: {response.raw_text[:100]}...")
        print(f"   Processing time: {response.processing_time:.2f}s")
        print(f"   Device used: {response.device_used}")

        # Unload model
        model.unload_model()
        print("✅ Model unloaded successfully")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("🔬 Testing InternVL Device Handling Fixes")
    print("=" * 50)

    success = test_internvl_device_handling()

    if success:
        print("\n🎉 All tests passed! InternVL device handling is working correctly.")
        print("You can now run: uvp-process datasets/image25.png --model internvl3")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
