#!/usr/bin/env python3
"""
Test script to verify prompt manager fix.
"""

import logging
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_prompt_fix():
    """Test that prompt manager parameter names are fixed."""

    try:
        from vision_processor.classification.document_types import DocumentType
        from vision_processor.extraction.pipeline_components import PromptManager

        print("✅ Successfully imported PromptManager")

        # Create a prompt manager instance
        config = {}  # Empty config for testing
        prompt_manager = PromptManager(config)

        # Test calling get_prompt with the keyword argument
        prompt = prompt_manager.get_prompt(
            document_type=DocumentType.RECEIPT, has_highlights=False
        )

        print("✅ get_prompt() called successfully")
        print(f"   Prompt: {prompt}")

        # Test with highlights
        prompt_with_highlights = prompt_manager.get_prompt(
            document_type=DocumentType.RECEIPT, has_highlights=True
        )

        print("✅ get_prompt() with highlights called successfully")
        print(f"   Prompt: {prompt_with_highlights}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("🔬 Testing Prompt Manager Fix")
    print("=" * 40)

    success = test_prompt_fix()

    if success:
        print("\n🎉 Prompt manager fix successful!")
        print("The has_highlights parameter issue is resolved.")
    else:
        print("\n❌ Prompt manager fix failed.")
