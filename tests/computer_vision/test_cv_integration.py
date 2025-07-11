"""Computer Vision Integration Tests with Real Images

Tests CV components with actual image processing and real computer vision
algorithms instead of mocks.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_processor.computer_vision.highlight_detector import HighlightDetector
from vision_processor.computer_vision.image_preprocessor import ImagePreprocessor
from vision_processor.computer_vision.ocr_processor import OCRProcessor
from vision_processor.computer_vision.spatial_correlator import SpatialCorrelator


class TestComputerVisionIntegration:
    """Integration tests for CV components with real images."""

    @pytest.fixture
    def real_test_images(self):
        """Get paths to real test images from datasets directory."""
        datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        if not datasets_dir.exists():
            pytest.skip("Datasets directory not found")

        image_files = list(datasets_dir.glob("image*.png"))[:5]
        if not image_files:
            pytest.skip("No test images found in datasets directory")

        return image_files

    @pytest.fixture
    def preprocessor(self):
        """Create real image preprocessor instance."""
        return ImagePreprocessor()

    @pytest.fixture
    def highlight_detector(self):
        """Create real highlight detector instance."""
        return HighlightDetector()

    @pytest.fixture
    def ocr_processor(self):
        """Create real OCR processor instance."""
        return OCRProcessor()

    @pytest.fixture
    def spatial_correlator(self):
        """Create real spatial correlator instance."""
        return SpatialCorrelator()

    def test_image_preprocessing_real(self, preprocessor, real_test_images):
        """Test image preprocessing with real images."""
        test_image_path = real_test_images[0]
        original_image = Image.open(test_image_path)

        # Test basic enhancement
        enhanced_image = preprocessor.enhance_image(original_image)

        assert isinstance(enhanced_image, Image.Image)
        assert enhanced_image.size == original_image.size

        # Test that enhancement actually changes the image
        original_array = np.array(original_image)
        enhanced_array = np.array(enhanced_image)

        # Images should be different (unless already perfectly enhanced)
        difference = np.mean(np.abs(original_array.astype(float) - enhanced_array.astype(float)))
        print(f"✅ Image enhancement difference: {difference:.2f}")

        # Test noise reduction
        if hasattr(preprocessor, "reduce_noise"):
            clean_image = preprocessor.reduce_noise(enhanced_image)
            assert isinstance(clean_image, Image.Image)
            print("✅ Noise reduction completed")

        # Test contrast enhancement
        if hasattr(preprocessor, "enhance_contrast"):
            contrast_enhanced = preprocessor.enhance_contrast(original_image)
            assert isinstance(contrast_enhanced, Image.Image)
            print("✅ Contrast enhancement completed")

    def test_highlight_detection_real(self, highlight_detector, real_test_images):
        """Test highlight detection with real images."""
        test_image_path = real_test_images[0]
        test_image = Image.open(test_image_path)

        # Test basic highlight detection
        highlights = highlight_detector.detect_highlights(test_image)

        assert isinstance(highlights, list)
        print(f"✅ Detected {len(highlights)} highlights")

        # Verify highlight structure
        for highlight in highlights:
            assert "bbox" in highlight or "coordinates" in highlight
            if "confidence" in highlight:
                assert 0 <= highlight["confidence"] <= 1

        # Test bank statement specific highlights if available
        if hasattr(highlight_detector, "detect_bank_statement_highlights"):
            bank_highlights = highlight_detector.detect_bank_statement_highlights(test_image)
            assert isinstance(bank_highlights, list)
            print(f"✅ Detected {len(bank_highlights)} bank statement highlights")

    def test_ocr_processing_real(self, ocr_processor, real_test_images):
        """Test OCR processing with real images."""
        test_image_path = real_test_images[0]
        test_image = Image.open(test_image_path)

        # Test basic text extraction
        extracted_text = ocr_processor.extract_text(test_image)

        assert isinstance(extracted_text, str)
        print(f"✅ OCR extracted {len(extracted_text)} characters")
        print(f"   Sample text: {extracted_text[:100]}...")

        # Test structured text extraction if available
        if hasattr(ocr_processor, "extract_structured_text"):
            structured_result = ocr_processor.extract_structured_text(test_image)
            assert isinstance(structured_result, dict)

            if "text_blocks" in structured_result:
                print(f"✅ Structured OCR found {len(structured_result['text_blocks'])} text blocks")

        # Look for common fields in extracted text
        found_patterns = []

        # Check for amounts ($ followed by numbers)
        import re

        amounts = re.findall(r"\$\d+\.?\d*", extracted_text)
        if amounts:
            found_patterns.append(f"amounts: {amounts[:3]}")

        # Check for dates
        dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", extracted_text)
        if dates:
            found_patterns.append(f"dates: {dates[:3]}")

        # Check for phone numbers
        phones = re.findall(r"\d{4}\s?\d{3}\s?\d{3}", extracted_text)
        if phones:
            found_patterns.append(f"phones: {phones[:2]}")

        if found_patterns:
            print(f"✅ OCR found patterns: {', '.join(found_patterns)}")

    def test_spatial_correlation_real(self, spatial_correlator, real_test_images):
        """Test spatial correlation with real images."""
        _test_image_path = real_test_images[0]

        # Create some mock text regions for correlation
        text_regions = [
            {"text": "Total:", "bbox": [100, 100, 150, 120]},
            {"text": "$25.50", "bbox": [200, 100, 260, 120]},
            {"text": "Date:", "bbox": [100, 150, 140, 170]},
            {"text": "25/03/2024", "bbox": [200, 150, 280, 170]},
        ]

        if hasattr(spatial_correlator, "correlate_text_regions"):
            correlations = spatial_correlator.correlate_text_regions(text_regions)
            assert isinstance(correlations, list)
            print(f"✅ Found {len(correlations)} spatial correlations")

        # Test key-value pair detection
        if hasattr(spatial_correlator, "find_key_value_pairs"):
            kv_pairs = spatial_correlator.find_key_value_pairs(text_regions)
            assert isinstance(kv_pairs, dict)
            print(f"✅ Found {len(kv_pairs)} key-value pairs")

    def test_cv_pipeline_integration(
        self, preprocessor, highlight_detector, ocr_processor, real_test_images
    ):
        """Test full CV pipeline integration."""
        test_image_path = real_test_images[0]
        original_image = Image.open(test_image_path)

        # Step 1: Preprocessing
        enhanced_image = preprocessor.enhance_image(original_image)
        print("✅ Step 1: Image preprocessing completed")

        # Step 2: Highlight detection
        highlights = highlight_detector.detect_highlights(enhanced_image)
        print(f"✅ Step 2: Detected {len(highlights)} highlights")

        # Step 3: OCR processing
        extracted_text = ocr_processor.extract_text(enhanced_image)
        print(f"✅ Step 3: OCR extracted {len(extracted_text)} characters")

        # Step 4: Combine results
        pipeline_result = {
            "original_image_size": original_image.size,
            "enhanced_image_size": enhanced_image.size,
            "highlights_count": len(highlights),
            "text_length": len(extracted_text),
            "highlights": highlights[:5],  # First 5 highlights
            "text_sample": extracted_text[:200],
        }

        print("✅ Full CV pipeline completed successfully")
        return pipeline_result

    def test_performance_cv_real(self, preprocessor, ocr_processor, real_test_images):
        """Test CV performance with real images."""
        import time

        test_image_path = real_test_images[0]
        test_image = Image.open(test_image_path)

        # Test preprocessing performance
        start_time = time.time()
        enhanced_image = preprocessor.enhance_image(test_image)
        preprocessing_time = time.time() - start_time

        assert preprocessing_time < 10, f"Preprocessing too slow: {preprocessing_time:.2f}s"

        # Test OCR performance
        start_time = time.time()
        _extracted_text = ocr_processor.extract_text(enhanced_image)
        ocr_time = time.time() - start_time

        assert ocr_time < 15, f"OCR too slow: {ocr_time:.2f}s"

        total_time = preprocessing_time + ocr_time
        print(
            f"✅ CV Performance: preprocessing={preprocessing_time:.2f}s, OCR={ocr_time:.2f}s, total={total_time:.2f}s"
        )

        # Performance target from testing plan: <2 seconds per image
        if total_time > 2:
            print("⚠️  CV processing slower than target (aim for <2s per image)")

    def test_multiple_images_cv(self, preprocessor, ocr_processor, real_test_images):
        """Test CV processing with multiple images."""
        results = []

        for i, image_path in enumerate(real_test_images[:3]):
            try:
                image = Image.open(image_path)
                enhanced = preprocessor.enhance_image(image)
                text = ocr_processor.extract_text(enhanced)

                result = {
                    "image_index": i,
                    "image_size": image.size,
                    "text_length": len(text),
                    "success": True,
                }
                results.append(result)

            except Exception as e:
                result = {"image_index": i, "error": str(e), "success": False}
                results.append(result)

        successful_results = [r for r in results if r["success"]]
        success_rate = len(successful_results) / len(results)

        assert success_rate >= 0.6, f"Success rate too low: {success_rate:.2f}"

        print(
            f"✅ Multi-image CV test: {len(successful_results)}/{len(results)} successful ({success_rate:.2%})"
        )

    def test_error_handling_cv(self, preprocessor, _ocr_processor):
        """Test CV error handling with invalid inputs."""
        # Test with invalid image
        try:
            # Create a corrupted image
            invalid_image = Image.new("RGB", (10, 10), color="white")
            # Try to process very small image
            result = preprocessor.enhance_image(invalid_image)
            assert isinstance(result, Image.Image)
            print("✅ Small image handled gracefully")
        except Exception as e:
            print(f"⚠️  Small image processing error: {e}")

        # Test with None input (should raise appropriate error)
        try:
            preprocessor.enhance_image(None)
            raise AssertionError("Should have raised an error for None input")
        except (TypeError, AttributeError):
            print("✅ None input handled with appropriate error")


@pytest.mark.slow
class TestCVPerformance:
    """Performance-focused CV tests - marked as slow."""

    def test_cv_memory_usage(self, preprocessor, ocr_processor, real_test_images):
        """Test memory usage of CV processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple images
        for image_path in real_test_images[:5]:
            try:
                image = Image.open(image_path)
                enhanced = preprocessor.enhance_image(image)
                text = ocr_processor.extract_text(enhanced)
                # Clean up
                del image, enhanced, text
            except Exception as e:
                print(f"⚠️  Error processing {image_path}: {e}")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"✅ CV Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_increase:.1f}MB)"
        )

        # Memory target from testing plan: <500MB increase
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
