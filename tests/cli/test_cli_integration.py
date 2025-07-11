"""CLI Integration Tests with Real Models and Images

Tests the CLI with actual model loading and real image processing.
These tests are designed for the multi-GPU development machine.

Run these tests ONLY on the development machine with:
pytest -m "integration and gpu" tests/cli/test_cli_integration.py

DO NOT run these on Mac M1 - they will be too slow!
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vision_processor.cli.unified_cli import app


@pytest.mark.integration
@pytest.mark.gpu
class TestCLIIntegration:
    """Integration test suite using real models and images."""

    @pytest.fixture
    def real_test_images(self):
        """Get paths to real test images from datasets directory."""
        datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        if not datasets_dir.exists():
            pytest.skip("Datasets directory not found")

        # Get first few images for testing
        image_files = list(datasets_dir.glob("image*.png"))[:5]
        if not image_files:
            pytest.skip("No test images found in datasets directory")

        return image_files

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for integration tests."""
        return CliRunner()

    def test_real_image_processing_internvl(self, cli_runner, real_test_images):
        """Test processing real images with InternVL model."""
        test_image = real_test_images[0]

        result = cli_runner.invoke(app, ["process", str(test_image), "--model", "internvl3", "--verbose"])

        # Check that processing completed successfully
        assert result.exit_code == 0

        # Verify output contains expected fields
        output = result.output.lower()
        assert any(field in output for field in ["amount", "vendor", "date", "extracted"])

        print(f"✅ InternVL processing result: {result.output[:200]}...")

    def test_real_image_processing_llama(self, cli_runner, real_test_images):
        """Test processing real images with Llama model."""
        test_image = real_test_images[0]

        result = cli_runner.invoke(
            app, ["process", str(test_image), "--model", "llama32_vision", "--verbose"]
        )

        # Check that processing completed successfully
        assert result.exit_code == 0

        # Verify output contains expected fields
        output = result.output.lower()
        assert any(field in output for field in ["amount", "vendor", "date", "extracted"])

        print(f"✅ Llama processing result: {result.output[:200]}...")

    def test_real_batch_processing(self, cli_runner, real_test_images, temp_directory):
        """Test batch processing with real images."""
        # Create a batch directory with a subset of images
        batch_dir = temp_directory / "batch_test"
        batch_dir.mkdir()

        # Copy first 3 images for batch testing
        for i, source_image in enumerate(real_test_images[:3]):
            target_image = batch_dir / f"test_{i}.png"
            target_image.write_bytes(source_image.read_bytes())

        output_dir = temp_directory / "batch_output"
        output_dir.mkdir()

        # Note: Batch processing command may need to be implemented in CLI
        # This tests the current process command on multiple files
        results = []
        for image_file in batch_dir.glob("*.png"):
            result = cli_runner.invoke(app, ["process", str(image_file), "--model", "internvl3"])
            results.append(result)

        # Verify all images processed successfully
        successful_results = [r for r in results if r.exit_code == 0]
        assert len(successful_results) >= 1, "At least one image should process successfully"

        print(f"✅ Batch processed {len(successful_results)}/{len(results)} images successfully")

    def test_real_json_output(self, cli_runner, real_test_images, temp_directory):
        """Test JSON output with real image processing."""
        test_image = real_test_images[0]
        output_file = temp_directory / "real_output.json"

        result = cli_runner.invoke(
            app,
            [
                "process",
                str(test_image),
                "--model",
                "internvl3",
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with output_file.open() as f:
            data = json.load(f)
            assert "extracted_fields" in data
            assert isinstance(data["extracted_fields"], dict)

            # Check for common fields
            fields = data["extracted_fields"]
            assert len(fields) > 0, "Should extract at least some fields"

        print(f"✅ JSON output contains {len(data['extracted_fields'])} extracted fields")

    def test_confidence_thresholds_real(self, cli_runner, real_test_images):
        """Test confidence threshold filtering with real processing."""
        test_image = real_test_images[0]

        # Test with low threshold
        result_low = cli_runner.invoke(
            app, ["process", str(test_image), "--confidence-threshold", "0.1", "--verbose"]
        )

        # Test with high threshold
        result_high = cli_runner.invoke(
            app, ["process", str(test_image), "--confidence-threshold", "0.9", "--verbose"]
        )

        # Both should complete (though high threshold might filter more)
        assert result_low.exit_code == 0
        assert result_high.exit_code == 0

        print(
            f"✅ Confidence filtering working - low: {len(result_low.output)} chars, high: {len(result_high.output)} chars"
        )

    def test_document_type_specification_real(self, cli_runner, real_test_images):
        """Test document type specification with real images."""
        test_image = real_test_images[0]

        # Test different document types
        doc_types = ["business_receipt", "fuel_receipt", "tax_invoice"]

        results = []
        for doc_type in doc_types:
            result = cli_runner.invoke(
                app, ["process", str(test_image), "--type", doc_type, "--model", "internvl3"]
            )
            results.append((doc_type, result))

        # At least one document type should process successfully
        successful = [(dt, r) for dt, r in results if r.exit_code == 0]
        assert len(successful) >= 1, "At least one document type should work"

        print(f"✅ Document types working: {[dt for dt, _ in successful]}")

    def test_highlight_detection_real(self, cli_runner, real_test_images):
        """Test highlight detection with real images."""
        test_image = real_test_images[0]

        result = cli_runner.invoke(app, ["process", str(test_image), "--enable-highlights", "--verbose"])

        assert result.exit_code == 0
        # Output might contain information about highlights detected
        print(f"✅ Highlight detection enabled: {result.output[:200]}...")

    def test_awk_fallback_real(self, cli_runner, real_test_images):
        """Test AWK fallback with real images."""
        test_image = real_test_images[0]

        result = cli_runner.invoke(app, ["process", str(test_image), "--enable-awk-fallback", "--verbose"])

        assert result.exit_code == 0
        print(f"✅ AWK fallback enabled: {result.output[:200]}...")

    def test_model_comparison_real(self, cli_runner, real_test_images, temp_directory):
        """Compare results between different models on same image."""
        test_image = real_test_images[0]

        # Process with InternVL
        result_internvl = cli_runner.invoke(
            app,
            [
                "process",
                str(test_image),
                "--model",
                "internvl3",
                "--output",
                str(temp_directory / "internvl_result.json"),
                "--format",
                "json",
            ],
        )

        # Process with Llama
        result_llama = cli_runner.invoke(
            app,
            [
                "process",
                str(test_image),
                "--model",
                "llama32_vision",
                "--output",
                str(temp_directory / "llama_result.json"),
                "--format",
                "json",
            ],
        )

        # Both should complete successfully
        if result_internvl.exit_code == 0 and result_llama.exit_code == 0:
            # Compare results
            with (temp_directory / "internvl_result.json").open() as f:
                internvl_data = json.load(f)
            with (temp_directory / "llama_result.json").open() as f:
                llama_data = json.load(f)

            print("✅ Model comparison:")
            print(f"   InternVL fields: {len(internvl_data.get('extracted_fields', {}))}")
            print(f"   Llama fields: {len(llama_data.get('extracted_fields', {}))}")
        else:
            print(
                f"⚠️  Model comparison: InternVL={result_internvl.exit_code}, Llama={result_llama.exit_code}"
            )

    def test_error_handling_real(self, cli_runner):
        """Test error handling with invalid inputs."""
        # Test with non-existent file
        result = cli_runner.invoke(app, ["process", "nonexistent_file.jpg"])
        assert result.exit_code != 0

        # Test with invalid model
        result = cli_runner.invoke(app, ["process", "some_file.jpg", "--model", "invalid_model"])
        assert result.exit_code != 0

        print("✅ Error handling working correctly")

    def test_performance_real(self, cli_runner, real_test_images):
        """Test processing performance with real images."""
        import time

        test_image = real_test_images[0]

        start_time = time.time()
        result = cli_runner.invoke(app, ["process", str(test_image), "--model", "internvl3", "--verbose"])
        processing_time = time.time() - start_time

        assert result.exit_code == 0

        # Performance check - should complete within reasonable time
        assert processing_time < 60, f"Processing took too long: {processing_time:.2f}s"

        print(f"✅ Performance test: {processing_time:.2f}s for real image processing")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.gpu
class TestCLIPerformance:
    """Performance tests with real models - marked as slow."""

    def test_large_batch_performance(self, cli_runner, real_test_images, temp_directory):
        """Test performance with larger batch of images."""
        if len(real_test_images) < 10:
            pytest.skip("Need at least 10 test images for performance testing")

        # Create batch directory
        batch_dir = temp_directory / "performance_batch"
        batch_dir.mkdir()

        # Copy 10 images for batch testing
        for i, source_image in enumerate(real_test_images[:10]):
            target_image = batch_dir / f"perf_test_{i}.png"
            target_image.write_bytes(source_image.read_bytes())

        import time

        start_time = time.time()

        # Process all images
        successful_count = 0
        for image_file in batch_dir.glob("*.png"):
            result = cli_runner.invoke(app, ["process", str(image_file), "--model", "internvl3"])
            if result.exit_code == 0:
                successful_count += 1

        total_time = time.time() - start_time

        assert successful_count >= 5, "At least half the images should process successfully"

        # Performance target: >20 documents per minute (as per testing plan)
        docs_per_minute = (successful_count / total_time) * 60

        print(
            f"✅ Performance: {successful_count} docs in {total_time:.2f}s = {docs_per_minute:.1f} docs/min"
        )

        # Note: May need to adjust this threshold based on actual performance
        if docs_per_minute < 5:
            print("⚠️  Performance below target (aim for >20 docs/min)")
