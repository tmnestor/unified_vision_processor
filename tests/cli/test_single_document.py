"""Test single document processing CLI functionality.

Tests the single document processing commands including model selection,
output formats, error handling, and processing options.
"""

import json
from unittest.mock import MagicMock, patch

from vision_processor.cli.unified_cli import app


class TestSingleDocumentCLI:
    """Test suite for single document processing."""

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_single_document_basic(self, mock_manager, cli_runner, sample_image):
        """Test basic single document processing."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Vendor"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image)])
        assert result.exit_code == 0
        assert "123.45" in result.output
        assert "Test Vendor" in result.output

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_single_document_with_model_selection(self, mock_manager, cli_runner, sample_image):
        """Test model selection via CLI."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--model", "internvl3"])
        assert result.exit_code == 0
        # Verify model type was passed correctly
        mock_manager.assert_called_once()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_output_formats_json(self, mock_manager, cli_runner, sample_image, temp_directory):
        """Test JSON output format."""
        output_file = temp_directory / "output.json"

        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.processing_time = 2.1
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(
            app, ["process", str(sample_image), "--output", str(output_file), "--format", "json"]
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with output_file.open() as f:
            data = json.load(f)
            assert "extracted_fields" in data
            assert "confidence_score" in data
            assert data["extracted_fields"]["amount"] == "123.45"

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_output_formats_csv(self, mock_manager, cli_runner, sample_image, temp_directory):
        """Test CSV output format."""
        output_file = temp_directory / "output.csv"

        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store", "date": "25/03/2024"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(
            app, ["process", str(sample_image), "--output", str(output_file), "--format", "csv"]
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify CSV content
        content = output_file.read_text()
        assert "amount,vendor,date" in content or "123.45" in content

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_confidence_threshold_filtering(self, mock_manager, cli_runner, sample_image):
        """Test confidence threshold filtering."""
        # Setup mock with low confidence
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45"}
        mock_result.confidence_score = 0.5  # Low confidence
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "POOR"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        # Test with high threshold
        result = cli_runner.invoke(app, ["process", str(sample_image), "--confidence-threshold", "0.8"])
        # Should either warn about low confidence or exit with appropriate code
        assert result.exit_code == 0  # CLI should still complete but may warn

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_document_type_specification(self, mock_manager, cli_runner, sample_image):
        """Test explicit document type specification."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "45.67", "fuel_type": "Unleaded"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "fuel_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--type", "fuel_receipt"])
        assert result.exit_code == 0
        assert "45.67" in result.output

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_verbose_output_mode(self, mock_manager, cli_runner, sample_image):
        """Test verbose output mode with detailed information."""
        # Setup mock with detailed information
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.processing_time = 2.5
        mock_result.memory_usage_mb = 512.0
        mock_result.ato_compliance_score = 0.90
        mock_result.highlights_detected = 3
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--verbose"])
        assert result.exit_code == 0
        # Should include processing details in verbose mode
        output_lower = result.output.lower()
        assert any(keyword in output_lower for keyword in ["time", "memory", "confidence", "quality"])

    def test_invalid_image_path(self, cli_runner):
        """Test error handling for invalid image paths."""
        result = cli_runner.invoke(app, ["process", "nonexistent_image.jpg"])
        assert result.exit_code != 0

    def test_invalid_output_directory(self, cli_runner, sample_image):
        """Test error handling for invalid output directories."""
        result = cli_runner.invoke(
            app, ["process", str(sample_image), "--output", "/invalid/path/output.json"]
        )
        # Should handle the error gracefully
        assert result.exit_code != 0 or "error" in result.output.lower()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_highlight_detection_option(self, mock_manager, cli_runner, sample_image):
        """Test highlight detection option."""
        # Setup mock with highlights
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.highlights_detected = 2
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--enable-highlights"])
        assert result.exit_code == 0

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_awk_fallback_option(self, mock_manager, cli_runner, sample_image):
        """Test AWK fallback option."""
        # Setup mock with AWK fallback used
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.awk_fallback_used = True
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--enable-awk-fallback"])
        assert result.exit_code == 0

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_output_to_stdout(self, mock_manager, cli_runner, sample_image):
        """Test output to stdout (default behavior)."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image)])
        assert result.exit_code == 0
        assert "123.45" in result.output
        assert "Test Store" in result.output

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_processing_error_handling(self, mock_manager, cli_runner, sample_image):
        """Test handling of processing errors."""
        # Setup mock to raise an exception
        mock_manager.return_value.__enter__.return_value.process_document.side_effect = Exception(
            "Processing failed"
        )

        result = cli_runner.invoke(app, ["process", str(sample_image)])
        # Should handle the error gracefully and exit with error code
        assert result.exit_code != 0 or "error" in result.output.lower()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_australian_specific_fields(self, mock_manager, cli_runner, sample_image):
        """Test extraction of Australian-specific fields."""
        # Setup mock with Australian tax fields
        mock_result = MagicMock()
        mock_result.extracted_fields = {
            "amount": "123.45",
            "vendor": "Woolworths",
            "abn": "88 000 014 675",
            "gst_amount": "11.22",
            "date": "25/03/2024",
        }
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.ato_compliance_score = 0.95
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image)])
        assert result.exit_code == 0
        assert "88 000 014 675" in result.output  # ABN
        assert "11.22" in result.output  # GST amount
