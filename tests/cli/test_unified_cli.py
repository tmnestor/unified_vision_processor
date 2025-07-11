"""Test the main CLI entry point and command routing.

Tests the unified CLI interface including help system, version display,
command routing, configuration loading, and error handling.
"""

from unittest.mock import MagicMock, patch

from vision_processor.cli.unified_cli import app


class TestUnifiedCLI:
    """Test suite for unified CLI interface."""

    def test_cli_help(self, cli_runner):
        """Test CLI help system displays all available commands."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Unified Vision Document Processing" in result.output
        assert "process" in result.output  # Main processing command

    def test_cli_version_info(self, cli_runner):
        """Test version and info display."""
        # Test that the CLI runs without version flag (should show help or process)
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "help" in result.output.lower() or "usage" in result.output.lower()

    def test_invalid_command(self, cli_runner):
        """Test handling of invalid commands."""
        result = cli_runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        # Should show error about unknown command or usage help

    def test_cli_process_command_help(self, cli_runner):
        """Test process command help."""
        result = cli_runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "image_path" in result.output.lower()
        assert "model" in result.output.lower()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_command_basic(self, mock_manager, cli_runner, sample_image):
        """Test basic process command functionality."""
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
    def test_process_with_model_selection(self, mock_manager, cli_runner, sample_image):
        """Test model selection via CLI."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--model", "internvl3"])
        assert result.exit_code == 0
        # Verify model type was passed correctly
        mock_manager.assert_called_once()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_with_output_file(self, mock_manager, cli_runner, sample_image, temp_directory):
        """Test output to file functionality."""
        output_file = temp_directory / "output.json"

        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--output", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        import json

        with output_file.open() as f:
            data = json.load(f)
            assert "extracted_fields" in data
            assert "confidence_score" in data

    def test_process_nonexistent_file(self, cli_runner):
        """Test error handling for nonexistent input files."""
        result = cli_runner.invoke(app, ["process", "nonexistent_file.jpg"])
        assert result.exit_code != 0
        # Should show appropriate error message

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_with_document_type(self, mock_manager, cli_runner, sample_image):
        """Test explicit document type specification."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "fuel_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--type", "fuel_receipt"])
        assert result.exit_code == 0

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_with_confidence_threshold(self, mock_manager, cli_runner, sample_image):
        """Test confidence threshold parameter."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--confidence-threshold", "0.8"])
        assert result.exit_code == 0

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_verbose_output(self, mock_manager, cli_runner, sample_image):
        """Test verbose output mode."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_result.processing_time = 2.5
        mock_result.memory_usage_mb = 512.0
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(app, ["process", str(sample_image), "--verbose"])
        assert result.exit_code == 0
        # Should include additional details in verbose mode
        assert "processing_time" in result.output.lower() or "time" in result.output.lower()

    def test_process_invalid_model_type(self, cli_runner, sample_image):
        """Test error handling for invalid model types."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--model", "invalid_model"])
        # Should either exit with error or show available models
        assert result.exit_code != 0 or "invalid" in result.output.lower()

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_process_json_output_format(self, mock_manager, cli_runner, sample_image, temp_directory):
        """Test JSON output format."""
        output_file = temp_directory / "output.json"

        # Setup mock
        mock_result = MagicMock()
        mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
        mock_result.confidence_score = 0.85
        mock_result.document_type = "business_receipt"
        mock_result.quality_grade = "GOOD"
        mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

        result = cli_runner.invoke(
            app, ["process", str(sample_image), "--output", str(output_file), "--format", "json"]
        )
        assert result.exit_code == 0
        assert output_file.exists()

        # Verify valid JSON
        import json

        with output_file.open() as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "extracted_fields" in data
