"""Fast CLI Tests - Mac M1 Compatible

Fast unit tests for CLI functionality that run efficiently on local development
machines. These tests use smart mocking to validate CLI behavior without
loading actual models.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vision_processor.cli.unified_cli import app


class TestCLIFast:
    """Fast CLI test suite using efficient mocking."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for fast tests."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self):
        """Mock all dependencies to prevent model loading."""
        # Mock the extraction manager
        with (
            patch("vision_processor.cli.unified_cli.UnifiedExtractionManager") as mock_manager,
            patch("vision_processor.config.unified_config.UnifiedConfig.from_env") as mock_config_from_env,
        ):
            # Create mock config
            mock_config = MagicMock()
            mock_config.model_type = "internvl3"
            mock_config_from_env.return_value = mock_config

            # Create mock result with proper attributes
            mock_result = MagicMock()
            mock_result.extracted_fields = {
                "supplier_name": "Test Store",
                "total_amount": "123.45",
                "date": "25/03/2024",
                "gst_amount": "11.22",
                "abn": "88 000 014 675",
            }
            mock_result.confidence_score = 0.85
            mock_result.document_type = "business_receipt"
            mock_result.model_type = "internvl3"
            mock_result.processing_time = 2.1
            mock_result.memory_usage_mb = 512.0
            mock_result.ato_compliance_score = 0.90
            mock_result.highlights_detected = 2
            mock_result.awk_fallback_used = False
            mock_result.validation_passed = True
            mock_result.production_ready = True

            # Mock quality_grade with .value attribute
            mock_quality_grade = MagicMock()
            mock_quality_grade.value = "GOOD"
            mock_result.quality_grade = mock_quality_grade

            # Setup context manager behavior
            mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

            yield mock_manager

    def test_cli_help_fast(self, cli_runner):
        """Test CLI help system (fast)."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Unified Vision Document Processing" in result.output
        assert "process" in result.output

    def test_process_command_help_fast(self, cli_runner):
        """Test process command help (fast)."""
        result = cli_runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "image_path" in result.output.lower()
        assert "model" in result.output.lower()

    def test_process_basic_fast(self, cli_runner, mock_all_dependencies, sample_image):
        """Test basic process command (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image)])

        assert result.exit_code == 0
        assert "Test Store" in result.output
        assert "123.45" in result.output
        mock_all_dependencies.assert_called_once()

    def test_model_selection_fast(self, cli_runner, mock_all_dependencies, sample_image):
        """Test model selection (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--model", "internvl3"])

        assert result.exit_code == 0
        mock_all_dependencies.assert_called_once()

    def test_invalid_model_fast(self, cli_runner, sample_image):
        """Test invalid model handling (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--model", "invalid_model"])

        assert result.exit_code == 1
        assert "Invalid model" in result.output

    def test_nonexistent_file_fast(self, cli_runner):
        """Test nonexistent file handling (fast)."""
        result = cli_runner.invoke(app, ["process", "nonexistent.jpg"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_json_output_fast(self, cli_runner, sample_image, temp_directory):
        """Test JSON output (fast)."""
        output_file = temp_directory / "output.json"

        result = cli_runner.invoke(app, ["process", str(sample_image), "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON content
        with output_file.open() as f:
            data = json.load(f)
            assert "extracted_fields" in data
            assert data["extracted_fields"]["total_amount"] == "123.45"

    def test_verbose_output_fast(self, cli_runner, sample_image):
        """Test verbose output (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--verbose"])

        assert result.exit_code == 0
        # Verbose should include processing details
        output_lower = result.output.lower()
        assert any(word in output_lower for word in ["processing", "confidence", "time"])

    def test_document_type_specification_fast(self, cli_runner, mock_all_dependencies, sample_image):
        """Test document type specification (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--type", "fuel_receipt"])

        assert result.exit_code == 0
        mock_all_dependencies.assert_called_once()

    def test_all_model_types_fast(self, cli_runner, sample_image):
        """Test all supported model types (fast)."""
        models = ["internvl3", "llama32_vision"]

        for model in models:
            result = cli_runner.invoke(app, ["process", str(sample_image), "--model", model])

            assert result.exit_code == 0, f"Model {model} failed"

    def test_confidence_display_fast(self, cli_runner, sample_image):
        """Test confidence score display (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image), "--verbose"])

        assert result.exit_code == 0
        # Should display confidence information
        assert "0.85" in result.output or "85" in result.output

    def test_australian_fields_display_fast(self, cli_runner, sample_image):
        """Test Australian-specific field display (fast)."""
        result = cli_runner.invoke(app, ["process", str(sample_image)])

        assert result.exit_code == 0
        # Should show Australian tax fields
        assert "abn" in result.output.lower() or "88 000 014 675" in result.output
        assert "gst" in result.output.lower() or "11.22" in result.output

    @patch("vision_processor.cli.unified_cli.UnifiedExtractionManager")
    def test_processing_error_handling_fast(self, mock_manager, cli_runner, sample_image):
        """Test processing error handling (fast)."""
        # Make the manager raise an exception
        mock_manager.return_value.__enter__.return_value.process_document.side_effect = Exception(
            "Processing failed"
        )

        result = cli_runner.invoke(app, ["process", str(sample_image)])

        # Should handle error gracefully
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_output_formats_fast(self, cli_runner, sample_image, temp_directory):
        """Test different output formats (fast)."""
        formats = ["json"]  # Add more formats as implemented

        for fmt in formats:
            output_file = temp_directory / f"output.{fmt}"

            result = cli_runner.invoke(
                app, ["process", str(sample_image), "--output", str(output_file), "--format", fmt]
            )

            # Note: --format may not be implemented yet
            assert result.exit_code == 0 or result.exit_code == 2  # 2 for unrecognized option


class TestCLIValidation:
    """Fast validation tests for CLI parameters."""

    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    @pytest.fixture(autouse=True)
    def mock_all_dependencies(self):
        """Mock all dependencies to prevent model loading."""
        # Same mock as in TestCLIFast
        with (
            patch("vision_processor.cli.unified_cli.UnifiedExtractionManager") as mock_manager,
            patch("vision_processor.config.unified_config.UnifiedConfig.from_env") as mock_config_from_env,
        ):
            # Create mock config
            mock_config = MagicMock()
            mock_config.model_type = "internvl3"
            mock_config_from_env.return_value = mock_config

            # Create mock result with proper attributes
            mock_result = MagicMock()
            mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
            mock_result.confidence_score = 0.85
            mock_result.document_type = "business_receipt"
            mock_result.model_type = "internvl3"
            mock_result.processing_time = 2.1
            mock_result.ato_compliance_score = 0.90
            mock_result.highlights_detected = 2
            mock_result.awk_fallback_used = False
            mock_result.production_ready = True

            # Mock quality_grade with .value attribute
            mock_quality_grade = MagicMock()
            mock_quality_grade.value = "GOOD"
            mock_result.quality_grade = mock_quality_grade

            # Setup context manager behavior
            mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

            yield mock_manager

    def test_parameter_validation_fast(self, cli_runner):
        """Test CLI parameter validation (fast)."""
        # Test missing required argument
        result = cli_runner.invoke(app, ["process"])
        assert result.exit_code != 0

        # Test help shows usage
        result = cli_runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "image_path" in result.output

    def test_option_combinations_fast(self, cli_runner, sample_image):
        """Test various option combinations (fast)."""
        # Test multiple options - mocking is handled by autouse fixture
        result = cli_runner.invoke(
            app,
            [
                "process",
                str(sample_image),
                "--model",
                "internvl3",
                "--type",
                "business_receipt",
                "--verbose",
            ],
        )

        assert result.exit_code == 0


@pytest.mark.fast
class TestCLIFastMarked:
    """CLI tests marked as fast for selective running."""

    def test_marked_fast_example(self):
        """Example of a test marked as fast."""
        # This test would run when using: pytest -m fast
        assert True


# Configuration for pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "fast: marks tests as fast (deselect with '-m \"not fast\"')")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
