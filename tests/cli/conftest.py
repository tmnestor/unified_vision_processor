"""CLI Test Configuration and Fixtures

Provides CLI-specific test fixtures, mock data, and utilities for testing
the command-line interface components of the unified vision processor.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from typer.testing import CliRunner

from vision_processor.config.unified_config import ModelType, UnifiedConfig


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI test runner for testing typer commands."""
    return CliRunner()


@pytest.fixture
def sample_image(temp_directory) -> Path:
    """Create a sample test image for CLI testing."""
    img = Image.new("RGB", (800, 600), color="white")
    img_path = temp_directory / "test_receipt.jpg"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_documents_dir(temp_directory) -> Path:
    """Create directory with multiple test documents for batch processing."""
    docs_dir = temp_directory / "documents"
    docs_dir.mkdir()

    # Create various document types
    for i, doc_type in enumerate(["receipt", "invoice", "statement"]):
        img = Image.new("RGB", (800, 600), color="white")
        img.save(docs_dir / f"{doc_type}_{i}.jpg")

    return docs_dir


@pytest.fixture
def temp_config_file(temp_directory) -> Path:
    """Create a temporary configuration file for CLI testing."""
    config_data = {
        "model_type": "internvl3",
        "processing_pipeline": "7step",
        "confidence_threshold": 0.8,
        "highlight_detection": True,
        "awk_fallback": True,
    }

    config_file = temp_directory / "test_config.json"
    with config_file.open("w") as f:
        json.dump(config_data, f)

    return config_file


@pytest.fixture
def expected_cli_output() -> dict:
    """Expected CLI output structure for validation."""
    return {
        "extracted_fields": {
            "amount": "123.45",
            "vendor": "Test Vendor",
            "date": "25/03/2024",
            "gst_amount": "12.35",
        },
        "confidence_score": 0.85,
        "processing_time": 1.5,
        "model_type": "internvl3",
    }


@pytest.fixture
def mock_extraction_manager():
    """Mock the UnifiedExtractionManager for CLI testing."""
    manager = MagicMock()
    mock_result = MagicMock()
    mock_result.extracted_fields = {
        "amount": "123.45",
        "vendor": "Test Vendor",
        "date": "25/03/2024",
        "gst_amount": "12.35",
    }
    mock_result.confidence_score = 0.85
    mock_result.processing_time = 1.5
    mock_result.model_type = "internvl3"
    mock_result.document_type = "business_receipt"
    mock_result.quality_grade = "GOOD"

    manager.return_value.__enter__.return_value.process_document.return_value = mock_result
    return manager


@pytest.fixture
def large_document_set(temp_directory) -> Path:
    """Create a large set of documents for performance testing."""
    docs_dir = temp_directory / "large_dataset"
    docs_dir.mkdir()

    # Create 100 test documents
    for i in range(100):
        img = Image.new("RGB", (400, 600), color="white")
        img.save(docs_dir / f"doc_{i:03d}.jpg")

    return docs_dir


@pytest.fixture
def cli_test_config() -> UnifiedConfig:
    """Create a specific configuration for CLI testing."""
    config = UnifiedConfig(testing_mode=True)
    config.model_type = ModelType.INTERNVL3
    config.model_path = "mock_model_path"
    config.confidence_threshold = 0.7
    config.highlight_detection = True
    config.awk_fallback = True
    config.batch_size = 2
    config.max_workers = 1
    return config


@pytest.fixture
def real_test_images():
    """Get paths to real test images from datasets directory."""
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    if not datasets_dir.exists():
        pytest.skip("Datasets directory not found")

    image_files = list(datasets_dir.glob("image*.png"))
    if not image_files:
        pytest.skip("No test images found in datasets directory")

    return image_files


@pytest.fixture
def mock_cli_dependencies(monkeypatch):
    """Mock all external dependencies for CLI testing."""
    # Mock the extraction manager
    mock_manager = MagicMock()
    mock_result = MagicMock()
    mock_result.extracted_fields = {"amount": "123.45", "vendor": "Test Store"}
    mock_result.confidence_score = 0.85
    mock_manager.return_value.__enter__.return_value.process_document.return_value = mock_result

    monkeypatch.setattr("vision_processor.cli.unified_cli.UnifiedExtractionManager", mock_manager)
    monkeypatch.setattr("vision_processor.cli.single_document.UnifiedExtractionManager", mock_manager)
    monkeypatch.setattr("vision_processor.cli.batch_processing.UnifiedExtractionManager", mock_manager)

    return mock_manager
