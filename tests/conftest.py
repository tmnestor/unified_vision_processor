"""
Test Configuration and Fixtures

Provides shared test fixtures, configurations, and utilities for the
unified vision processor testing framework.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock

import pytest

from vision_processor.classification import DocumentType
from vision_processor.config.unified_config import ModelType, UnifiedConfig
from vision_processor.extraction.hybrid_extraction_manager import (
    ProcessingResult,
    QualityGrade,
)


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_image_path(temp_directory: Path) -> Path:
    """Create a mock image file for testing."""
    image_path = temp_directory / "test_image.jpg"
    image_path.write_bytes(b"mock_image_data")
    return image_path


@pytest.fixture
def test_config() -> UnifiedConfig:
    """Create a test configuration with safe defaults."""
    config = UnifiedConfig()

    # Model configuration
    config.model_type = ModelType.INTERNVL3
    config.model_path = "mock_model_path"
    config.device_config = "cpu"  # Use CPU for testing

    # Processing configuration
    config.processing_pipeline = "7step"
    config.extraction_method = "hybrid"
    config.quality_threshold = 0.6
    config.confidence_threshold = 0.7

    # Feature flags
    config.highlight_detection = True
    config.awk_fallback = True
    config.computer_vision = False  # Disable for unit tests
    config.graceful_degradation = True

    # Performance settings
    config.batch_size = 1
    config.max_workers = 1
    config.gpu_memory_fraction = 0.5

    return config


@pytest.fixture
def mock_model_response() -> Mock:
    """Create a mock model response."""
    response = Mock()
    response.raw_text = """
    Store Name: Woolworths
    Date: 25/03/2024
    Total Amount: $45.67
    GST: $4.15
    ABN: 88 000 014 675
    """
    response.confidence = 0.85
    response.processing_time = 2.3
    return response


@pytest.fixture
def sample_extracted_fields() -> Dict[str, Any]:
    """Sample extracted fields for testing."""
    return {
        "supplier_name": "Woolworths",
        "date": "25/03/2024",
        "total_amount": "45.67",
        "gst_amount": "4.15",
        "abn": "88000014675",
        "address": "123 Test St, Sydney NSW",
        "invoice_number": "INV-12345",
    }


@pytest.fixture
def sample_ground_truth() -> Dict[str, Any]:
    """Sample ground truth data for testing."""
    return {
        "supplier_name": "woolworths",
        "date": "25/03/2024",
        "total_amount": "$45.67",
        "gst_amount": "$4.15",
        "abn": "88 000 014 675",
        "address": "123 test st, sydney nsw",
        "invoice_number": "inv-12345",
    }


@pytest.fixture
def sample_processing_result() -> ProcessingResult:
    """Create a sample processing result for testing."""
    return ProcessingResult(
        model_type="internvl3",
        document_type="business_receipt",
        raw_response="Mock model response text",
        extracted_fields={
            "supplier_name": "Test Store",
            "total_amount": "25.50",
            "date": "20/03/2024",
        },
        awk_fallback_used=False,
        highlights_detected=2,
        confidence_score=0.85,
        quality_grade=QualityGrade.GOOD,
        ato_compliance_score=0.90,
        production_ready=True,
        validation_passed=True,
        processing_time=2.1,
        memory_usage_mb=1024.5,
        quality_flags=["high_confidence"],
        recommendations=["Ready for production"],
        stages_completed=[],
        errors=[],
        warnings=[],
    )


@pytest.fixture
def mock_unified_config() -> MagicMock:
    """Create a mock unified configuration."""
    config = MagicMock(spec=UnifiedConfig)
    config.model_type = ModelType.INTERNVL3
    config.processing_pipeline = "7step"
    config.confidence_threshold = 0.7
    config.highlight_detection = True
    config.awk_fallback = True
    config.graceful_degradation = True
    return config


@pytest.fixture
def sample_confidence_scores() -> Dict[str, float]:
    """Sample confidence scores for testing."""
    return {
        "supplier_name": 0.95,
        "total_amount": 0.88,
        "date": 0.92,
        "abn": 0.78,
        "gst_amount": 0.85,
    }


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock vision model."""
    model = MagicMock()
    model.process_image.return_value = Mock(
        raw_text="Mock response", confidence=0.85, processing_time=1.5
    )
    model.model_type = "internvl3"
    model.device = "cpu"
    return model


@pytest.fixture
def sample_dataset_files(temp_directory: Path) -> tuple[Path, Path]:
    """Create sample dataset and ground truth files."""
    dataset_dir = temp_directory / "dataset"
    ground_truth_dir = temp_directory / "ground_truth"

    dataset_dir.mkdir()
    ground_truth_dir.mkdir()

    # Create test images
    for i in range(3):
        image_file = dataset_dir / f"test_image_{i}.jpg"
        image_file.write_bytes(b"mock_image_data")

        # Create corresponding ground truth
        gt_file = ground_truth_dir / f"test_image_{i}.json"
        gt_data = {
            "supplier_name": f"Store {i}",
            "total_amount": f"{20 + i * 5}.50",
            "date": f"2{i}/03/2024",
        }
        gt_file.write_text(json.dumps(gt_data))

    return dataset_dir, ground_truth_dir


@pytest.fixture
def mock_awk_extractor() -> MagicMock:
    """Create a mock AWK extractor."""
    extractor = MagicMock()
    extractor.extract.return_value = {
        "awk_supplier_name": "AWK Store",
        "awk_total_amount": "30.00",
    }
    return extractor


@pytest.fixture
def mock_highlight_detector() -> MagicMock:
    """Create a mock highlight detector."""
    detector = MagicMock()
    detector.detect_highlights.return_value = [
        {"color": "yellow", "bbox": [100, 100, 200, 120], "text": "Important text"},
        {"color": "green", "bbox": [100, 150, 250, 170], "text": "Another highlight"},
    ]
    return detector


class MockDocumentClassifier:
    """Mock document classifier for testing."""

    def __init__(self, config):
        self.config = config

    def classify_with_evidence(self, _image_path):
        """Mock classification with evidence."""
        return DocumentType.BUSINESS_RECEIPT, 0.85, ["confidence_evidence"]

    def classify(self, _image_path):
        """Mock simple classification."""
        return DocumentType.BUSINESS_RECEIPT


class MockATOCompliance:
    """Mock ATO compliance checker for testing."""

    def __init__(self, config):
        self.config = config

    def assess_compliance(self, _fields, _document_type):
        """Mock compliance assessment."""
        result = Mock()
        result.compliance_score = 0.90
        result.passed = True
        result.violations = []
        result.warnings = []
        return result


@pytest.fixture
def mock_document_classifier() -> MockDocumentClassifier:
    """Create a mock document classifier."""
    return MockDocumentClassifier(None)


@pytest.fixture
def mock_ato_compliance() -> MockATOCompliance:
    """Create a mock ATO compliance checker."""
    return MockATOCompliance(None)


# Test data constants
AUSTRALIAN_TEST_BUSINESSES = [
    "Woolworths",
    "Coles",
    "JB Hi-Fi",
    "Harvey Norman",
    "Big W",
    "Bunnings Warehouse",
]

VALID_ABNS = [
    "88 000 014 675",  # Woolworths
    "88000014675",  # Woolworths (no spaces)
    "51 004 085 616",  # Coles
    "63 000 240 417",  # JB Hi-Fi
]

INVALID_ABNS = [
    "12345",  # Too short
    "abc123def456",  # Contains letters
    "88 000 014 999",  # Invalid check digit
    "",  # Empty
]

SAMPLE_DATES = ["25/03/2024", "1/1/2024", "31/12/2023", "15-03-2024", "25 Mar 2024"]

SAMPLE_AMOUNTS = ["$45.67", "45.67", "$1,234.56", "1234.56", "$0.50"]
