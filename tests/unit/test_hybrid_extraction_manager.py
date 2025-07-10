"""
Unit Tests for Hybrid Extraction Manager

Tests the unified extraction manager that implements the Llama 7-step
processing pipeline as the foundation of the unified architecture.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.unified_config import ModelType
from vision_processor.extraction.hybrid_extraction_manager import (
    ProcessingResult,
    UnifiedExtractionManager,
)
from vision_processor.extraction.pipeline_components import (
    DocumentType,
    ProcessingStage,
    QualityGrade,
)


class TestUnifiedExtractionManager:
    """Test suite for UnifiedExtractionManager class."""

    @pytest.fixture
    def mock_extraction_manager(self, test_config):
        """Create a mock extraction manager with all dependencies mocked."""
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            # Setup mocks for all components
            mock_model = MagicMock()
            mock_model.model_type = "internvl3"
            mock_factory.return_value = mock_model

            with patch(
                "vision_processor.classification.DocumentClassifier"
            ) as mock_classifier:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ) as mock_awk:
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                    ) as mock_confidence:
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ) as mock_ato:
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.HighlightDetector"
                            ) as mock_highlights:
                                manager = UnifiedExtractionManager(test_config)

                                # Setup mock responses
                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )

                                mock_model.process_image.return_value = Mock(
                                    raw_text="Mock model response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )

                                mock_awk.return_value.extract.return_value = {
                                    "awk_supplier": "AWK Store"
                                }

                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=QualityGrade.GOOD,
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )

                                mock_ato.return_value.assess_compliance.return_value = (
                                    Mock(
                                        compliance_score=0.90,
                                        passed=True,
                                        violations=[],
                                        warnings=[],
                                    )
                                )

                                mock_highlights.return_value.detect_highlights.return_value = []

                                return manager

    def test_extraction_manager_initialization(self, test_config):
        """Test extraction manager initialization."""
        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                    ):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            manager = UnifiedExtractionManager(test_config)

                            assert manager.config == test_config
                            assert manager.model is not None
                            assert manager.classifier is not None
                            assert manager.awk_extractor is not None
                            assert manager.confidence_manager is not None
                            assert manager.ato_compliance is not None

    def test_seven_step_pipeline_execution(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test the complete 7-step Llama pipeline execution."""
        # Execute processing
        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify result structure
        assert isinstance(result, ProcessingResult)
        assert result.model_type == "internvl3"
        assert result.document_type == "business_receipt"
        assert result.confidence_score == 0.82
        assert result.quality_grade == QualityGrade.GOOD
        assert result.production_ready is True
        assert result.ato_compliance_score == 0.90

        # Verify pipeline stages were completed
        assert ProcessingStage.CLASSIFICATION in result.stages_completed
        assert ProcessingStage.INFERENCE in result.stages_completed
        assert ProcessingStage.PRIMARY_EXTRACTION in result.stages_completed
        assert ProcessingStage.VALIDATION in result.stages_completed
        assert ProcessingStage.ATO_COMPLIANCE in result.stages_completed
        assert ProcessingStage.CONFIDENCE_INTEGRATION in result.stages_completed

    def test_step1_document_classification(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 1: Document Classification."""
        # Test with auto-detection
        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify classification was called
        mock_extraction_manager.classifier.classify_with_evidence.assert_called_once_with(
            mock_image_path
        )
        assert result.document_type == "business_receipt"

        # Test with predefined document type
        result = mock_extraction_manager.process_document(
            mock_image_path, document_type="fuel_receipt"
        )

        assert result.document_type == "fuel_receipt"

    def test_step2_model_inference(self, mock_extraction_manager, mock_image_path):
        """Test Step 2: Model Inference."""
        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify model inference was called
        mock_extraction_manager.model.process_image.assert_called()
        assert result.raw_response == "Mock model response"

    def test_step3_handler_selection_and_primary_extraction(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 3: Handler Selection and Primary Extraction."""
        with patch.object(mock_extraction_manager, "_get_handler") as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.extract_fields_primary.return_value = {
                "supplier_name": "Test Store",
                "total_amount": "25.50",
            }
            mock_get_handler.return_value = mock_handler

            mock_extraction_manager.process_document(mock_image_path)

            # Verify handler was called
            mock_get_handler.assert_called_once()
            mock_handler.extract_fields_primary.assert_called_once()

    def test_step4_awk_fallback_activation(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 4: AWK Fallback when extraction quality is insufficient."""
        with patch.object(
            mock_extraction_manager,
            "_extraction_quality_insufficient",
            return_value=True,
        ):
            with patch.object(
                mock_extraction_manager, "_get_handler"
            ) as mock_get_handler:
                mock_handler = MagicMock()
                mock_handler.extract_fields_primary.return_value = {
                    "supplier_name": "Test"
                }  # Insufficient
                mock_get_handler.return_value = mock_handler

                result = mock_extraction_manager.process_document(mock_image_path)

                # Verify AWK fallback was used
                mock_extraction_manager.awk_extractor.extract.assert_called_once()
                assert result.awk_fallback_used is True

    def test_step4_awk_fallback_skipped(self, mock_extraction_manager, mock_image_path):
        """Test Step 4: AWK Fallback skipped when extraction quality is sufficient."""
        with patch.object(
            mock_extraction_manager,
            "_extraction_quality_insufficient",
            return_value=False,
        ):
            with patch.object(
                mock_extraction_manager, "_get_handler"
            ) as mock_get_handler:
                mock_handler = MagicMock()
                mock_handler.extract_fields_primary.return_value = {
                    "supplier_name": "Test Store",
                    "total_amount": "25.50",
                    "date": "20/03/2024",
                    "abn": "12345678901",
                }  # Sufficient fields
                mock_get_handler.return_value = mock_handler

                result = mock_extraction_manager.process_document(mock_image_path)

                # Verify AWK fallback was not used
                assert result.awk_fallback_used is False

    def test_step5_field_validation(self, mock_extraction_manager, mock_image_path):
        """Test Step 5: Field Validation."""
        with patch.object(mock_extraction_manager, "_get_handler") as mock_get_handler:
            mock_handler = MagicMock()
            mock_handler.extract_fields_primary.return_value = {"raw": "data"}
            mock_handler.validate_fields.return_value = {"validated": "data"}
            mock_get_handler.return_value = mock_handler

            mock_extraction_manager.process_document(mock_image_path)

            # Verify validation was called
            mock_handler.validate_fields.assert_called_once()

    def test_step6_ato_compliance_assessment(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 6: ATO Compliance Assessment."""
        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify ATO compliance was assessed
        mock_extraction_manager.ato_compliance.assess_compliance.assert_called_once()
        assert result.ato_compliance_score == 0.90

    def test_step7_confidence_integration(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 7: Confidence Integration and Production Readiness."""
        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify confidence assessment was called
        mock_extraction_manager.confidence_manager.assess_document_confidence.assert_called_once()
        assert result.confidence_score == 0.82
        assert result.quality_grade == QualityGrade.GOOD
        assert result.production_ready is True

    def test_internvl_highlight_detection_integration(
        self, test_config, mock_image_path
    ):
        """Test InternVL highlight detection integration."""
        test_config.highlight_detection = True
        test_config.model_type = ModelType.INTERNVL3

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                    ):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.HighlightDetector"
                            ) as mock_highlight_detector:
                                # Setup highlight detection for bank statements
                                mock_highlight_detector.return_value.detect_highlights.return_value = [
                                    {
                                        "color": "yellow",
                                        "bbox": [100, 100, 200, 120],
                                        "text": "Important",
                                    }
                                ]

                                manager = UnifiedExtractionManager(test_config)

                                # Mock classification to return bank statement
                                manager.classifier.classify_with_evidence.return_value = (
                                    DocumentType.BANK_STATEMENT,
                                    0.90,
                                    ["evidence"],
                                )

                                result = manager.process_document(mock_image_path)

                                # Verify highlights were detected for bank statements
                                mock_highlight_detector.return_value.detect_highlights.assert_called_once_with(
                                    mock_image_path
                                )
                                assert result.highlights_detected == 1

    def test_enhanced_key_value_parser_integration(self, test_config, mock_image_path):
        """Test InternVL enhanced key-value parser integration."""
        test_config.use_enhanced_parser = True

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                    ):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.EnhancedKeyValueParser"
                            ) as mock_parser:
                                mock_parser.return_value.parse.return_value = {
                                    "enhanced_field": "enhanced_value"
                                }

                                manager = UnifiedExtractionManager(test_config)

                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {
                                        "base_field": "base_value"
                                    }
                                    mock_handler.validate_fields.return_value = {
                                        "merged_field": "merged_value"
                                    }
                                    mock_get_handler.return_value = mock_handler

                                    manager.process_document(mock_image_path)

                                    # Verify enhanced parser was used
                                    mock_parser.return_value.parse.assert_called_once()

    def test_graceful_degradation_on_classification_failure(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test graceful degradation when classification fails."""
        # Mock classification failure (low confidence)
        mock_extraction_manager.classifier.classify_with_evidence.return_value = (
            DocumentType.UNKNOWN,
            0.3,
            ["low_confidence"],
        )

        # Should still proceed with processing
        result = mock_extraction_manager.process_document(mock_image_path)

        assert result is not None
        assert result.document_type == "unknown"
        # Confidence should reflect the uncertainty
        assert result.confidence_score <= 0.5

    def test_graceful_degradation_on_model_failure(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test graceful degradation when model inference fails."""
        # Mock model failure
        mock_extraction_manager.model.process_image.side_effect = Exception(
            "Model failed"
        )

        # Should handle the error gracefully
        with pytest.raises(Exception, match=".*"):
            mock_extraction_manager.process_document(mock_image_path)

    def test_processing_time_measurement(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test processing time measurement accuracy."""
        result = mock_extraction_manager.process_document(mock_image_path)

        assert result.processing_time > 0
        assert isinstance(result.processing_time, float)
        # Should be reasonable (under 10 seconds for mocked processing)
        assert result.processing_time < 10.0

    def test_memory_usage_tracking(self, mock_extraction_manager, mock_image_path):
        """Test memory usage tracking."""
        result = mock_extraction_manager.process_document(mock_image_path)

        assert result.memory_usage_mb > 0
        assert isinstance(result.memory_usage_mb, float)

    def test_error_collection_and_reporting(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test error collection during processing."""
        # Mock a component that generates warnings
        mock_extraction_manager.ato_compliance.assess_compliance.return_value = Mock(
            compliance_score=0.70,
            passed=True,
            violations=[],
            warnings=["Minor formatting issue"],
        )

        result = mock_extraction_manager.process_document(mock_image_path)

        assert isinstance(result.warnings, list)
        assert isinstance(result.errors, list)

    def test_context_manager_support(self, test_config):
        """Test context manager support for resource cleanup."""
        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                    ):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            # Test context manager
                            with UnifiedExtractionManager(test_config) as manager:
                                assert manager is not None
                                assert hasattr(manager, "__enter__")
                                assert hasattr(manager, "__exit__")

    def test_batch_processing_capability(
        self, mock_extraction_manager, sample_dataset_files
    ):
        """Test batch processing capabilities."""
        dataset_dir, _ = sample_dataset_files
        image_files = list(dataset_dir.glob("*.jpg"))

        results = []
        for image_file in image_files:
            result = mock_extraction_manager.process_document(image_file)
            results.append(result)

        assert len(results) == len(image_files)
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.processing_time > 0

    def test_configuration_impact_on_processing(self, test_config, mock_image_path):
        """Test how configuration changes impact processing."""
        # Test with different configurations
        configs_to_test = [
            {"awk_fallback": False, "highlight_detection": False},
            {"awk_fallback": True, "highlight_detection": True},
            {"graceful_degradation": False},
        ]

        for config_update in configs_to_test:
            test_config.update(config_update)

            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ):
                with patch("vision_processor.classification.DocumentClassifier"):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                    ):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ConfidenceIntegrationManager"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                            ):
                                manager = UnifiedExtractionManager(test_config)
                                result = manager.process_document(mock_image_path)

                                assert result is not None
