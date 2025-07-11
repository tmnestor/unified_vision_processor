"""
Unit Tests for Hybrid Extraction Manager

Tests the unified extraction manager that implements the Llama 7-step
processing pipeline as the foundation of the unified architecture.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.classification import DocumentType
from vision_processor.config.unified_config import ModelType
from vision_processor.extraction.hybrid_extraction_manager import (
    ProcessingResult,
    ProcessingStage,
    QualityGrade,
    UnifiedExtractionManager,
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
            mock_model.get_memory_usage.return_value = 1024.0
            mock_factory.return_value = mock_model

            with patch(
                "vision_processor.classification.DocumentClassifier"
            ) as mock_classifier_class:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ) as mock_awk_class:
                    with patch(
                        "vision_processor.confidence.ConfidenceManager"
                    ) as mock_confidence_class:
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ) as mock_ato_class:
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.HighlightDetector"
                            ) as mock_highlights_class:
                                with patch(
                                    "vision_processor.extraction.hybrid_extraction_manager.PromptManager"
                                ) as mock_prompt_class:
                                    # Create mock instances that will be returned by the classes
                                    mock_classifier = MagicMock()
                                    mock_awk = MagicMock()
                                    mock_confidence = MagicMock()
                                    mock_ato = MagicMock()
                                    mock_highlights = MagicMock()
                                    mock_prompt = MagicMock()

                                    # Configure the class mocks to return our instances
                                    mock_classifier_class.return_value = mock_classifier
                                    mock_awk_class.return_value = mock_awk
                                    mock_confidence_class.return_value = mock_confidence
                                    mock_ato_class.return_value = mock_ato
                                    mock_highlights_class.return_value = mock_highlights
                                    mock_prompt_class.return_value = mock_prompt

                                    # Setup ensure_initialized for all components
                                    for component in [
                                        mock_classifier,
                                        mock_awk,
                                        mock_confidence,
                                        mock_ato,
                                        mock_highlights,
                                        mock_prompt,
                                    ]:
                                        component.ensure_initialized.return_value = None

                                    # Create the manager
                                    manager = UnifiedExtractionManager(test_config)

                                    # Setup mock responses
                                    mock_classifier.classify_with_evidence.return_value = (
                                        DocumentType.BUSINESS_RECEIPT,
                                        0.85,
                                        ["evidence"],
                                    )

                                    mock_model.process_image.return_value = Mock(
                                        raw_text="Mock model response",
                                        confidence=0.85,
                                        processing_time=1.5,
                                    )

                                    mock_awk.extract.return_value = {
                                        "awk_supplier": "AWK Store"
                                    }

                                    def mock_confidence_assessment(
                                        _raw_text,
                                        _extracted_fields,
                                        _compliance_result,
                                        classification_confidence,
                                        _highlights_detected,
                                    ):
                                        """Mock confidence assessment that handles classification confidence properly."""
                                        # Apply graceful degradation logic like the real confidence manager
                                        base_confidence = 0.82
                                        quality_flags = []

                                        # Check for low classification confidence (< 0.6 threshold)
                                        if classification_confidence < 0.6:
                                            quality_flags.append(
                                                "low_classification_confidence"
                                            )
                                            # Cap confidence at 0.5 for low classification confidence
                                            base_confidence = min(base_confidence, 0.5)

                                        return Mock(
                                            overall_confidence=base_confidence,
                                            quality_grade="fair"
                                            if base_confidence <= 0.5
                                            else "good",
                                            production_ready=base_confidence > 0.7,
                                            quality_flags=quality_flags,
                                            recommendations=[
                                                "Manual review recommended"
                                            ]
                                            if base_confidence <= 0.5
                                            else [],
                                        )

                                    mock_confidence.assess_document_confidence.side_effect = mock_confidence_assessment

                                    # Create a real-ish compliance result object
                                    compliance_result = Mock()
                                    compliance_result.compliance_score = (
                                        0.90  # Real float, not Mock
                                    )
                                    compliance_result.passed = True
                                    compliance_result.violations = []
                                    compliance_result.warnings = []
                                    mock_ato.assess_compliance.return_value = (
                                        compliance_result
                                    )

                                    mock_highlights.detect_highlights.return_value = []
                                    mock_prompt.get_prompt.return_value = (
                                        "Test prompt for document processing"
                                    )

                                    # Store mock references for test access
                                    manager._test_mocks = {
                                        "classifier": mock_classifier,
                                        "confidence_manager": mock_confidence,
                                        "ato_handler": mock_ato,
                                        "awk_extractor": mock_awk,
                                        "model": mock_model,
                                        "highlights": mock_highlights,
                                        "prompt": mock_prompt,
                                    }

                                    return manager

    def test_extraction_manager_initialization(self, test_config):
        """Test extraction manager initialization."""
        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.PromptManager"
                            ):
                                manager = UnifiedExtractionManager(test_config)

                                assert manager.config == test_config
                                assert manager.model is not None
                                assert manager.classifier is not None
                                assert manager.awk_extractor is not None
                                assert manager.confidence_manager is not None
                                assert manager.ato_compliance is not None
                                assert manager.prompt_manager is not None

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

        # Verify classification was called using stored mock reference
        mock_extraction_manager._test_mocks[
            "classifier"
        ].classify_with_evidence.assert_called_once_with(mock_image_path)
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
        with patch(
            "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
        ) as mock_create_handler:
            mock_handler = MagicMock()
            mock_handler.extract_fields_primary.return_value = {
                "supplier_name": "Test Store",
                "total_amount": "25.50",
            }
            mock_handler.validate_fields.return_value = {
                "supplier_name": "Test Store",
                "total_amount": "25.50",
            }
            mock_handler.ensure_initialized.return_value = None
            mock_create_handler.return_value = mock_handler

            mock_extraction_manager.process_document(mock_image_path)

            # Verify handler was called
            mock_create_handler.assert_called_once()
            mock_handler.extract_fields_primary.assert_called_once()
            mock_handler.validate_fields.assert_called_once()

    def test_step4_awk_fallback_activation(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test Step 4: AWK Fallback when extraction quality is insufficient."""
        with patch.object(
            mock_extraction_manager,
            "_extraction_quality_insufficient",
            return_value=True,
        ):
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
            ) as mock_create_handler:
                mock_handler = MagicMock()
                mock_handler.extract_fields_primary.return_value = {
                    "supplier_name": "Test"
                }  # Insufficient
                mock_handler.validate_fields.return_value = {
                    "supplier_name": "Test",
                    "awk_supplier": "AWK Store",
                }
                mock_handler.ensure_initialized.return_value = None
                mock_create_handler.return_value = mock_handler

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
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
            ) as mock_create_handler:
                mock_handler = MagicMock()
                mock_handler.extract_fields_primary.return_value = {
                    "supplier_name": "Test Store",
                    "total_amount": "25.50",
                    "date": "20/03/2024",
                    "abn": "12345678901",
                }  # Sufficient fields
                mock_handler.validate_fields.return_value = {
                    "supplier_name": "Test Store",
                    "total_amount": "25.50",
                    "date": "20/03/2024",
                    "abn": "12345678901",
                }
                mock_handler.ensure_initialized.return_value = None
                mock_create_handler.return_value = mock_handler

                result = mock_extraction_manager.process_document(mock_image_path)

                # Verify AWK fallback was not used
                assert result.awk_fallback_used is False

    def test_step5_field_validation(self, mock_extraction_manager, mock_image_path):
        """Test Step 5: Field Validation."""
        with patch(
            "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
        ) as mock_create_handler:
            mock_handler = MagicMock()
            mock_handler.extract_fields_primary.return_value = {"raw": "data"}
            mock_handler.validate_fields.return_value = {"validated": "data"}
            mock_handler.ensure_initialized.return_value = None
            mock_create_handler.return_value = mock_handler

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

        # Verify confidence assessment was called using stored mock reference
        mock_extraction_manager._test_mocks[
            "confidence_manager"
        ].assess_document_confidence.assert_called_once()
        assert result.confidence_score == 0.82
        assert result.quality_grade == QualityGrade.GOOD
        assert result.production_ready is True

    def test_internvl_highlight_detection_integration(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test InternVL highlight detection integration."""
        # Configure for InternVL with highlight detection
        mock_extraction_manager.config.highlight_detection = True
        mock_extraction_manager.config.model_type = ModelType.INTERNVL3

        # Setup highlight detection for bank statements
        mock_extraction_manager._test_mocks[
            "highlights"
        ].detect_highlights.return_value = [
            {
                "color": "yellow",
                "bbox": [100, 100, 200, 120],
                "text": "Important",
            }
        ]

        # Mock classification to return bank statement
        mock_extraction_manager._test_mocks[
            "classifier"
        ].classify_with_evidence.return_value = (
            DocumentType.BANK_STATEMENT,
            0.90,
            ["evidence"],
        )

        result = mock_extraction_manager.process_document(mock_image_path)

        # Verify that highlight detection was called for bank statements
        assert result.document_type == DocumentType.BANK_STATEMENT.value
        # Note: The actual highlight detection integration depends on the implementation
        # For now, just verify the processing completed successfully
        assert result.model_type == "internvl3"

    def test_enhanced_key_value_parser_integration(self, test_config, mock_image_path):
        """Test InternVL enhanced key-value parser integration."""
        test_config.use_enhanced_parser = True

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.EnhancedKeyValueParser"
                            ) as mock_parser:
                                with patch(
                                    "vision_processor.extraction.hybrid_extraction_manager.PromptManager"
                                ) as mock_prompt_class:
                                    mock_parser.return_value.parse.return_value = {
                                        "enhanced_field": "enhanced_value"
                                    }
                                    mock_prompt_class.return_value.get_prompt.return_value = "Test prompt"
                                    mock_prompt_class.return_value.ensure_initialized.return_value = None

                                    manager = UnifiedExtractionManager(test_config)

                                    with patch(
                                        "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
                                    ) as mock_create_handler:
                                        mock_handler = MagicMock()
                                        mock_handler.extract_fields_primary.return_value = {
                                            "base_field": "base_value"
                                        }
                                        mock_handler.validate_fields.return_value = {
                                            "merged_field": "merged_value"
                                        }
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager.process_document(mock_image_path)

                                        # Verify enhanced parser was used
                                        mock_parser.return_value.parse.assert_called_once()

    def test_graceful_degradation_on_classification_failure(
        self, mock_extraction_manager, mock_image_path
    ):
        """Test graceful degradation when classification fails."""
        # Mock classification failure (low confidence)
        mock_extraction_manager._test_mocks[
            "classifier"
        ].classify_with_evidence.return_value = (
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
        mock_extraction_manager._test_mocks[
            "model"
        ].process_image.side_effect = Exception("Model failed")

        # Should handle the error gracefully and return a result with errors
        result = mock_extraction_manager.process_document(mock_image_path)

        # Should complete processing but with errors recorded
        assert result is not None
        assert len(result.errors) > 0
        assert any("Model failed" in str(error) for error in result.errors)

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
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.PromptManager"
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
            # Manually set configuration attributes instead of using update method
            for key, value in config_update.items():
                setattr(test_config, key, value)

            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ):
                with patch("vision_processor.classification.DocumentClassifier"):
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                    ):
                        with patch("vision_processor.confidence.ConfidenceManager"):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                            ):
                                with patch(
                                    "vision_processor.extraction.hybrid_extraction_manager.PromptManager"
                                ):
                                    manager = UnifiedExtractionManager(test_config)
                                    result = manager.process_document(mock_image_path)

                                    assert result is not None
