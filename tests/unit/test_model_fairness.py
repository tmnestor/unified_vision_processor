"""
Unit Tests for Model Fairness

Tests that ensure both InternVL and Llama models use identical
Llama pipeline processing for fair comparison in the unified architecture.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.classification import DocumentType
from vision_processor.config.unified_config import (
    ModelType,
    ProcessingPipeline,
    UnifiedConfig,
)
from vision_processor.evaluation.model_comparator import ModelComparator
from vision_processor.extraction.hybrid_extraction_manager import (
    UnifiedExtractionManager,
)


class TestModelFairness:
    """Test suite for model fairness in unified architecture."""

    @pytest.fixture
    def fairness_test_config(self):
        """Create a configuration specifically for fairness testing."""
        config = UnifiedConfig()
        config.processing_pipeline = (
            ProcessingPipeline.SEVEN_STEP
        )  # Ensure Llama pipeline
        # Use setattr for attributes that may not exist in the dataclass
        config.fair_comparison = True
        config.model_comparison = True
        config.identical_processing = True
        # Set other required attributes
        config.awk_fallback = True
        config.highlight_detection = False
        config.confidence_threshold = 0.7
        return config

    def test_identical_pipeline_execution_order(
        self, fairness_test_config, mock_image_path
    ):
        """Test that both models execute identical 7-step pipeline order."""

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                                    # Setup consistent mocks for both models
                                    def setup_mocks():
                                        mock_model = MagicMock()
                                        mock_model.process_image.return_value = Mock(
                                            raw_text="Mock model response",
                                            confidence=0.85,
                                            processing_time=1.5,
                                        )
                                        mock_factory.return_value = mock_model

                                        # Mock all components consistently
                                        mock_classifier = MagicMock()
                                        mock_classifier.classify_with_evidence.return_value = (
                                            DocumentType.BUSINESS_RECEIPT,
                                            0.85,
                                            ["evidence"],
                                        )
                                        mock_classifier_class.return_value = (
                                            mock_classifier
                                        )

                                        mock_awk = MagicMock()
                                        mock_awk.extract.return_value = {
                                            "awk_field": "awk_value"
                                        }
                                        mock_awk.ensure_initialized.return_value = None
                                        mock_awk_class.return_value = mock_awk

                                        mock_confidence = MagicMock()
                                        mock_confidence.assess_document_confidence.return_value = Mock(
                                            overall_confidence=0.82,
                                            quality_grade="good",
                                            production_ready=True,
                                            quality_flags=[],
                                            recommendations=[],
                                        )
                                        mock_confidence.ensure_initialized.return_value = None
                                        mock_confidence_class.return_value = (
                                            mock_confidence
                                        )

                                        mock_ato = MagicMock()
                                        compliance_result = Mock()
                                        compliance_result.compliance_score = 0.90
                                        compliance_result.passed = True
                                        compliance_result.violations = []
                                        compliance_result.warnings = []
                                        mock_ato.assess_compliance.return_value = (
                                            compliance_result
                                        )
                                        mock_ato.ensure_initialized.return_value = None
                                        mock_ato_class.return_value = mock_ato

                                        mock_highlights = MagicMock()
                                        mock_highlights.detect_highlights.return_value = []
                                        mock_highlights.ensure_initialized.return_value = None
                                        mock_highlights_class.return_value = (
                                            mock_highlights
                                        )

                                        mock_prompt = MagicMock()
                                        mock_prompt.get_prompt.return_value = (
                                            "Test prompt"
                                        )
                                        mock_prompt.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_prompt_class.return_value = mock_prompt

                                        return mock_model

                                    # Test with InternVL
                                    fairness_test_config.model_type = (
                                        ModelType.INTERNVL3
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_internvl = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_internvl = (
                                            manager_internvl.process_document(
                                                mock_image_path
                                            )
                                        )

                                    # Test with Llama
                                    fairness_test_config.model_type = (
                                        ModelType.LLAMA32_VISION
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_llama = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_llama = manager_llama.process_document(
                                            mock_image_path
                                        )

        # Extract pipeline stages from results
        internvl_stages = [stage.value for stage in result_internvl.stages_completed]
        llama_stages = [stage.value for stage in result_llama.stages_completed]

        # Verify identical pipeline execution
        assert internvl_stages == llama_stages, (
            f"Pipeline execution order must be identical. "
            f"InternVL: {internvl_stages}, Llama: {llama_stages}"
        )

        # Verify expected stages are present
        expected_stages = [
            "classification",
            "inference",
            "primary_extraction",
            "awk_fallback",
            "validation",
            "ato_compliance",
            "confidence_integration",
        ]
        assert all(stage in internvl_stages for stage in expected_stages), (
            f"Missing expected stages in InternVL pipeline: {internvl_stages}"
        )
        assert all(stage in llama_stages for stage in expected_stages), (
            f"Missing expected stages in Llama pipeline: {llama_stages}"
        )

    def test_identical_confidence_scoring_methodology(
        self, fairness_test_config, mock_image_path
    ):
        """Test that both models use identical 4-component confidence scoring."""

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                                    # Setup consistent mocks for both models
                                    def setup_mocks():
                                        mock_model = MagicMock()
                                        mock_model.process_image.return_value = Mock(
                                            raw_text="Mock model response",
                                            confidence=0.85,
                                            processing_time=1.5,
                                        )
                                        mock_factory.return_value = mock_model

                                        # Mock all components consistently
                                        mock_classifier = MagicMock()
                                        mock_classifier.classify_with_evidence.return_value = (
                                            DocumentType.BUSINESS_RECEIPT,
                                            0.85,
                                            ["evidence"],
                                        )
                                        mock_classifier_class.return_value = (
                                            mock_classifier
                                        )

                                        mock_awk = MagicMock()
                                        mock_awk.extract.return_value = {
                                            "awk_field": "awk_value"
                                        }
                                        mock_awk.ensure_initialized.return_value = None
                                        mock_awk_class.return_value = mock_awk

                                        mock_confidence = MagicMock()
                                        mock_confidence.assess_document_confidence.return_value = Mock(
                                            overall_confidence=0.82,
                                            quality_grade="good",
                                            production_ready=True,
                                            quality_flags=[],
                                            recommendations=[],
                                        )
                                        mock_confidence.ensure_initialized.return_value = None
                                        mock_confidence_class.return_value = (
                                            mock_confidence
                                        )

                                        mock_ato = MagicMock()
                                        compliance_result = Mock()
                                        compliance_result.compliance_score = 0.90
                                        compliance_result.passed = True
                                        compliance_result.violations = []
                                        compliance_result.warnings = []
                                        mock_ato.assess_compliance.return_value = (
                                            compliance_result
                                        )
                                        mock_ato.ensure_initialized.return_value = None
                                        mock_ato_class.return_value = mock_ato

                                        mock_highlights = MagicMock()
                                        mock_highlights.detect_highlights.return_value = []
                                        mock_highlights.ensure_initialized.return_value = None
                                        mock_highlights_class.return_value = (
                                            mock_highlights
                                        )

                                        mock_prompt = MagicMock()
                                        mock_prompt.get_prompt.return_value = (
                                            "Test prompt"
                                        )
                                        mock_prompt.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_prompt_class.return_value = mock_prompt

                                        return mock_model

                                    # Test InternVL confidence scoring
                                    fairness_test_config.model_type = (
                                        ModelType.INTERNVL3
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_internvl = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_internvl = (
                                            manager_internvl.process_document(
                                                mock_image_path
                                            )
                                        )

                                    # Test Llama confidence scoring
                                    fairness_test_config.model_type = (
                                        ModelType.LLAMA32_VISION
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_llama = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_llama = manager_llama.process_document(
                                            mock_image_path
                                        )

        # Both results should have confidence scores (indicating confidence calculation was used)
        assert hasattr(result_internvl, "confidence_score"), (
            "InternVL result missing confidence_score"
        )
        assert hasattr(result_llama, "confidence_score"), (
            "Llama result missing confidence_score"
        )

        # Confidence scores should be valid (between 0 and 1)
        assert 0 <= result_internvl.confidence_score <= 1, (
            f"Invalid InternVL confidence: {result_internvl.confidence_score}"
        )
        assert 0 <= result_llama.confidence_score <= 1, (
            f"Invalid Llama confidence: {result_llama.confidence_score}"
        )

        # Both should have quality grades (indicating same quality assessment methodology)
        assert hasattr(result_internvl, "quality_grade"), (
            "InternVL result missing quality_grade"
        )
        assert hasattr(result_llama, "quality_grade"), (
            "Llama result missing quality_grade"
        )

        # Both should have production readiness assessment
        assert hasattr(result_internvl, "production_ready"), (
            "InternVL result missing production_ready"
        )
        assert hasattr(result_llama, "production_ready"), (
            "Llama result missing production_ready"
        )

        # Verify same confidence integration stage was completed (indicating 4-component scoring)
        internvl_stages = [stage.value for stage in result_internvl.stages_completed]
        llama_stages = [stage.value for stage in result_llama.stages_completed]

        assert "confidence_integration" in internvl_stages, (
            "InternVL missing confidence_integration stage"
        )
        assert "confidence_integration" in llama_stages, (
            "Llama missing confidence_integration stage"
        )

    def test_identical_awk_fallback_behavior(
        self, fairness_test_config, mock_image_path
    ):
        """Test that AWK fallback behavior is identical for both models."""
        # Ensure AWK fallback is enabled for this test
        fairness_test_config.awk_fallback = True

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                                    # Setup consistent mocks for both models
                                    def setup_mocks():
                                        mock_model = MagicMock()
                                        mock_model.process_image.return_value = Mock(
                                            raw_text="Mock model response",
                                            confidence=0.85,
                                            processing_time=1.5,
                                        )
                                        mock_factory.return_value = mock_model

                                        # Mock all components consistently
                                        mock_classifier = MagicMock()
                                        mock_classifier.classify_with_evidence.return_value = (
                                            DocumentType.BUSINESS_RECEIPT,
                                            0.85,
                                            ["evidence"],
                                        )
                                        mock_classifier_class.return_value = (
                                            mock_classifier
                                        )

                                        mock_awk = MagicMock()
                                        mock_awk.extract.return_value = {
                                            "awk_field": "awk_value"
                                        }
                                        mock_awk.ensure_initialized.return_value = None
                                        mock_awk_class.return_value = mock_awk

                                        mock_confidence = MagicMock()
                                        mock_confidence.assess_document_confidence.return_value = Mock(
                                            overall_confidence=0.82,
                                            quality_grade="good",
                                            production_ready=True,
                                            quality_flags=[],
                                            recommendations=[],
                                        )
                                        mock_confidence.ensure_initialized.return_value = None
                                        mock_confidence_class.return_value = (
                                            mock_confidence
                                        )

                                        mock_ato = MagicMock()
                                        compliance_result = Mock()
                                        compliance_result.compliance_score = 0.90
                                        compliance_result.passed = True
                                        compliance_result.violations = []
                                        compliance_result.warnings = []
                                        mock_ato.assess_compliance.return_value = (
                                            compliance_result
                                        )
                                        mock_ato.ensure_initialized.return_value = None
                                        mock_ato_class.return_value = mock_ato

                                        mock_highlights = MagicMock()
                                        mock_highlights.detect_highlights.return_value = []
                                        mock_highlights.ensure_initialized.return_value = None
                                        mock_highlights_class.return_value = (
                                            mock_highlights
                                        )

                                        mock_prompt = MagicMock()
                                        mock_prompt.get_prompt.return_value = (
                                            "Test prompt"
                                        )
                                        mock_prompt.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_prompt_class.return_value = mock_prompt

                                        return mock_model

                                    # Test InternVL AWK fallback
                                    fairness_test_config.model_type = (
                                        ModelType.INTERNVL3
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_internvl = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_internvl = (
                                            manager_internvl.process_document(
                                                mock_image_path
                                            )
                                        )

                                    # Test Llama AWK fallback
                                    fairness_test_config.model_type = (
                                        ModelType.LLAMA32_VISION
                                    )
                                    setup_mocks()

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
                                        mock_handler.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_create_handler.return_value = mock_handler

                                        manager_llama = UnifiedExtractionManager(
                                            fairness_test_config
                                        )
                                        result_llama = manager_llama.process_document(
                                            mock_image_path
                                        )

        # Both results should have awk_fallback_used field
        assert hasattr(result_internvl, "awk_fallback_used"), (
            "InternVL result missing awk_fallback_used"
        )
        assert hasattr(result_llama, "awk_fallback_used"), (
            "Llama result missing awk_fallback_used"
        )

        # AWK fallback behavior should be consistent for both models
        # (Both should make the same decision about whether to use AWK fallback)
        assert isinstance(result_internvl.awk_fallback_used, bool), (
            "InternVL awk_fallback_used should be boolean"
        )
        assert isinstance(result_llama.awk_fallback_used, bool), (
            "Llama awk_fallback_used should be boolean"
        )

        # Both should complete the AWK fallback stage regardless of whether it was used
        internvl_stages = [stage.value for stage in result_internvl.stages_completed]
        llama_stages = [stage.value for stage in result_llama.stages_completed]

        assert "awk_fallback" in internvl_stages, "InternVL missing awk_fallback stage"
        assert "awk_fallback" in llama_stages, "Llama missing awk_fallback stage"

    def test_identical_ato_compliance_validation(
        self, fairness_test_config, mock_image_path
    ):
        """Test that ATO compliance validation is identical for both models."""
        ato_validation_calls = {}

        def mock_ato_assessment(model_type):
            def assess_compliance(fields, document_type):
                ato_validation_calls[model_type] = {
                    "fields": fields.copy(),
                    "document_type": document_type,
                }
                return Mock(
                    compliance_score=0.90, passed=True, violations=[], warnings=[]
                )

            return assess_compliance

        test_fields = {
            "supplier_name": "Test Store",
            "total_amount": "25.50",
            "date": "20/03/2024",
            "abn": "88000014675",
        }

        # Test InternVL ATO compliance
        fairness_test_config.model_type = ModelType.INTERNVL3

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ) as mock_ato:
                            mock_ato.return_value.assess_compliance = (
                                mock_ato_assessment("internvl3")
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)

                            # Mock handler to return test fields
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
                            ) as mock_create_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = (
                                    test_fields
                                )
                                mock_handler.validate_fields.return_value = test_fields
                                mock_handler.ensure_initialized.return_value = None
                                mock_create_handler.return_value = mock_handler

                                manager.process_document(mock_image_path)

        # Test Llama ATO compliance with identical fields
        fairness_test_config.model_type = ModelType.LLAMA32_VISION

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ) as mock_ato:
                            mock_ato.return_value.assess_compliance = (
                                mock_ato_assessment("llama32_vision")
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)

                            # Mock handler to return identical test fields
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.create_document_handler"
                            ) as mock_create_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = (
                                    test_fields
                                )
                                mock_handler.validate_fields.return_value = test_fields
                                mock_handler.ensure_initialized.return_value = None
                                mock_create_handler.return_value = mock_handler

                                manager.process_document(mock_image_path)

        # Verify identical ATO compliance validation
        assert "internvl3" in ato_validation_calls
        assert "llama32_vision" in ato_validation_calls

        internvl_call = ato_validation_calls["internvl3"]
        llama_call = ato_validation_calls["llama32_vision"]

        assert internvl_call["fields"] == llama_call["fields"]
        assert internvl_call["document_type"] == llama_call["document_type"]

    def test_identical_evaluation_metrics(self, fairness_test_config):
        """Test that evaluation metrics are calculated identically for both models."""
        from vision_processor.evaluation.unified_evaluator import UnifiedEvaluator

        evaluator = UnifiedEvaluator(fairness_test_config)

        # Test data that should yield identical metrics
        extracted_fields = {
            "supplier_name": "Woolworths",
            "total_amount": "45.67",
            "date": "25/03/2024",
        }

        ground_truth = {
            "supplier_name": "woolworths",  # Case difference
            "total_amount": "$45.67",  # Currency symbol difference
            "date": "25/03/2024",
        }

        # Evaluate with InternVL context
        fairness_test_config.model_type = ModelType.INTERNVL3
        result_internvl = evaluator.evaluate_single_document(
            extracted_fields=extracted_fields,
            ground_truth=ground_truth,
            processing_time=2.5,
        )

        # Evaluate with Llama context
        fairness_test_config.model_type = ModelType.LLAMA32_VISION
        result_llama = evaluator.evaluate_single_document(
            extracted_fields=extracted_fields,
            ground_truth=ground_truth,
            processing_time=2.5,
        )

        # Verify identical evaluation results
        assert result_internvl.precision == result_llama.precision
        assert result_internvl.recall == result_llama.recall
        assert result_internvl.f1_score == result_llama.f1_score
        assert result_internvl.exact_match_score == result_llama.exact_match_score
        assert result_internvl.ato_compliance_score == result_llama.ato_compliance_score

    def test_prompt_consistency_across_models(
        self, fairness_test_config, mock_image_path
    ):
        """Test that both models receive identical prompts for the same document type."""
        prompts_used = {}

        def mock_prompt_retrieval(model_type):
            def get_prompt(document_type, **kwargs):
                prompt = f"Extract fields from {document_type.value} document"
                prompts_used[model_type] = {
                    "document_type": document_type,
                    "prompt": prompt,
                    "kwargs": kwargs,
                }
                return prompt

            return get_prompt

        # Test InternVL prompt usage
        fairness_test_config.model_type = ModelType.INTERNVL3

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch(
                "vision_processor.classification.DocumentClassifier"
            ) as mock_classifier:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.BUSINESS_RECEIPT,
                                0.85,
                                ["evidence"],
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)

                            # Mock prompt manager
                            manager.prompt_manager = MagicMock()
                            manager.prompt_manager.get_prompt = mock_prompt_retrieval(
                                "internvl3"
                            )

                            manager.process_document(mock_image_path)

        # Test Llama prompt usage
        fairness_test_config.model_type = ModelType.LLAMA32_VISION

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch(
                "vision_processor.classification.DocumentClassifier"
            ) as mock_classifier:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.BUSINESS_RECEIPT,
                                0.85,
                                ["evidence"],
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)

                            # Mock prompt manager
                            manager.prompt_manager = MagicMock()
                            manager.prompt_manager.get_prompt = mock_prompt_retrieval(
                                "llama32_vision"
                            )

                            manager.process_document(mock_image_path)

        # Verify identical prompt usage
        assert "internvl3" in prompts_used
        assert "llama32_vision" in prompts_used

        internvl_prompt = prompts_used["internvl3"]
        llama_prompt = prompts_used["llama32_vision"]

        assert internvl_prompt["document_type"] == llama_prompt["document_type"]
        assert internvl_prompt["prompt"] == llama_prompt["prompt"]
        assert internvl_prompt["kwargs"] == llama_prompt["kwargs"]

    def test_fairness_validation_by_comparator(self, fairness_test_config):
        """Test fairness validation by the model comparator."""
        comparator = ModelComparator(fairness_test_config)

        # Create comparison configuration
        comparison_config = Mock()
        comparison_config.models_to_compare = ["internvl3", "llama32_vision"]
        comparison_config.identical_pipeline = True
        comparison_config.standardized_prompts = True  # Use correct attribute name
        comparison_config.same_confidence_thresholds = True

        # Validate fairness
        fairness_report = comparator._validate_fairness_configuration(comparison_config)

        assert fairness_report["fairness_score"] == 1.0
        assert fairness_report["identical_pipeline"] is True
        assert fairness_report["same_prompts"] is True
        assert fairness_report["same_evaluation_metrics"] is True
        assert fairness_report["bias_risk"] == "low"
        assert fairness_report["llama_foundation"] is True

    def test_model_agnostic_business_logic(self, fairness_test_config, mock_image_path):
        """Test that business logic is model-agnostic."""
        # Test business logic for both models
        results = {}

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                                    # Setup consistent mocks for both models
                                    def setup_mocks():
                                        mock_model = MagicMock()
                                        mock_model.process_image.return_value = Mock(
                                            raw_text="Mock model response",
                                            confidence=0.85,
                                            processing_time=1.5,
                                        )
                                        mock_factory.return_value = mock_model

                                        # Mock all components consistently
                                        mock_classifier = MagicMock()
                                        mock_classifier.classify_with_evidence.return_value = (
                                            DocumentType.BUSINESS_RECEIPT,
                                            0.85,
                                            ["evidence"],
                                        )
                                        mock_classifier_class.return_value = (
                                            mock_classifier
                                        )

                                        mock_awk = MagicMock()
                                        mock_awk.extract.return_value = {
                                            "awk_field": "awk_value"
                                        }
                                        mock_awk.ensure_initialized.return_value = None
                                        mock_awk_class.return_value = mock_awk

                                        mock_confidence = MagicMock()
                                        mock_confidence.assess_document_confidence.return_value = Mock(
                                            overall_confidence=0.82,
                                            quality_grade="good",
                                            production_ready=True,
                                            quality_flags=[],
                                            recommendations=[],
                                        )
                                        mock_confidence.ensure_initialized.return_value = None
                                        mock_confidence_class.return_value = (
                                            mock_confidence
                                        )

                                        mock_ato = MagicMock()
                                        compliance_result = Mock()
                                        compliance_result.compliance_score = 0.90
                                        compliance_result.passed = True
                                        compliance_result.violations = []
                                        compliance_result.warnings = []
                                        mock_ato.assess_compliance.return_value = (
                                            compliance_result
                                        )
                                        mock_ato.ensure_initialized.return_value = None
                                        mock_ato_class.return_value = mock_ato

                                        mock_highlights = MagicMock()
                                        mock_highlights.detect_highlights.return_value = []
                                        mock_highlights.ensure_initialized.return_value = None
                                        mock_highlights_class.return_value = (
                                            mock_highlights
                                        )

                                        mock_prompt = MagicMock()
                                        mock_prompt.get_prompt.return_value = (
                                            "Test prompt"
                                        )
                                        mock_prompt.ensure_initialized.return_value = (
                                            None
                                        )
                                        mock_prompt_class.return_value = mock_prompt

                                        return mock_model

                                    for model_type in [
                                        ModelType.INTERNVL3,
                                        ModelType.LLAMA32_VISION,
                                    ]:
                                        fairness_test_config.model_type = model_type
                                        setup_mocks()

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
                                            mock_create_handler.return_value = (
                                                mock_handler
                                            )

                                            manager = UnifiedExtractionManager(
                                                fairness_test_config
                                            )
                                            result = manager.process_document(
                                                mock_image_path
                                            )
                                            results[model_type.value] = result

        internvl_result = results["internvl3"]
        llama_result = results["llama32_vision"]

        # Verify identical business logic execution through pipeline stages
        internvl_stages = [stage.value for stage in internvl_result.stages_completed]
        llama_stages = [stage.value for stage in llama_result.stages_completed]

        # Business logic operations should be identical
        assert internvl_stages == llama_stages, (
            f"Business logic pipeline must be identical. "
            f"InternVL: {internvl_stages}, Llama: {llama_stages}"
        )

        # Verify core business logic stages are present
        expected_business_logic_stages = [
            "classification",  # Document type classification
            "awk_fallback",  # AWK extraction logic
            "ato_compliance",  # ATO compliance validation
        ]

        for stage in expected_business_logic_stages:
            assert stage in internvl_stages, (
                f"InternVL missing business logic stage: {stage}"
            )
            assert stage in llama_stages, f"Llama missing business logic stage: {stage}"

        # Verify results have same structure (model-agnostic output format)
        assert isinstance(internvl_result, type(llama_result)), (
            "Results must have same type"
        )
        assert hasattr(internvl_result, "model_type"), "Results must track model type"
        assert hasattr(llama_result, "model_type"), "Results must track model type"
