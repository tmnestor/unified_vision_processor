"""
Unit Tests for Model Fairness

Tests that ensure both InternVL and Llama models use identical
Llama pipeline processing for fair comparison in the unified architecture.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.unified_config import (
    ModelType,
    ProcessingPipeline,
    UnifiedConfig,
)
from vision_processor.evaluation.model_comparator import ModelComparator
from vision_processor.extraction.hybrid_extraction_manager import (
    UnifiedExtractionManager,
)
from vision_processor.extraction.pipeline_components import DocumentType


class TestModelFairness:
    """Test suite for model fairness in unified architecture."""

    @pytest.fixture
    def fairness_test_config(self):
        """Create a configuration specifically for fairness testing."""
        config = UnifiedConfig()
        config.processing_pipeline = (
            ProcessingPipeline.SEVEN_STEP
        )  # Ensure Llama pipeline
        config.fair_comparison = True
        config.model_comparison = True
        config.identical_processing = True
        return config

    def test_identical_pipeline_execution_order(
        self, fairness_test_config, mock_image_path
    ):
        """Test that both models execute identical 7-step pipeline order."""
        pipeline_execution_logs = []

        def log_pipeline_step(step_name, model_type):
            pipeline_execution_logs.append((step_name, model_type))

        # Test with InternVL
        fairness_test_config.model_type = ModelType.INTERNVL3

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                            # Setup mocks to log execution
                            def mock_classify_internvl(_x):
                                log_pipeline_step("classification", "internvl3")
                                return (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )

                            mock_classifier.return_value.classify_with_evidence.side_effect = mock_classify_internvl

                            mock_model = MagicMock()

                            def mock_process_internvl(_x, _y):
                                log_pipeline_step("inference", "internvl3")
                                return Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )

                            mock_model.process_image.side_effect = mock_process_internvl
                            mock_factory.return_value = mock_model

                            # Ensure all required mocks are configured
                            mock_classifier.return_value.ensure_initialized = (
                                MagicMock()
                            )

                            manager_internvl = UnifiedExtractionManager(
                                fairness_test_config
                            )
                            manager_internvl.process_document(mock_image_path)

        internvl_steps = [
            step for step, model in pipeline_execution_logs if model == "internvl3"
        ]

        # Reset logs
        pipeline_execution_logs.clear()

        # Test with Llama
        fairness_test_config.model_type = ModelType.LLAMA32_VISION

        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
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
                            # Setup identical mocks
                            def mock_classify_llama(_x):
                                log_pipeline_step("classification", "llama32_vision")
                                return (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )

                            mock_classifier.return_value.classify_with_evidence.side_effect = mock_classify_llama

                            mock_model = MagicMock()

                            def mock_process_llama(_x, _y):
                                log_pipeline_step("inference", "llama32_vision")
                                return Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )

                            mock_model.process_image.side_effect = mock_process_llama
                            mock_factory.return_value = mock_model

                            # Ensure all required mocks are configured
                            mock_classifier.return_value.ensure_initialized = (
                                MagicMock()
                            )

                            manager_llama = UnifiedExtractionManager(
                                fairness_test_config
                            )
                            manager_llama.process_document(mock_image_path)

        llama_steps = [
            step for step, model in pipeline_execution_logs if model == "llama32_vision"
        ]

        # Verify identical pipeline execution
        assert internvl_steps == llama_steps, (
            "Pipeline execution order must be identical"
        )
        assert "classification" in internvl_steps
        assert "inference" in internvl_steps

    def test_identical_confidence_scoring_methodology(
        self, fairness_test_config, mock_image_path
    ):
        """Test that both models use identical 4-component confidence scoring."""
        confidence_components_used = {}

        def mock_confidence_assessment(model_type):
            def assess_confidence(*_args, **kwargs):
                confidence_components_used[model_type] = {
                    "classification_confidence": kwargs.get(
                        "classification_confidence", 0.85
                    ),
                    "extraction_quality": 0.80,
                    "ato_compliance": 0.90,
                    "business_logic": 0.75,
                }
                return Mock(
                    overall_confidence=0.82,
                    quality_grade=Mock(value="good"),
                    production_ready=True,
                    quality_flags=[],
                    recommendations=[],
                )

            return assess_confidence

        # Test InternVL confidence scoring
        fairness_test_config.model_type = ModelType.INTERNVL3

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.confidence.ConfidenceManager"
                    ) as mock_confidence:
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_confidence.return_value.assess_document_confidence = (
                                mock_confidence_assessment("internvl3")
                            )
                            mock_confidence.return_value.ensure_initialized = (
                                MagicMock()
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)
                            manager.process_document(mock_image_path)

        # Test Llama confidence scoring
        fairness_test_config.model_type = ModelType.LLAMA32_VISION

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ):
                    with patch(
                        "vision_processor.confidence.ConfidenceManager"
                    ) as mock_confidence:
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_confidence.return_value.assess_document_confidence = (
                                mock_confidence_assessment("llama32_vision")
                            )
                            mock_confidence.return_value.ensure_initialized = (
                                MagicMock()
                            )

                            manager = UnifiedExtractionManager(fairness_test_config)
                            manager.process_document(mock_image_path)

        # Verify identical confidence components
        assert "internvl3" in confidence_components_used
        assert "llama32_vision" in confidence_components_used

        internvl_components = set(confidence_components_used["internvl3"].keys())
        llama_components = set(confidence_components_used["llama32_vision"].keys())

        assert internvl_components == llama_components, (
            "Confidence components must be identical"
        )

        # Verify 4-component system
        expected_components = {
            "classification_confidence",
            "extraction_quality",
            "ato_compliance",
            "business_logic",
        }
        assert internvl_components == expected_components

    def test_identical_awk_fallback_behavior(
        self, fairness_test_config, mock_image_path
    ):
        """Test that AWK fallback behavior is identical for both models."""
        awk_fallback_triggers = {}

        def mock_extraction_quality_check(model_type):
            def check_quality(extracted_fields):
                # Simulate insufficient extraction (triggers AWK fallback)
                field_count = len(extracted_fields)
                triggers_awk = field_count < 3
                awk_fallback_triggers[model_type] = triggers_awk
                return triggers_awk

            return check_quality

        # Test InternVL AWK fallback
        fairness_test_config.model_type = ModelType.INTERNVL3

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ) as mock_awk:
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_awk.return_value.extract.return_value = {
                                "awk_field": "awk_value"
                            }
                            mock_awk.return_value.ensure_initialized = MagicMock()

                            manager = UnifiedExtractionManager(fairness_test_config)
                            # Patch the quality check method
                            manager._extraction_quality_insufficient = (
                                mock_extraction_quality_check("internvl3")
                            )

                            # Mock handler to return insufficient fields
                            with patch.object(
                                manager, "_get_handler"
                            ) as mock_get_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = {
                                    "field1": "value1"
                                }  # Only 1 field
                                mock_handler.validate_fields.return_value = {
                                    "field1": "value1"
                                }
                                mock_get_handler.return_value = mock_handler

                                manager.process_document(mock_image_path)

        # Test Llama AWK fallback with identical conditions
        fairness_test_config.model_type = ModelType.LLAMA32_VISION

        with patch("vision_processor.config.model_factory.ModelFactory.create_model"):
            with patch("vision_processor.classification.DocumentClassifier"):
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                ) as mock_awk:
                    with patch("vision_processor.confidence.ConfidenceManager"):
                        with patch(
                            "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                        ):
                            mock_awk.return_value.extract.return_value = {
                                "awk_field": "awk_value"
                            }
                            mock_awk.return_value.ensure_initialized = MagicMock()

                            manager = UnifiedExtractionManager(fairness_test_config)
                            manager._extraction_quality_insufficient = (
                                mock_extraction_quality_check("llama32_vision")
                            )

                            # Mock handler to return identical insufficient fields
                            with patch.object(
                                manager, "_get_handler"
                            ) as mock_get_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = {
                                    "field1": "value1"
                                }  # Only 1 field
                                mock_handler.validate_fields.return_value = {
                                    "field1": "value1"
                                }
                                mock_get_handler.return_value = mock_handler

                                manager.process_document(mock_image_path)

        # Verify identical AWK fallback behavior
        assert "internvl3" in awk_fallback_triggers
        assert "llama32_vision" in awk_fallback_triggers
        assert (
            awk_fallback_triggers["internvl3"]
            == awk_fallback_triggers["llama32_vision"]
        )

        # Both should have triggered AWK fallback
        assert awk_fallback_triggers["internvl3"] is True
        assert awk_fallback_triggers["llama32_vision"] is True

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
                            with patch.object(
                                manager, "_get_handler"
                            ) as mock_get_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = (
                                    test_fields
                                )
                                mock_handler.validate_fields.return_value = test_fields
                                mock_get_handler.return_value = mock_handler

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
                            with patch.object(
                                manager, "_get_handler"
                            ) as mock_get_handler:
                                mock_handler = MagicMock()
                                mock_handler.extract_fields_primary.return_value = (
                                    test_fields
                                )
                                mock_handler.validate_fields.return_value = test_fields
                                mock_get_handler.return_value = mock_handler

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
        comparison_config.same_prompts = True
        comparison_config.same_evaluation_metrics = True
        comparison_config.llama_foundation = True

        # Validate fairness
        fairness_report = comparator._validate_fairness_configuration(comparison_config)

        assert fairness_report["fairness_score"] == 1.0
        assert fairness_report["identical_pipeline"] is True
        assert fairness_report["same_prompts"] is True
        assert fairness_report["same_evaluation_metrics"] is True
        assert fairness_report["bias_risk"] == "low"
        assert fairness_report["fairness_status"] == "fair"

    def test_model_agnostic_business_logic(self, fairness_test_config, mock_image_path):
        """Test that business logic is model-agnostic."""
        business_logic_calls = {}

        def track_business_logic(model_type, operation, *args, **kwargs):
            if model_type not in business_logic_calls:
                business_logic_calls[model_type] = []
            business_logic_calls[model_type].append(
                {"operation": operation, "args": args, "kwargs": kwargs}
            )

        # Test business logic independence for both models
        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            fairness_test_config.model_type = model_type
            model_name = model_type.value

            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ):
                with patch(
                    "vision_processor.classification.DocumentClassifier"
                ) as mock_classifier:
                    with patch(
                        "vision_processor.extraction.hybrid_extraction_manager.AWKExtractor"
                    ) as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager"):
                            with patch(
                                "vision_processor.extraction.hybrid_extraction_manager.ATOComplianceHandler"
                            ) as mock_ato:
                                # Track business logic calls
                                def mock_classify_tracker(_x, m=model_name):
                                    track_business_logic(m, "classification", _x)
                                    return (
                                        DocumentType.BUSINESS_RECEIPT,
                                        0.85,
                                        ["evidence"],
                                    )

                                mock_classifier.return_value.classify_with_evidence.side_effect = mock_classify_tracker

                                def mock_awk_tracker(text, doc_type, m=model_name):
                                    track_business_logic(
                                        m, "awk_extraction", text, doc_type
                                    )
                                    return {"awk_field": "value"}

                                mock_awk.return_value.extract.side_effect = (
                                    mock_awk_tracker
                                )
                                mock_awk.return_value.ensure_initialized = MagicMock()

                                def mock_ato_tracker(fields, doc_type, m=model_name):
                                    track_business_logic(
                                        m, "ato_compliance", fields, doc_type
                                    )
                                    return Mock(
                                        compliance_score=0.90,
                                        passed=True,
                                        violations=[],
                                        warnings=[],
                                    )

                                mock_ato.return_value.assess_compliance.side_effect = (
                                    mock_ato_tracker
                                )
                                mock_ato.return_value.ensure_initialized = MagicMock()

                                manager = UnifiedExtractionManager(fairness_test_config)
                                manager.process_document(mock_image_path)

        # Verify business logic is model-agnostic
        assert "internvl3" in business_logic_calls
        assert "llama32_vision" in business_logic_calls

        internvl_operations = [
            call["operation"] for call in business_logic_calls["internvl3"]
        ]
        llama_operations = [
            call["operation"] for call in business_logic_calls["llama32_vision"]
        ]

        # Same business logic operations should be called
        assert internvl_operations == llama_operations
        assert "classification" in internvl_operations
        assert "awk_extraction" in internvl_operations
        assert "ato_compliance" in internvl_operations
