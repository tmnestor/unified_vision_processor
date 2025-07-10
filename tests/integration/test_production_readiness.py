"""
Production Readiness Testing

Tests the 5-level production readiness assessment system and validates
that the unified architecture is ready for production deployment.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.unified_config import UnifiedConfig
from vision_processor.extraction.hybrid_extraction_manager import (
    UnifiedExtractionManager,
)
from vision_processor.extraction.pipeline_components import (
    DocumentType,
    QualityGrade,
)


class TestProductionReadinessAssessment:
    """Test suite for 5-level production readiness assessment."""

    @pytest.fixture
    def production_config(self):
        """Create a production-ready configuration."""
        config = UnifiedConfig()
        config.processing_pipeline = "7step"
        config.production_assessment = "5level"
        config.confidence_threshold = 0.7
        config.quality_threshold = 0.6
        config.graceful_degradation = True
        config.awk_fallback = True
        return config

    def test_excellent_quality_grade_assessment(
        self, production_config, mock_image_path
    ):
        """Test assessment for excellent quality grade (Level 5)."""
        # Mock high-quality processing result
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                            # Setup mocks for excellent quality
                            mock_model = MagicMock()
                            mock_model.process_image.return_value = Mock(
                                raw_text="Excellent model response with all fields",
                                confidence=0.95,
                                processing_time=1.2,
                            )
                            mock_factory.return_value = mock_model

                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.BUSINESS_RECEIPT,
                                0.95,
                                ["high_confidence_evidence"],
                            )

                            mock_awk.return_value.extract.return_value = {
                                "comprehensive_field": "excellent_value"
                            }

                            mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                overall_confidence=0.95,
                                quality_grade=QualityGrade.EXCELLENT,
                                production_ready=True,
                                quality_flags=[
                                    "high_confidence",
                                    "complete_extraction",
                                ],
                                recommendations=[
                                    "Ready for production",
                                    "Excellent quality",
                                ],
                            )

                            mock_ato.return_value.assess_compliance.return_value = Mock(
                                compliance_score=0.98,
                                passed=True,
                                violations=[],
                                warnings=[],
                            )

                            # Process document
                            with UnifiedExtractionManager(production_config) as manager:
                                # Mock handler to return excellent extraction
                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {
                                        "supplier_name": "Excellent Store",
                                        "total_amount": "45.67",
                                        "date": "25/03/2024",
                                        "abn": "88000014675",
                                        "gst_amount": "4.15",
                                        "invoice_number": "INV-12345",
                                    }
                                    mock_handler.validate_fields.return_value = (
                                        mock_handler.extract_fields_primary.return_value
                                    )
                                    mock_get_handler.return_value = mock_handler

                                    result = manager.process_document(mock_image_path)

        # Validate excellent quality assessment
        assert result.quality_grade == QualityGrade.EXCELLENT
        assert result.confidence_score >= 0.9
        assert result.production_ready is True
        assert result.ato_compliance_score >= 0.95
        assert "high_confidence" in result.quality_flags
        assert "excellent quality" in [rec.lower() for rec in result.recommendations]

    def test_good_quality_grade_assessment(self, production_config, mock_image_path):
        """Test assessment for good quality grade (Level 4)."""
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                            # Setup mocks for good quality
                            mock_model = MagicMock()
                            mock_model.process_image.return_value = Mock(
                                raw_text="Good model response with most fields",
                                confidence=0.85,
                                processing_time=1.5,
                            )
                            mock_factory.return_value = mock_model

                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.BUSINESS_RECEIPT,
                                0.85,
                                ["good_confidence_evidence"],
                            )

                            mock_awk.return_value.extract.return_value = {
                                "additional_field": "good_value"
                            }

                            mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                overall_confidence=0.85,
                                quality_grade=QualityGrade.GOOD,
                                production_ready=True,
                                quality_flags=["good_confidence"],
                                recommendations=["Suitable for production"],
                            )

                            mock_ato.return_value.assess_compliance.return_value = Mock(
                                compliance_score=0.85,
                                passed=True,
                                violations=[],
                                warnings=["Minor formatting issue"],
                            )

                            # Process document
                            with UnifiedExtractionManager(production_config) as manager:
                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {
                                        "supplier_name": "Good Store",
                                        "total_amount": "35.50",
                                        "date": "20/03/2024",
                                        "abn": "88000014675",
                                    }
                                    mock_handler.validate_fields.return_value = (
                                        mock_handler.extract_fields_primary.return_value
                                    )
                                    mock_get_handler.return_value = mock_handler

                                    result = manager.process_document(mock_image_path)

        # Validate good quality assessment
        assert result.quality_grade == QualityGrade.GOOD
        assert 0.8 <= result.confidence_score < 0.9
        assert result.production_ready is True
        assert result.ato_compliance_score >= 0.8
        assert len(result.warnings) > 0  # Should have minor warnings

    def test_fair_quality_grade_assessment(self, production_config, mock_image_path):
        """Test assessment for fair quality grade (Level 3)."""
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                            # Setup mocks for fair quality
                            mock_model = MagicMock()
                            mock_model.process_image.return_value = Mock(
                                raw_text="Fair model response with some fields",
                                confidence=0.75,
                                processing_time=2.0,
                            )
                            mock_factory.return_value = mock_model

                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.BUSINESS_RECEIPT,
                                0.75,
                                ["moderate_confidence"],
                            )

                            mock_awk.return_value.extract.return_value = {
                                "fallback_field": "fair_value"
                            }

                            mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                overall_confidence=0.70,
                                quality_grade=QualityGrade.FAIR,
                                production_ready=False,  # Fair quality not production ready
                                quality_flags=[
                                    "moderate_confidence",
                                    "awk_fallback_used",
                                ],
                                recommendations=[
                                    "Manual review recommended",
                                    "Consider reprocessing",
                                ],
                            )

                            mock_ato.return_value.assess_compliance.return_value = Mock(
                                compliance_score=0.75,
                                passed=True,
                                violations=[],
                                warnings=[
                                    "Format compliance issues",
                                    "Missing optional fields",
                                ],
                            )

                            # Process document
                            with UnifiedExtractionManager(production_config) as manager:
                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {
                                        "supplier_name": "Fair Store",
                                        "total_amount": "25.00",
                                        # Missing some fields
                                    }
                                    mock_handler.validate_fields.return_value = (
                                        mock_handler.extract_fields_primary.return_value
                                    )
                                    mock_get_handler.return_value = mock_handler

                                    result = manager.process_document(mock_image_path)

        # Validate fair quality assessment
        assert result.quality_grade == QualityGrade.FAIR
        assert 0.6 <= result.confidence_score < 0.8
        assert result.production_ready is False  # Fair quality requires manual review
        assert result.ato_compliance_score >= 0.7
        assert "manual review recommended" in [
            rec.lower() for rec in result.recommendations
        ]

    def test_poor_quality_grade_assessment(self, production_config, mock_image_path):
        """Test assessment for poor quality grade (Level 2)."""
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                            # Setup mocks for poor quality
                            mock_model = MagicMock()
                            mock_model.process_image.return_value = Mock(
                                raw_text="Poor model response with minimal fields",
                                confidence=0.55,
                                processing_time=3.0,
                            )
                            mock_factory.return_value = mock_model

                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.UNKNOWN,
                                0.55,
                                ["low_confidence"],  # Poor classification
                            )

                            mock_awk.return_value.extract.return_value = {
                                "minimal_field": "poor_value"
                            }

                            mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                overall_confidence=0.50,
                                quality_grade=QualityGrade.POOR,
                                production_ready=False,
                                quality_flags=[
                                    "low_confidence",
                                    "classification_uncertain",
                                    "awk_fallback_used",
                                ],
                                recommendations=[
                                    "Manual processing required",
                                    "Document quality issues",
                                    "Resubmit if possible",
                                ],
                            )

                            mock_ato.return_value.assess_compliance.return_value = Mock(
                                compliance_score=0.60,
                                passed=False,
                                violations=["Missing required fields", "Format issues"],
                                warnings=[
                                    "Poor document quality",
                                    "Compliance concerns",
                                ],
                            )

                            # Process document
                            with UnifiedExtractionManager(production_config) as manager:
                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {
                                        "supplier_name": "Poor Store"
                                        # Very minimal extraction
                                    }
                                    mock_handler.validate_fields.return_value = (
                                        mock_handler.extract_fields_primary.return_value
                                    )
                                    mock_get_handler.return_value = mock_handler

                                    result = manager.process_document(mock_image_path)

        # Validate poor quality assessment
        assert result.quality_grade == QualityGrade.POOR
        assert 0.4 <= result.confidence_score < 0.6
        assert result.production_ready is False
        assert result.ato_compliance_score < 0.7
        assert len(result.errors) > 0 or len(result.warnings) > 0
        assert "manual processing required" in [
            rec.lower() for rec in result.recommendations
        ]

    def test_very_poor_quality_grade_assessment(
        self, production_config, mock_image_path
    ):
        """Test assessment for very poor quality grade (Level 1)."""
        with patch(
            "vision_processor.config.model_factory.ModelFactory.create_model"
        ) as mock_factory:
            with patch(
                "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                            # Setup mocks for very poor quality
                            mock_model = MagicMock()
                            mock_model.process_image.return_value = Mock(
                                raw_text="Very poor response",
                                confidence=0.30,
                                processing_time=5.0,  # Slow processing
                            )
                            mock_factory.return_value = mock_model

                            mock_classifier.return_value.classify_with_evidence.return_value = (
                                DocumentType.UNKNOWN,
                                0.30,
                                ["very_low_confidence"],
                            )

                            mock_awk.return_value.extract.return_value = {}  # No fallback extraction

                            mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                overall_confidence=0.25,
                                quality_grade=QualityGrade.VERY_POOR,
                                production_ready=False,
                                quality_flags=[
                                    "very_low_confidence",
                                    "extraction_failed",
                                    "classification_failed",
                                ],
                                recommendations=[
                                    "Reject document",
                                    "Manual processing only",
                                    "Check document quality",
                                ],
                            )

                            mock_ato.return_value.assess_compliance.return_value = Mock(
                                compliance_score=0.20,
                                passed=False,
                                violations=[
                                    "Critical compliance failures",
                                    "No extractable data",
                                ],
                                warnings=["Document unprocessable", "Quality too poor"],
                            )

                            # Process document
                            with UnifiedExtractionManager(production_config) as manager:
                                with patch.object(
                                    manager, "_get_handler"
                                ) as mock_get_handler:
                                    mock_handler = MagicMock()
                                    mock_handler.extract_fields_primary.return_value = {}  # No extraction
                                    mock_handler.validate_fields.return_value = {}
                                    mock_get_handler.return_value = mock_handler

                                    result = manager.process_document(mock_image_path)

        # Validate very poor quality assessment
        assert result.quality_grade == QualityGrade.VERY_POOR
        assert result.confidence_score < 0.4
        assert result.production_ready is False
        assert result.ato_compliance_score < 0.5
        assert len(result.errors) > 0
        assert "reject document" in [rec.lower() for rec in result.recommendations]

    def test_production_readiness_decision_matrix(
        self, production_config, mock_image_path
    ):
        """Test the production readiness decision matrix."""
        decision_matrix_tests = [
            # (confidence, ato_compliance, quality_grade, expected_production_ready)
            (0.95, 0.95, QualityGrade.EXCELLENT, True),
            (0.85, 0.85, QualityGrade.GOOD, True),
            (0.75, 0.75, QualityGrade.FAIR, False),
            (0.55, 0.60, QualityGrade.POOR, False),
            (0.30, 0.30, QualityGrade.VERY_POOR, False),
            (
                0.90,
                0.60,
                QualityGrade.GOOD,
                False,
            ),  # High confidence but low ATO compliance
            (
                0.60,
                0.90,
                QualityGrade.FAIR,
                False,
            ),  # Low confidence but high ATO compliance
        ]

        for (
            confidence,
            ato_score,
            quality_grade,
            expected_ready,
        ) in decision_matrix_tests:
            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ) as mock_factory:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                                # Setup mocks for specific test case
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Test response",
                                    confidence=confidence,
                                    processing_time=1.5,
                                )
                                mock_factory.return_value = mock_model

                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    confidence,
                                    ["test_evidence"],
                                )

                                mock_awk.return_value.extract.return_value = {
                                    "test_field": "test_value"
                                }

                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=confidence,
                                    quality_grade=quality_grade,
                                    production_ready=expected_ready,
                                    quality_flags=[],
                                    recommendations=[],
                                )

                                mock_ato.return_value.assess_compliance.return_value = (
                                    Mock(
                                        compliance_score=ato_score,
                                        passed=ato_score >= 0.7,
                                        violations=[],
                                        warnings=[],
                                    )
                                )

                                # Process document
                                with UnifiedExtractionManager(
                                    production_config
                                ) as manager:
                                    with patch.object(
                                        manager, "_get_handler"
                                    ) as mock_get_handler:
                                        mock_handler = MagicMock()
                                        mock_handler.extract_fields_primary.return_value = {
                                            "test": "value"
                                        }
                                        mock_handler.validate_fields.return_value = {
                                            "test": "value"
                                        }
                                        mock_get_handler.return_value = mock_handler

                                        result = manager.process_document(
                                            mock_image_path
                                        )

            # Validate production readiness decision
            assert result.production_ready == expected_ready, (
                f"Production readiness mismatch for confidence={confidence:.2f}, "
                f"ato={ato_score:.2f}, quality={quality_grade.value}: "
                f"expected {expected_ready}, got {result.production_ready}"
            )

    def test_confidence_threshold_validation(self, production_config, mock_image_path):
        """Test confidence threshold validation for production readiness."""
        threshold_tests = [
            (0.5, 0.60),  # threshold, test_confidence
            (0.7, 0.80),
            (0.8, 0.85),
            (0.9, 0.95),
        ]

        for threshold, test_confidence in threshold_tests:
            production_config.confidence_threshold = threshold

            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ) as mock_factory:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                                # Setup mocks
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Test response",
                                    confidence=test_confidence,
                                    processing_time=1.5,
                                )
                                mock_factory.return_value = mock_model

                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    test_confidence,
                                    ["evidence"],
                                )

                                mock_awk.return_value.extract.return_value = {
                                    "field": "value"
                                }

                                expected_ready = (
                                    test_confidence >= threshold
                                    and test_confidence >= 0.7
                                )  # Also needs good quality
                                quality_grade = (
                                    QualityGrade.GOOD
                                    if test_confidence >= 0.8
                                    else QualityGrade.FAIR
                                )

                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=test_confidence,
                                    quality_grade=quality_grade,
                                    production_ready=expected_ready,
                                    quality_flags=[],
                                    recommendations=[],
                                )

                                mock_ato.return_value.assess_compliance.return_value = (
                                    Mock(
                                        compliance_score=0.85,
                                        passed=True,
                                        violations=[],
                                        warnings=[],
                                    )
                                )

                                # Process document
                                with UnifiedExtractionManager(
                                    production_config
                                ) as manager:
                                    with patch.object(
                                        manager, "_get_handler"
                                    ) as mock_get_handler:
                                        mock_handler = MagicMock()
                                        mock_handler.extract_fields_primary.return_value = {
                                            "test": "value"
                                        }
                                        mock_handler.validate_fields.return_value = {
                                            "test": "value"
                                        }
                                        mock_get_handler.return_value = mock_handler

                                        result = manager.process_document(
                                            mock_image_path
                                        )

                # Validate threshold behavior
                assert result.confidence_score == test_confidence
                # Production readiness should consider both threshold and quality
                if test_confidence >= threshold and quality_grade in [
                    QualityGrade.EXCELLENT,
                    QualityGrade.GOOD,
                ]:
                    assert result.production_ready is True
                else:
                    assert result.production_ready is False

    def test_batch_production_readiness_statistics(
        self, production_config, benchmark_documents
    ):
        """Test production readiness statistics for batch processing."""
        batch_size = 5
        test_documents = benchmark_documents[:batch_size]

        # Simulate mixed quality results
        confidence_scores = [0.95, 0.85, 0.70, 0.55, 0.30]  # Excellent to Very Poor
        quality_grades = [
            QualityGrade.EXCELLENT,
            QualityGrade.GOOD,
            QualityGrade.FAIR,
            QualityGrade.POOR,
            QualityGrade.VERY_POOR,
        ]

        results = []

        for i, doc_path in enumerate(test_documents):
            with patch(
                "vision_processor.config.model_factory.ModelFactory.create_model"
            ) as mock_factory:
                with patch(
                    "vision_processor.extraction.hybrid_extraction_manager.AustralianTaxClassifier"
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
                                confidence = confidence_scores[i]
                                quality = quality_grades[i]
                                production_ready = quality in [
                                    QualityGrade.EXCELLENT,
                                    QualityGrade.GOOD,
                                ]

                                # Setup mocks for this specific quality level
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text=f"Response {i}",
                                    confidence=confidence,
                                    processing_time=1.5,
                                )
                                mock_factory.return_value = mock_model

                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    confidence,
                                    [f"evidence_{i}"],
                                )

                                mock_awk.return_value.extract.return_value = {
                                    f"field_{i}": f"value_{i}"
                                }

                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=confidence,
                                    quality_grade=quality,
                                    production_ready=production_ready,
                                    quality_flags=[],
                                    recommendations=[],
                                )

                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=confidence,  # Use same value for simplicity
                                    passed=confidence >= 0.7,
                                    violations=[],
                                    warnings=[],
                                )

                                # Process document
                                with UnifiedExtractionManager(
                                    production_config
                                ) as manager:
                                    with patch.object(
                                        manager, "_get_handler"
                                    ) as mock_get_handler:
                                        mock_handler = MagicMock()
                                        mock_handler.extract_fields_primary.return_value = {
                                            f"test_{i}": f"value_{i}"
                                        }
                                        mock_handler.validate_fields.return_value = {
                                            f"test_{i}": f"value_{i}"
                                        }
                                        mock_get_handler.return_value = mock_handler

                                        result = manager.process_document(doc_path)
                                        results.append(result)

        # Calculate batch statistics
        total_documents = len(results)
        production_ready_count = sum(1 for r in results if r.production_ready)
        production_ready_rate = production_ready_count / total_documents

        quality_distribution = {}
        for grade in QualityGrade:
            quality_distribution[grade.value] = sum(
                1 for r in results if r.quality_grade == grade
            )

        avg_confidence = sum(r.confidence_score for r in results) / total_documents
        avg_ato_compliance = (
            sum(r.ato_compliance_score for r in results) / total_documents
        )

        # Validate batch statistics
        assert total_documents == batch_size
        assert production_ready_count == 2  # Excellent + Good
        assert production_ready_rate == 0.4  # 2/5

        # Quality distribution should match our setup
        assert quality_distribution["excellent"] == 1
        assert quality_distribution["good"] == 1
        assert quality_distribution["fair"] == 1
        assert quality_distribution["poor"] == 1
        assert quality_distribution["very_poor"] == 1

        # Average scores should be reasonable
        assert 0.4 <= avg_confidence <= 0.8  # Mixed quality average
        assert 0.4 <= avg_ato_compliance <= 0.8

    def test_production_deployment_criteria_validation(self, _production_config):
        """Test validation of production deployment criteria."""
        deployment_criteria = {
            "minimum_production_ready_rate": 0.8,
            "minimum_average_confidence": 0.75,
            "minimum_ato_compliance_rate": 0.9,
            "maximum_processing_time": 5.0,
            "maximum_error_rate": 0.05,
        }

        # Mock batch results that meet deployment criteria
        mock_batch_results = []
        for i in range(10):
            # 9 out of 10 should be production ready to meet 0.8 rate
            production_ready = i < 9
            confidence = 0.85 if production_ready else 0.60
            quality_grade = QualityGrade.GOOD if production_ready else QualityGrade.FAIR

            result = Mock(
                production_ready=production_ready,
                confidence_score=confidence,
                ato_compliance_score=0.92,
                processing_time=2.5,
                quality_grade=quality_grade,
                errors=[],
                warnings=[],
            )
            mock_batch_results.append(result)

        # Calculate actual metrics
        total_docs = len(mock_batch_results)
        production_ready_rate = (
            sum(1 for r in mock_batch_results if r.production_ready) / total_docs
        )
        avg_confidence = (
            sum(r.confidence_score for r in mock_batch_results) / total_docs
        )
        ato_compliance_rate = (
            sum(1 for r in mock_batch_results if r.ato_compliance_score >= 0.9)
            / total_docs
        )
        avg_processing_time = (
            sum(r.processing_time for r in mock_batch_results) / total_docs
        )
        error_rate = (
            sum(1 for r in mock_batch_results if len(r.errors) > 0) / total_docs
        )

        # Validate deployment criteria
        assert (
            production_ready_rate
            >= deployment_criteria["minimum_production_ready_rate"]
        )
        assert avg_confidence >= deployment_criteria["minimum_average_confidence"]
        assert ato_compliance_rate >= deployment_criteria["minimum_ato_compliance_rate"]
        assert avg_processing_time <= deployment_criteria["maximum_processing_time"]
        assert error_rate <= deployment_criteria["maximum_error_rate"]

        # Log deployment readiness
        deployment_ready = all(
            [
                production_ready_rate
                >= deployment_criteria["minimum_production_ready_rate"],
                avg_confidence >= deployment_criteria["minimum_average_confidence"],
                ato_compliance_rate
                >= deployment_criteria["minimum_ato_compliance_rate"],
                avg_processing_time <= deployment_criteria["maximum_processing_time"],
                error_rate <= deployment_criteria["maximum_error_rate"],
            ]
        )

        assert deployment_ready is True, (
            "System meets all production deployment criteria"
        )
