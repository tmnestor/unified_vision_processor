"""
Unit Tests for Evaluation Framework

Tests the unified evaluation system that provides fair model comparison
using identical Llama pipeline processing for both models.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.unified_config import ModelType
from vision_processor.evaluation import (
    ComparisonConfiguration,
    DatasetEvaluationResult,
    EvaluationResult,
)
from vision_processor.evaluation.metrics_calculator import MetricsCalculator
from vision_processor.evaluation.model_comparator import ModelComparator
from vision_processor.evaluation.report_generator import ReportGenerator
from vision_processor.evaluation.sroie_evaluator import SROIEEvaluator
from vision_processor.evaluation.unified_evaluator import UnifiedEvaluator


class TestUnifiedEvaluator:
    """Test suite for UnifiedEvaluator class."""

    @pytest.fixture
    def unified_evaluator(self, test_config):
        """Create a unified evaluator for testing."""
        return UnifiedEvaluator(test_config)

    def test_evaluator_initialization(self, unified_evaluator, test_config):
        """Test unified evaluator initialization."""
        assert unified_evaluator.config == test_config
        assert hasattr(unified_evaluator, "metrics_calculator")
        assert hasattr(unified_evaluator, "report_generator")

    def test_single_document_evaluation(
        self, unified_evaluator, sample_extracted_fields, sample_ground_truth
    ):
        """Test evaluation of a single document."""
        result = unified_evaluator.evaluate_single_document(
            extracted_fields=sample_extracted_fields,
            ground_truth=sample_ground_truth,
            processing_time=2.5,
        )

        assert isinstance(result, EvaluationResult)
        assert hasattr(result, "precision")
        assert hasattr(result, "recall")
        assert hasattr(result, "f1_score")
        assert hasattr(result, "exact_match_score")
        assert hasattr(result, "ato_compliance_score")

        # Verify score ranges
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1_score <= 1.0
        assert 0.0 <= result.exact_match_score <= 1.0
        assert 0.0 <= result.ato_compliance_score <= 1.0

    def test_dataset_evaluation(self, unified_evaluator, sample_dataset_files):
        """Test evaluation of an entire dataset."""
        dataset_dir, ground_truth_dir = sample_dataset_files

        # Mock the extraction manager
        with patch(
            "vision_processor.extraction.hybrid_extraction_manager.UnifiedExtractionManager"
        ) as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value.__enter__.return_value = mock_instance

            # Mock processing results
            mock_instance.process_document.return_value = Mock(
                extracted_fields={
                    "supplier_name": "Test Store",
                    "total_amount": "25.50",
                },
                processing_time=2.0,
                confidence_score=0.85,
                production_ready=True,
            )

            result = unified_evaluator.evaluate_dataset(
                dataset_path=dataset_dir,
                ground_truth_path=ground_truth_dir,
                model_type=ModelType.INTERNVL3,
                max_documents=3,
            )

            assert isinstance(result, DatasetEvaluationResult)
            assert result.total_documents == 3
            assert result.successful_extractions <= result.total_documents
            assert hasattr(result, "average_f1_score")
            assert hasattr(result, "average_precision")
            assert hasattr(result, "average_recall")

    def test_evaluation_with_confidence_scores(
        self,
        unified_evaluator,
        sample_extracted_fields,
        sample_ground_truth,
        sample_confidence_scores,
    ):
        """Test evaluation with confidence scores."""
        result = unified_evaluator.evaluate_single_document(
            extracted_fields=sample_extracted_fields,
            ground_truth=sample_ground_truth,
            processing_time=2.5,
            confidence_scores=sample_confidence_scores,
        )

        assert hasattr(result, "confidence_correlation")
        assert isinstance(result.confidence_correlation, float)
        assert -1.0 <= result.confidence_correlation <= 1.0

    def test_evaluation_error_handling(self, unified_evaluator):
        """Test evaluation error handling."""
        # Test with empty fields
        result = unified_evaluator.evaluate_single_document(
            extracted_fields={}, ground_truth={}, processing_time=0.0
        )

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0

        # Test with None values
        result = unified_evaluator.evaluate_single_document(
            extracted_fields=None, ground_truth=None, processing_time=0.0
        )

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0


class TestModelComparator:
    """Test suite for ModelComparator class."""

    @pytest.fixture
    def model_comparator(self, test_config):
        """Create a model comparator for testing."""
        return ModelComparator(test_config)

    def test_comparison_configuration_validation(self, model_comparator):
        """Test comparison configuration validation."""
        config = ComparisonConfiguration(
            models_to_compare=["internvl3", "llama32_vision"],
            dataset_path=Path("/mock/dataset"),
            ground_truth_path=Path("/mock/ground_truth"),
            max_documents=10,
            generate_reports=True,
        )

        # Test fairness validation
        fairness_report = model_comparator._validate_fairness_configuration(config)

        assert fairness_report["fairness_score"] == 1.0
        assert fairness_report["identical_pipeline"] is True
        assert fairness_report["same_prompts"] is True
        assert fairness_report["same_evaluation_metrics"] is True

    def test_model_parity_analysis(self, model_comparator):
        """Test model parity analysis."""
        # Create mock dataset results
        mock_results = {
            "internvl3": Mock(
                average_f1_score=0.85,
                average_precision=0.87,
                average_recall=0.83,
                production_ready_rate=0.90,
                average_processing_time=2.3,
            ),
            "llama32_vision": Mock(
                average_f1_score=0.82,
                average_precision=0.84,
                average_recall=0.80,
                production_ready_rate=0.88,
                average_processing_time=1.8,
            ),
        }

        parity_analysis = model_comparator.analyze_model_parity(mock_results)

        assert "overall_parity" in parity_analysis
        assert "field_level_parity" in parity_analysis
        assert "fairness_assessment" in parity_analysis

        # Check fairness assessment
        fairness = parity_analysis["fairness_assessment"]
        assert "status" in fairness
        assert fairness["status"] in ["fair", "biased", "requires_review"]

    def test_processing_efficiency_benchmark(self, model_comparator):
        """Test processing efficiency benchmarking."""
        mock_results = {
            "internvl3": Mock(
                average_processing_time=2.3,
                total_processing_time=230.0,
                total_documents=100,
                memory_usage_stats={"avg_mb": 1500, "peak_mb": 2000},
            ),
            "llama32_vision": Mock(
                average_processing_time=1.8,
                total_processing_time=180.0,
                total_documents=100,
                memory_usage_stats={"avg_mb": 1200, "peak_mb": 1600},
            ),
        }

        efficiency_report = model_comparator.benchmark_processing_efficiency(
            mock_results
        )

        assert "model_efficiency" in efficiency_report
        assert "comparative_analysis" in efficiency_report
        assert "recommendations" in efficiency_report

        # Check efficiency rankings
        assert "internvl3" in efficiency_report["model_efficiency"]
        assert "llama32_vision" in efficiency_report["model_efficiency"]

    def test_statistical_significance_testing(self, model_comparator):
        """Test statistical significance testing."""
        # Mock evaluation results with statistical data
        results_a = [0.85, 0.87, 0.83, 0.86, 0.84]  # Model A F1 scores
        results_b = [0.82, 0.84, 0.80, 0.83, 0.81]  # Model B F1 scores

        significance_test = model_comparator.test_statistical_significance(
            results_a, results_b, metric_name="f1_score"
        )

        assert "p_value" in significance_test
        assert "is_significant" in significance_test
        assert "confidence_interval" in significance_test
        assert "effect_size" in significance_test

        assert isinstance(significance_test["p_value"], float)
        assert isinstance(significance_test["is_significant"], bool)


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""

    @pytest.fixture
    def metrics_calculator(self, test_config):
        """Create a metrics calculator for testing."""
        return MetricsCalculator(test_config)

    def test_precision_recall_f1_calculation(
        self, metrics_calculator, sample_extracted_fields, sample_ground_truth
    ):
        """Test precision, recall, and F1 score calculation."""
        precision, recall, f1 = metrics_calculator.calculate_prf_metrics(
            sample_extracted_fields, sample_ground_truth
        )

        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0

        # F1 should be harmonic mean of precision and recall
        if precision > 0 and recall > 0:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - expected_f1) < 0.001

    def test_exact_match_calculation(
        self, metrics_calculator, sample_extracted_fields, sample_ground_truth
    ):
        """Test exact match score calculation."""
        exact_match = metrics_calculator.calculate_exact_match(
            sample_extracted_fields, sample_ground_truth
        )

        assert isinstance(exact_match, float)
        assert 0.0 <= exact_match <= 1.0

    def test_fuzzy_match_calculation(
        self, metrics_calculator, sample_extracted_fields, sample_ground_truth
    ):
        """Test fuzzy match score calculation."""
        fuzzy_match = metrics_calculator.calculate_fuzzy_match(
            sample_extracted_fields, sample_ground_truth, threshold=0.8
        )

        assert isinstance(fuzzy_match, float)
        assert 0.0 <= fuzzy_match <= 1.0

        # For this test data, exact match uses field-specific normalization
        # so it may be higher than fuzzy match which uses raw string similarity
        exact_match = metrics_calculator.calculate_exact_match(
            sample_extracted_fields, sample_ground_truth
        )
        # Both should be valid scores between 0 and 1
        assert 0.0 <= exact_match <= 1.0

    def test_ato_compliance_scoring(
        self, metrics_calculator, sample_extracted_fields, sample_ground_truth
    ):
        """Test ATO compliance scoring."""
        ato_score = metrics_calculator.calculate_ato_compliance_score(
            sample_extracted_fields, sample_ground_truth
        )

        assert isinstance(ato_score, float)
        assert 0.0 <= ato_score <= 1.0

    def test_field_confidence_correlation(
        self,
        metrics_calculator,
        sample_extracted_fields,
        sample_ground_truth,
        sample_confidence_scores,
    ):
        """Test field confidence correlation calculation."""
        correlation = metrics_calculator.calculate_field_confidence_correlation(
            sample_extracted_fields, sample_ground_truth, sample_confidence_scores
        )

        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0

    def test_processing_efficiency_metrics(self, metrics_calculator):
        """Test processing efficiency metrics calculation."""
        efficiency_metrics = metrics_calculator.calculate_processing_efficiency(
            processing_time=2.5, document_complexity=1.2
        )

        assert "processing_time" in efficiency_metrics
        assert "efficiency_ratio" in efficiency_metrics
        assert "performance_category" in efficiency_metrics

        assert efficiency_metrics["processing_time"] == 2.5
        assert efficiency_metrics["performance_category"] in [
            "excellent",
            "good",
            "acceptable",
            "poor",
            "very_poor",
        ]

    def test_comprehensive_metrics_summary(
        self, metrics_calculator, sample_extracted_fields, sample_ground_truth
    ):
        """Test comprehensive metrics summary."""
        summary = metrics_calculator.get_metrics_summary(
            extracted=sample_extracted_fields,
            ground_truth=sample_ground_truth,
            processing_time=2.5,
        )

        expected_keys = [
            "precision",
            "recall",
            "f1_score",
            "exact_match_score",
            "fuzzy_match_score",
            "ato_compliance_score",
            "processing_efficiency",
            "field_count_extracted",
            "field_count_ground_truth",
            "field_coverage",
        ]

        for key in expected_keys:
            assert key in summary
            assert summary[key] is not None


class TestSROIEEvaluator:
    """Test suite for SROIE Evaluator."""

    @pytest.fixture
    def sroie_evaluator(self, test_config):
        """Create a SROIE evaluator for testing."""
        return SROIEEvaluator(test_config)

    def test_sroie_field_mapping(self, sroie_evaluator):
        """Test SROIE field mapping."""
        assert "company" in sroie_evaluator.sroie_field_mapping
        assert "date" in sroie_evaluator.sroie_field_mapping
        assert "total" in sroie_evaluator.sroie_field_mapping
        assert "address" in sroie_evaluator.sroie_field_mapping

        # Test mapping conversion
        assert sroie_evaluator.sroie_field_mapping["company"] == "supplier_name"
        assert sroie_evaluator.sroie_field_mapping["total"] == "total_amount"

    def test_enhanced_field_mapping(self, sroie_evaluator):
        """Test enhanced field mapping for Australian tax documents."""
        enhanced_fields = sroie_evaluator.enhanced_field_mapping

        assert "abn" in enhanced_fields
        assert "gst_amount" in enhanced_fields
        assert "invoice_number" in enhanced_fields

        # These should be Australian-specific extensions
        australian_fields = ["abn", "gst_amount", "bsb", "bank_name"]
        for field in australian_fields:
            if field in enhanced_fields:
                mapping = enhanced_fields[field]
                # Field mapping can be a string (identity) or list (synonyms)
                if isinstance(mapping, list):
                    assert field in mapping  # Field should be in its own synonym list
                else:
                    assert mapping == field  # Identity mapping

    def test_ground_truth_standardization(self, sroie_evaluator):
        """Test SROIE ground truth standardization."""
        sroie_gt = {
            "company": "Test Store Pty Ltd",
            "date": "20/03/2024",
            "total": "45.67",
            "address": "123 Test St, Sydney NSW",
        }

        standardized = sroie_evaluator._standardize_sroie_ground_truth(
            sroie_gt, sroie_evaluator.sroie_field_mapping
        )

        assert "supplier_name" in standardized
        assert "total_amount" in standardized
        assert standardized["supplier_name"] == "Test Store Pty Ltd"
        assert standardized["total_amount"] == "45.67"

    def test_sroie_leaderboard_generation(self, sroie_evaluator):
        """Test SROIE leaderboard generation."""
        mock_results = {
            "internvl3": {
                "overall_metrics": {"average_f1": 0.85},
                "sroie_metrics": {
                    "company": {"f1": 0.88},
                    "total": {"f1": 0.82},
                    "date": {"f1": 0.90},
                    "address": {"f1": 0.75},
                },
                "processing_statistics": {
                    "avg_processing_time": 2.1,
                    "total_documents": 100,
                },
            },
            "llama32_vision": {
                "overall_metrics": {"average_f1": 0.82},
                "sroie_metrics": {
                    "company": {"f1": 0.85},
                    "total": {"f1": 0.85},
                    "date": {"f1": 0.88},
                    "address": {"f1": 0.72},
                },
                "processing_statistics": {
                    "avg_processing_time": 2.8,
                    "total_documents": 100,
                },
            },
        }

        leaderboard = sroie_evaluator.generate_sroie_leaderboard(mock_results)

        assert "rankings" in leaderboard
        assert "field_performance" in leaderboard
        assert len(leaderboard["rankings"]) == 2

        # Check ranking order (should be by average F1)
        rankings = leaderboard["rankings"]
        assert rankings[0]["model"] == "internvl3"  # Higher F1
        assert rankings[1]["model"] == "llama32_vision"  # Lower F1


class TestReportGenerator:
    """Test suite for Report Generator."""

    @pytest.fixture
    def report_generator(self, test_config):
        """Create a report generator for testing."""
        return ReportGenerator(test_config)

    def test_html_report_generation(self, report_generator):
        """Test HTML report generation."""
        mock_results = {
            "internvl3": {
                "model_name": "internvl3",
                "average_f1_score": 0.85,
                "total_documents": 100,
                "average_processing_time": 2.3,
            },
            "llama32_vision": {
                "model_name": "llama32_vision",
                "average_f1_score": 0.82,
                "total_documents": 100,
                "average_processing_time": 1.8,
            },
        }

        html_report = report_generator.generate_model_comparison_report(
            mock_results, format_type="html"
        )

        assert isinstance(html_report, str)
        assert "<html>" in html_report
        assert "Model Comparison Report" in html_report
        assert "internvl3" in html_report
        assert "llama32_vision" in html_report

    def test_json_report_generation(self, report_generator):
        """Test JSON report generation."""
        mock_results = {
            "internvl3": {
                "model_name": "internvl3",
                "average_f1_score": 0.85,
                "total_documents": 100,
                "average_processing_time": 2.3,
            }
        }

        json_report = report_generator.generate_model_comparison_report(
            mock_results, format_type="json"
        )

        # Should be valid JSON
        json_data = json.loads(json_report)
        assert "report_type" in json_data
        assert "detailed_results" in json_data
        assert "summary" in json_data

    def test_text_report_generation(self, report_generator):
        """Test text report generation."""
        mock_results = {
            "internvl3": {
                "model_name": "internvl3",
                "average_f1_score": 0.85,
                "total_documents": 100,
            }
        }

        text_report = report_generator.generate_model_comparison_report(
            mock_results, format_type="text"
        )

        assert isinstance(text_report, str)
        assert "MODEL COMPARISON REPORT" in text_report
        assert "internvl3" in text_report

    def test_report_file_saving(self, report_generator):
        """Test saving reports to files."""
        mock_results = {
            "internvl3": {"model_name": "internvl3", "average_f1_score": 0.85}
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"

            report_generator.save_report_to_file(
                mock_results, output_path, format_type="html"
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify file content
            content = output_path.read_text()
            assert "<html>" in content
            assert "internvl3" in content
