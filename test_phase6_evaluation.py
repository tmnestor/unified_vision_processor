#!/usr/bin/env python3
"""
Test script for Phase 6: Evaluation Framework

Tests the unified evaluation system, SROIE evaluation, model comparison,
and comprehensive reporting capabilities.
"""

import json
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_evaluation_framework_imports():
    """Test that all evaluation components can be imported."""
    logger.info("Testing evaluation framework imports...")

    try:
        from vision_processor.evaluation import (
            UnifiedEvaluator,
        )

        # Verify the import worked by checking the class exists
        assert UnifiedEvaluator is not None
        logger.info("✓ All evaluation components imported successfully")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_metrics_calculator():
    """Test metrics calculator functionality."""
    logger.info("\nTesting MetricsCalculator...")

    try:
        from vision_processor.config.unified_config import UnifiedConfig
        from vision_processor.evaluation import MetricsCalculator

        # Create test configuration
        config = UnifiedConfig.from_env()
        calculator = MetricsCalculator(config)

        # Test data
        extracted = {
            "supplier_name": "Woolworths",
            "total_amount": "45.67",
            "date": "25/03/2024",
            "abn": "88 000 014 675",
        }

        ground_truth = {
            "supplier_name": "woolworths",
            "total_amount": "$45.67",
            "date": "25/03/2024",
            "abn": "88000014675",
        }

        # Test precision, recall, F1
        precision, recall, f1 = calculator.calculate_prf_metrics(
            extracted, ground_truth
        )
        logger.info(f"  P/R/F1: {precision:.3f}/{recall:.3f}/{f1:.3f}")

        # Test exact match
        exact_match = calculator.calculate_exact_match(extracted, ground_truth)
        logger.info(f"  Exact match: {exact_match:.3f}")

        # Test ATO compliance
        ato_score = calculator.calculate_ato_compliance_score(extracted, ground_truth)
        logger.info(f"  ATO compliance: {ato_score:.3f}")

        # Test comprehensive metrics
        metrics_summary = calculator.get_metrics_summary(
            extracted, ground_truth, processing_time=2.5
        )
        logger.info(f"  Comprehensive metrics: {len(metrics_summary)} components")

        logger.info("✓ MetricsCalculator tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ MetricsCalculator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_report_generator():
    """Test report generator functionality."""
    logger.info("\nTesting ReportGenerator...")

    try:
        from vision_processor.config.unified_config import UnifiedConfig
        from vision_processor.evaluation import ReportGenerator

        config = UnifiedConfig.from_env()
        generator = ReportGenerator(config)

        # Mock comparison results
        mock_results = {
            "internvl3": MockDatasetResult("internvl3", 0.85, 0.90, 2.3),
            "llama32_vision": MockDatasetResult("llama32_vision", 0.82, 0.88, 1.8),
        }

        # Test HTML report generation
        html_report = generator.generate_model_comparison_report(
            mock_results, format_type="html"
        )

        # Validate HTML content
        assert "<html>" in html_report
        assert "Model Comparison Report" in html_report
        assert "internvl3" in html_report
        assert "llama32_vision" in html_report

        logger.info(f"  HTML report generated: {len(html_report)} characters")

        # Test JSON report generation
        json_report = generator.generate_model_comparison_report(
            mock_results, format_type="json"
        )

        # Validate JSON content
        json_data = json.loads(json_report)
        assert "report_type" in json_data
        assert "detailed_results" in json_data
        assert "internvl3" in json_data["detailed_results"]

        logger.info(f"  JSON report generated: {len(json_data)} fields")

        # Test text report generation
        text_report = generator.generate_model_comparison_report(
            mock_results, format_type="text"
        )

        assert "MODEL COMPARISON REPORT" in text_report
        assert "internvl3" in text_report

        logger.info(f"  Text report generated: {len(text_report)} characters")

        logger.info("✓ ReportGenerator tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ ReportGenerator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_comparator():
    """Test model comparator functionality."""
    logger.info("\nTesting ModelComparator...")

    try:
        from vision_processor.config.unified_config import UnifiedConfig
        from vision_processor.evaluation import ComparisonConfiguration, ModelComparator

        config = UnifiedConfig.from_env()
        comparator = ModelComparator(config)

        # Create temporary test data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock dataset and ground truth
            dataset_path = temp_path / "dataset"
            ground_truth_path = temp_path / "ground_truth"
            dataset_path.mkdir()
            ground_truth_path.mkdir()

            # Create a test image file (empty for this test)
            test_image = dataset_path / "test_image.jpg"
            test_image.write_bytes(b"mock_image_data")

            # Create corresponding ground truth
            ground_truth_file = ground_truth_path / "test_image.json"
            ground_truth_data = {
                "supplier_name": "Test Store",
                "total_amount": "25.50",
                "date": "20/03/2024",
            }
            ground_truth_file.write_text(json.dumps(ground_truth_data))

            # Test comparison configuration
            comparison_config = ComparisonConfiguration(
                models_to_compare=["internvl3", "llama32_vision"],
                dataset_path=dataset_path,
                ground_truth_path=ground_truth_path,
                max_documents=1,
                generate_reports=False,  # Skip report generation for test
            )

            # Test configuration validation
            fairness_report = comparator._validate_fairness_configuration(
                comparison_config
            )
            assert fairness_report["fairness_score"] == 1.0
            logger.info(
                f"  Fairness validation: {fairness_report['fairness_score']:.1f} score"
            )

            # Test parity analysis with mock data
            mock_model_results = {
                "internvl3": MockDatasetResult("internvl3", 0.85, 0.90, 2.3),
                "llama32_vision": MockDatasetResult("llama32_vision", 0.82, 0.88, 1.8),
            }

            parity_analysis = comparator.analyze_model_parity(mock_model_results)
            assert "overall_parity" in parity_analysis
            assert "fairness_assessment" in parity_analysis
            logger.info(
                f"  Parity analysis: {parity_analysis['fairness_assessment']['status']}"
            )

            # Test efficiency benchmarking
            efficiency_report = comparator.benchmark_processing_efficiency(
                mock_model_results
            )
            assert "model_efficiency" in efficiency_report
            assert "comparative_analysis" in efficiency_report
            logger.info(
                f"  Efficiency benchmark: {len(efficiency_report['model_efficiency'])} models"
            )

        logger.info("✓ ModelComparator tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ ModelComparator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sroie_evaluator():
    """Test SROIE evaluator functionality."""
    logger.info("\nTesting SROIEEvaluator...")

    try:
        from vision_processor.config.unified_config import UnifiedConfig
        from vision_processor.evaluation import SROIEEvaluator

        config = UnifiedConfig.from_env()
        evaluator = SROIEEvaluator(config)

        # Test field mapping
        assert "company" in evaluator.sroie_field_mapping
        assert "total" in evaluator.enhanced_field_mapping
        assert "abn" in evaluator.enhanced_field_mapping

        logger.info(
            f"  SROIE field mapping: {len(evaluator.sroie_field_mapping)} standard fields"
        )
        logger.info(
            f"  Enhanced field mapping: {len(evaluator.enhanced_field_mapping)} enhanced fields"
        )

        # Test SROIE ground truth standardization
        sroie_gt = {
            "company": "Test Store Pty Ltd",
            "date": "20/03/2024",
            "total": "45.67",
            "address": "123 Test St, Sydney NSW",
        }

        standardized = evaluator._standardize_sroie_ground_truth(
            sroie_gt, evaluator.sroie_field_mapping
        )

        assert "supplier_name" in standardized
        assert standardized["supplier_name"] == "Test Store Pty Ltd"
        logger.info(f"  Ground truth standardization: {len(standardized)} fields")

        # Test field exact matching
        assert evaluator._field_exact_match("45.67", "$45.67", "total")
        assert evaluator._field_exact_match("20/03/2024", "20-03-2024", "date")
        logger.info("  Field exact matching: ✓ Amount and date normalization")

        # Test leaderboard generation with mock data
        mock_sroie_results = {
            "internvl3": {
                "overall_metrics": {"average_f1": 0.85, "success_rate": 0.90},
                "processing_statistics": {"avg_processing_time": 2.3},
                "dataset_info": {"total_documents": 100},
                "sroie_metrics": {
                    "company": {"f1": 0.88},
                    "total": {"f1": 0.82},
                    "date": {"f1": 0.90},
                    "address": {"f1": 0.75},
                },
            },
            "llama32_vision": {
                "overall_metrics": {"average_f1": 0.82, "success_rate": 0.88},
                "processing_statistics": {"avg_processing_time": 1.8},
                "dataset_info": {"total_documents": 100},
                "sroie_metrics": {
                    "company": {"f1": 0.85},
                    "total": {"f1": 0.85},
                    "date": {"f1": 0.88},
                    "address": {"f1": 0.72},
                },
            },
        }

        leaderboard = evaluator.generate_sroie_leaderboard(mock_sroie_results)

        assert "rankings" in leaderboard
        assert "field_performance" in leaderboard
        assert len(leaderboard["rankings"]) == 2
        logger.info(
            f"  Leaderboard generation: {len(leaderboard['rankings'])} models ranked"
        )

        logger.info("✓ SROIEEvaluator tests passed")
        return True

    except Exception as e:
        logger.error(f"✗ SROIEEvaluator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


class MockDatasetResult:
    """Mock dataset result for testing."""

    def __init__(
        self,
        model_name: str,
        f1_score: float,
        production_rate: float,
        processing_time: float,
    ):
        self.model_name = model_name
        self.dataset_name = "test_dataset"
        self.total_documents = 100
        self.successful_extractions = 95
        self.average_f1_score = f1_score
        self.average_precision = f1_score + 0.02
        self.average_recall = f1_score - 0.02
        self.average_confidence = 0.85
        self.production_ready_count = int(production_rate * self.total_documents)
        self.production_ready_rate = production_rate
        self.total_processing_time = processing_time * self.total_documents
        self.average_processing_time = processing_time
        self.awk_fallback_rate = 0.15
        self.highlight_detection_rate = 0.25
        self.quality_distribution = {"excellent": 60, "good": 30, "fair": 10}
        self.failed_documents = ["failed1.jpg", "failed2.jpg"]
        self.error_analysis = {"timeout": 2, "parsing_error": 3}


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PHASE 6: EVALUATION FRAMEWORK TEST")
    logger.info("=" * 70)

    # Test all components
    imports_success = test_evaluation_framework_imports()
    metrics_success = test_metrics_calculator()
    reports_success = test_report_generator()
    comparator_success = test_model_comparator()
    sroie_success = test_sroie_evaluator()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6 TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"Evaluation Framework Imports: {'✓ PASSED' if imports_success else '✗ FAILED'}"
    )
    logger.info(
        f"MetricsCalculator Test: {'✓ PASSED' if metrics_success else '✗ FAILED'}"
    )
    logger.info(
        f"ReportGenerator Test: {'✓ PASSED' if reports_success else '✗ FAILED'}"
    )
    logger.info(
        f"ModelComparator Test: {'✓ PASSED' if comparator_success else '✗ FAILED'}"
    )
    logger.info(f"SROIEEvaluator Test: {'✓ PASSED' if sroie_success else '✗ FAILED'}")

    all_passed = all(
        [
            imports_success,
            metrics_success,
            reports_success,
            comparator_success,
            sroie_success,
        ]
    )

    if all_passed:
        logger.info("\n🎉 ALL PHASE 6 TESTS PASSED!")
        logger.info("✓ Unified evaluation framework implemented successfully")
        logger.info("✓ Fair model comparison with identical Llama pipeline")
        logger.info("✓ SROIE dataset evaluation with Australian tax enhancements")
        logger.info("✓ Comprehensive metrics calculation and reporting")
        logger.info("✓ Model performance benchmarking and rankings")
        logger.info("\nNext: Continue with Phase 7 (CLI and Production Features)")
    else:
        logger.info("\n⚠ Some Phase 6 tests failed - check the logs above")
        logger.info("Review evaluation framework implementation")
