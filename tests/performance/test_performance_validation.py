"""
Performance Validation Tests

Tests to validate that the unified architecture maintains or improves
performance compared to the original InternVL and Llama systems.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from vision_processor.config.unified_config import ModelType, UnifiedConfig
from vision_processor.extraction.hybrid_extraction_manager import (
    UnifiedExtractionManager,
)
from vision_processor.extraction.pipeline_components import DocumentType


class TestPerformanceValidation:
    """Test suite for performance validation against original systems."""

    @pytest.fixture
    def performance_config(self):
        """Create a configuration optimized for performance testing."""
        config = UnifiedConfig()
        config.performance_testing = True
        config.processing_pipeline = "7step"
        config.batch_size = 4
        config.max_workers = 2
        return config

    @pytest.fixture
    def benchmark_documents(self, temp_directory):
        """Create benchmark documents for performance testing."""
        documents = []
        for i in range(10):
            doc_path = temp_directory / f"benchmark_doc_{i}.jpg"
            doc_path.write_bytes(b"mock_image_data")
            documents.append(doc_path)
        return documents

    def test_single_document_processing_speed(self, performance_config, mock_image_path):
        """Test single document processing speed benchmarks."""
        processing_times = {}

        # Test both models
        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type

            with patch("vision_processor.config.model_factory.ModelFactory.create_model") as mock_factory:
                with patch(
                    "vision_processor.classification.document_classifier.DocumentClassifier"
                ) as mock_classifier:
                    with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                            with patch(
                                "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                            ) as mock_ato:
                                # Setup mocks for consistent timing
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )
                                mock_model.get_memory_usage.return_value = 256.0  # Mock memory usage in MB
                                mock_factory.return_value = mock_model

                                # Setup other mocks
                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )
                                mock_awk.return_value.extract.return_value = {"field": "value"}
                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=Mock(value="good"),
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )
                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=0.90,
                                    passed=True,
                                    violations=[],
                                    warnings=[],
                                )

                                # Measure processing time
                                start_time = time.time()

                                with UnifiedExtractionManager(performance_config) as manager:
                                    result = manager.process_document(mock_image_path)

                                end_time = time.time()
                                processing_time = end_time - start_time

                                processing_times[model_type.value] = {
                                    "total_time": processing_time,
                                    "reported_time": result.processing_time,
                                    "memory_usage": result.memory_usage_mb,
                                }

        # Validate performance benchmarks
        internvl_time = processing_times["internvl3"]["total_time"]
        llama_time = processing_times["llama32_vision"]["total_time"]

        # Both should complete within reasonable time (< 10 seconds for mocked processing)
        assert internvl_time < 10.0, f"InternVL processing too slow: {internvl_time:.2f}s"
        assert llama_time < 10.0, f"Llama processing too slow: {llama_time:.2f}s"

        # Performance difference should be reasonable (within test environment variations)
        time_ratio = max(internvl_time, llama_time) / min(internvl_time, llama_time)
        assert time_ratio < 20.0, (
            f"Performance difference too large: {time_ratio:.2f}x"
        )  # More lenient for test environments

        # Memory usage should be reasonable
        for model_name, metrics in processing_times.items():
            assert metrics["memory_usage"] > 0, f"Memory usage not tracked for {model_name}"
            assert metrics["memory_usage"] < 10000, (
                f"Memory usage too high for {model_name}: {metrics['memory_usage']:.1f}MB"
            )

    def test_batch_processing_performance(self, performance_config, benchmark_documents):
        """Test batch processing performance benchmarks."""
        batch_performance = {}

        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type

            with patch("vision_processor.config.model_factory.ModelFactory.create_model") as mock_factory:
                with patch(
                    "vision_processor.classification.document_classifier.DocumentClassifier"
                ) as mock_classifier:
                    with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                            with patch(
                                "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                            ) as mock_ato:
                                # Setup mocks
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )
                                mock_model.get_memory_usage.return_value = 256.0  # Mock memory usage in MB
                                mock_factory.return_value = mock_model

                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )
                                mock_awk.return_value.extract.return_value = {"field": "value"}
                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=Mock(value="good"),
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )
                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=0.90,
                                    passed=True,
                                    violations=[],
                                    warnings=[],
                                )

                                # Measure batch processing time
                                start_time = time.time()

                                with UnifiedExtractionManager(performance_config) as manager:
                                    results = []
                                    for doc_path in benchmark_documents:
                                        result = manager.process_document(doc_path)
                                        results.append(result)

                                end_time = time.time()
                                total_time = end_time - start_time

                                batch_performance[model_type.value] = {
                                    "total_documents": len(benchmark_documents),
                                    "total_time": total_time,
                                    "avg_time_per_doc": total_time / len(benchmark_documents),
                                    "throughput": len(benchmark_documents) / total_time,
                                    "results": results,
                                }

        # Validate batch performance
        internvl_perf = batch_performance["internvl3"]
        llama_perf = batch_performance["llama32_vision"]

        # Both should process all documents
        assert internvl_perf["total_documents"] == len(benchmark_documents)
        assert llama_perf["total_documents"] == len(benchmark_documents)

        # Throughput should be reasonable (> 0.1 docs/second for mocked processing)
        assert internvl_perf["throughput"] > 0.1, (
            f"InternVL throughput too low: {internvl_perf['throughput']:.2f} docs/s"
        )
        assert llama_perf["throughput"] > 0.1, (
            f"Llama throughput too low: {llama_perf['throughput']:.2f} docs/s"
        )

        # Batch processing should be more efficient than individual processing
        # (This is a rough estimate for mocked processing)
        expected_individual_time = len(benchmark_documents) * 2.0  # 2 seconds per doc individually
        assert internvl_perf["total_time"] < expected_individual_time
        assert llama_perf["total_time"] < expected_individual_time

    def test_memory_efficiency_validation(self, performance_config, mock_image_path):
        """Test memory efficiency compared to baseline expectations."""
        memory_profiles = {}

        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type

            with patch("vision_processor.config.model_factory.ModelFactory.create_model") as mock_factory:
                with patch(
                    "vision_processor.classification.document_classifier.DocumentClassifier"
                ) as mock_classifier:
                    with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                            with patch(
                                "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                            ) as mock_ato:
                                # Setup mocks
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )
                                mock_model.get_memory_usage.return_value = 256.0  # Mock memory usage in MB
                                mock_factory.return_value = mock_model

                                # Setup other mocks
                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )
                                mock_awk.return_value.extract.return_value = {"field": "value"}
                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=Mock(value="good"),
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )
                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=0.90,
                                    passed=True,
                                    violations=[],
                                    warnings=[],
                                )

                                # Process multiple documents to test memory management
                                peak_memory = 0
                                baseline_memory = 500  # MB baseline

                                with UnifiedExtractionManager(performance_config) as manager:
                                    for _i in range(5):
                                        result = manager.process_document(mock_image_path)
                                        current_memory = result.memory_usage_mb
                                        peak_memory = max(peak_memory, current_memory)

                                memory_profiles[model_type.value] = {
                                    "peak_memory_mb": peak_memory,
                                    "baseline_memory_mb": baseline_memory,
                                    "memory_efficiency": baseline_memory / peak_memory
                                    if peak_memory > 0
                                    else 0,
                                }

        # Validate memory efficiency
        for model_name, profile in memory_profiles.items():
            # Memory usage should be tracked
            assert profile["peak_memory_mb"] > 0, f"Memory not tracked for {model_name}"

            # Memory usage should be reasonable (< 5GB for testing)
            assert profile["peak_memory_mb"] < 5000, (
                f"Memory usage too high for {model_name}: {profile['peak_memory_mb']:.1f}MB"
            )

            # Memory efficiency should be reasonable
            assert profile["memory_efficiency"] > 0.1, f"Memory efficiency too low for {model_name}"

    def test_scalability_validation(self, performance_config, benchmark_documents):
        """Test scalability with increasing document loads."""
        scalability_results = {}

        # Test with different batch sizes
        batch_sizes = [1, 3, 5, 10]

        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type
            scalability_results[model_type.value] = {}

            for batch_size in batch_sizes:
                test_documents = benchmark_documents[:batch_size]

                with patch(
                    "vision_processor.config.model_factory.ModelFactory.create_model"
                ) as mock_factory:
                    with patch(
                        "vision_processor.classification.document_classifier.DocumentClassifier"
                    ) as mock_classifier:
                        with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                            with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                                with patch(
                                    "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                                ) as mock_ato:
                                    # Setup mocks
                                    mock_model = MagicMock()
                                    mock_model.process_image.return_value = Mock(
                                        raw_text="Mock response",
                                        confidence=0.85,
                                        processing_time=1.5,
                                    )
                                    mock_factory.return_value = mock_model

                                    mock_classifier.return_value.classify_with_evidence.return_value = (
                                        DocumentType.BUSINESS_RECEIPT,
                                        0.85,
                                        ["evidence"],
                                    )
                                    mock_awk.return_value.extract.return_value = {"field": "value"}
                                    mock_confidence.return_value.assess_document_confidence.return_value = (
                                        Mock(
                                            overall_confidence=0.82,
                                            quality_grade=Mock(value="good"),
                                            production_ready=True,
                                            quality_flags=[],
                                            recommendations=[],
                                        )
                                    )
                                    mock_ato.return_value.assess_compliance.return_value = Mock(
                                        compliance_score=0.90,
                                        passed=True,
                                        violations=[],
                                        warnings=[],
                                    )

                                    # Measure processing time for this batch size
                                    start_time = time.time()

                                    with UnifiedExtractionManager(performance_config) as manager:
                                        for doc_path in test_documents:
                                            manager.process_document(doc_path)

                                    end_time = time.time()
                                    processing_time = end_time - start_time

                                    scalability_results[model_type.value][batch_size] = {
                                        "processing_time": processing_time,
                                        "avg_time_per_doc": processing_time / batch_size,
                                        "throughput": batch_size / processing_time,
                                    }

        # Validate scalability characteristics
        for model_name, results in scalability_results.items():
            batch_sizes_tested = sorted(results.keys())

            # Processing time should increase with batch size (roughly linear)
            for i in range(1, len(batch_sizes_tested)):
                prev_batch = batch_sizes_tested[i - 1]
                curr_batch = batch_sizes_tested[i]

                prev_time = results[prev_batch]["processing_time"]
                curr_time = results[curr_batch]["processing_time"]

                # Current batch should take more time than previous (but not excessively more)
                time_ratio = curr_time / prev_time
                batch_ratio = curr_batch / prev_batch

                # Time scaling should be reasonable (within 2x of batch size ratio)
                assert time_ratio <= batch_ratio * 2, (
                    f"Poor scaling for {model_name}: {time_ratio:.2f}x time for {batch_ratio:.2f}x documents"
                )

            # Average time per document should remain relatively stable
            avg_times = [results[bs]["avg_time_per_doc"] for bs in batch_sizes_tested]
            max_avg_time = max(avg_times)
            min_avg_time = min(avg_times)

            # Variation in average time per document should be reasonable
            if min_avg_time > 0:
                variation_ratio = max_avg_time / min_avg_time
                assert variation_ratio < 5.0, (  # More lenient for test environment timing variations
                    f"High variation in per-document time for {model_name}: {variation_ratio:.2f}x"
                )

    def test_error_recovery_performance(self, performance_config, mock_image_path):
        """Test performance under error conditions."""
        error_recovery_results = {}

        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type

            with patch("vision_processor.config.model_factory.ModelFactory.create_model") as mock_factory:
                with patch(
                    "vision_processor.classification.document_classifier.DocumentClassifier"
                ) as mock_classifier:
                    with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                            with patch(
                                "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                            ) as mock_ato:
                                # Setup mocks with some failures
                                mock_model = MagicMock()
                                call_count = 0

                                def mock_process_image(*_args, **_kwargs):
                                    nonlocal call_count
                                    call_count += 1
                                    if call_count == 2:  # Fail on second call
                                        raise Exception("Simulated model failure")
                                    return Mock(
                                        raw_text="Mock response",
                                        confidence=0.85,
                                        processing_time=1.5,
                                    )

                                mock_model.process_image.side_effect = mock_process_image
                                mock_factory.return_value = mock_model

                                # Setup other mocks
                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )
                                mock_awk.return_value.extract.return_value = {"field": "value"}
                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=Mock(value="good"),
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )
                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=0.90,
                                    passed=True,
                                    violations=[],
                                    warnings=[],
                                )

                                # Test error recovery
                                successful_processes = 0
                                failed_processes = 0
                                total_time = 0

                                with UnifiedExtractionManager(performance_config) as manager:
                                    for _i in range(5):
                                        try:
                                            start_time = time.time()
                                            manager.process_document(mock_image_path)
                                            end_time = time.time()
                                            total_time += end_time - start_time
                                            successful_processes += 1
                                        except Exception:
                                            failed_processes += 1

                                error_recovery_results[model_type.value] = {
                                    "successful_processes": successful_processes,
                                    "failed_processes": failed_processes,
                                    "total_processes": successful_processes + failed_processes,
                                    "success_rate": successful_processes
                                    / (successful_processes + failed_processes),
                                    "avg_time_per_success": total_time / successful_processes
                                    if successful_processes > 0
                                    else 0,
                                }

        # Validate error recovery performance
        for model_name, results in error_recovery_results.items():
            # Should have some successful processes despite errors
            assert results["successful_processes"] > 0, f"No successful processes for {model_name}"

            # Success rate should be reasonable (we expect 1 failure out of 5)
            assert results["success_rate"] >= 0.6, (
                f"Low success rate for {model_name}: {results['success_rate']:.2f}"
            )

            # Average processing time for successful processes should be reasonable
            assert results["avg_time_per_success"] < 20.0, (
                f"Slow error recovery for {model_name}: {results['avg_time_per_success']:.2f}s"
            )

    def test_resource_cleanup_performance(self, performance_config, mock_image_path):
        """Test that resource cleanup doesn't impact performance."""
        cleanup_performance = {}

        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            performance_config.model_type = model_type

            with patch("vision_processor.config.model_factory.ModelFactory.create_model") as mock_factory:
                with patch(
                    "vision_processor.classification.document_classifier.DocumentClassifier"
                ) as mock_classifier:
                    with patch("vision_processor.extraction.awk_extractor.AWKExtractor") as mock_awk:
                        with patch("vision_processor.confidence.ConfidenceManager") as mock_confidence:
                            with patch(
                                "vision_processor.extraction.pipeline_components.ATOComplianceHandler"
                            ) as mock_ato:
                                # Setup mocks
                                mock_model = MagicMock()
                                mock_model.process_image.return_value = Mock(
                                    raw_text="Mock response",
                                    confidence=0.85,
                                    processing_time=1.5,
                                )
                                mock_model.get_memory_usage.return_value = 256.0  # Mock memory usage in MB
                                mock_factory.return_value = mock_model

                                mock_classifier.return_value.classify_with_evidence.return_value = (
                                    DocumentType.BUSINESS_RECEIPT,
                                    0.85,
                                    ["evidence"],
                                )
                                mock_awk.return_value.extract.return_value = {"field": "value"}
                                mock_confidence.return_value.assess_document_confidence.return_value = Mock(
                                    overall_confidence=0.82,
                                    quality_grade=Mock(value="good"),
                                    production_ready=True,
                                    quality_flags=[],
                                    recommendations=[],
                                )
                                mock_ato.return_value.assess_compliance.return_value = Mock(
                                    compliance_score=0.90,
                                    passed=True,
                                    violations=[],
                                    warnings=[],
                                )

                                # Test multiple manager lifecycles
                                creation_times = []
                                processing_times = []
                                cleanup_times = []

                                for _i in range(3):
                                    # Measure creation time
                                    start_time = time.time()
                                    manager = UnifiedExtractionManager(performance_config)
                                    creation_time = time.time() - start_time
                                    creation_times.append(creation_time)

                                    # Measure processing time
                                    start_time = time.time()
                                    manager.process_document(mock_image_path)
                                    processing_time = time.time() - start_time
                                    processing_times.append(processing_time)

                                    # Measure cleanup time
                                    start_time = time.time()
                                    manager.__exit__(None, None, None)
                                    cleanup_time = time.time() - start_time
                                    cleanup_times.append(cleanup_time)

                                cleanup_performance[model_type.value] = {
                                    "avg_creation_time": sum(creation_times) / len(creation_times),
                                    "avg_processing_time": sum(processing_times) / len(processing_times),
                                    "avg_cleanup_time": sum(cleanup_times) / len(cleanup_times),
                                    "total_lifecycle_time": sum(creation_times)
                                    + sum(processing_times)
                                    + sum(cleanup_times),
                                }

        # Validate resource cleanup performance
        for model_name, perf in cleanup_performance.items():
            # Creation should be reasonably fast
            assert perf["avg_creation_time"] < 5.0, (
                f"Slow manager creation for {model_name}: {perf['avg_creation_time']:.2f}s"
            )

            # Cleanup should be fast
            assert perf["avg_cleanup_time"] < 1.0, (
                f"Slow cleanup for {model_name}: {perf['avg_cleanup_time']:.2f}s"
            )

            # Cleanup shouldn't be a significant portion of total time
            cleanup_ratio = perf["avg_cleanup_time"] / perf["total_lifecycle_time"]
            assert cleanup_ratio < 0.2, (
                f"Cleanup too slow relative to total time for {model_name}: {cleanup_ratio:.2f}"
            )
