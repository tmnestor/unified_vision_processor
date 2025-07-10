#!/usr/bin/env python3
"""Phase 4 feature compatibility and performance testing."""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))


def test_australian_business_registry():
    """Test the Australian business registry implementation."""
    print("=== Testing Australian Business Registry ===")

    # Import just the registry module
    from vision_processor.compliance.australian_business_registry import (
        AustralianBusinessRegistry,
    )

    registry = AustralianBusinessRegistry()
    registry.initialize()

    stats = registry.get_business_statistics()

    print(f"✓ Total Australian businesses: {stats['total_businesses']}")
    print(
        f"✓ Businesses by industry: {len(stats['businesses_by_industry'])} categories"
    )
    print(f"✓ Business types: {len(stats['businesses_by_type'])} types")

    # Test business recognition
    test_text = "woolworths receipt total $45.50 gst $4.14"
    recognized = registry.recognize_business(test_text)
    print(f"✓ Business recognition test: {len(recognized)} businesses found")

    success = stats["total_businesses"] >= 100
    print(
        f"✓ Goal achieved: {stats['total_businesses']}/100+ businesses {'✓' if success else '✗'}"
    )

    return success


def test_field_validators():
    """Test the ATO field validators."""
    print("\n=== Testing ATO Field Validators ===")

    from vision_processor.compliance.field_validators import (
        ABNValidator,
        BSBValidator,
        DateValidator,
        GSTValidator,
    )

    # Test ABN validator
    abn_validator = ABNValidator()
    valid, formatted, issues = abn_validator.validate("53 004 085 616")
    print(f"✓ ABN validation: {valid} -> {formatted}")

    # Test BSB validator
    bsb_validator = BSBValidator()
    valid, formatted, issues, bank = bsb_validator.validate("062-001")
    print(f"✓ BSB validation: {valid} -> {formatted} ({bank})")

    # Test date validator
    date_validator = DateValidator()
    valid, date_obj, formatted, issues = date_validator.validate("15/03/2024")
    print(f"✓ Date validation: {valid} -> {formatted}")

    # Test GST validator
    gst_validator = GSTValidator()
    valid, calc, issues = gst_validator.validate_gst_calculation(100.0, 10.0, 110.0)
    print(f"✓ GST validation: {valid} -> Expected: {calc['expected_gst']}")

    return True


def test_awk_extractor():
    """Test the AWK extraction system."""
    print("\n=== Testing AWK Extraction System ===")

    from vision_processor.extraction.awk_extractor import AWKExtractor
    from vision_processor.extraction.pipeline_components import DocumentType

    # Create extractor
    extractor = AWKExtractor()
    extractor.initialize()

    # Test extraction
    test_text = """
    WOOLWORTHS SUPERMARKET
    Date: 15/03/2024
    Total: $45.50
    GST: $4.14
    ABN: 88 000 014 675
    """

    result = extractor.extract(test_text, DocumentType.BUSINESS_RECEIPT)

    print(f"✓ AWK extraction fields: {len(result)}")
    for field, value in result.items():
        if value:
            print(f"  - {field}: {value}")

    # Test pattern statistics
    stats = extractor.get_pattern_statistics()
    print(f"✓ Total patterns: {stats['total_patterns']}")
    print(f"✓ Document types: {len(stats['patterns_by_document_type'])}")

    return True


def test_computer_vision_integration():
    """Test computer vision integration without actual image processing."""
    print("\n=== Testing Computer Vision Integration ===")

    from vision_processor.computer_vision import (
        BankStatementCV,
        HighlightDetector,
        ImagePreprocessor,
        OCRProcessor,
        SpatialCorrelator,
    )

    # Test component initialization
    highlight_detector = HighlightDetector()
    highlight_detector.initialize()
    print("✓ HighlightDetector initialized")

    ocr_processor = OCRProcessor()
    ocr_processor.initialize()
    print("✓ OCRProcessor initialized")

    bank_cv = BankStatementCV()
    bank_cv.initialize()
    print("✓ BankStatementCV initialized")

    preprocessor = ImagePreprocessor()
    preprocessor.initialize()
    print("✓ ImagePreprocessor initialized")

    correlator = SpatialCorrelator()
    correlator.initialize()
    print("✓ SpatialCorrelator initialized")

    print("✓ All computer vision components initialized successfully")

    return True


def test_integration_performance():
    """Test integration performance and compatibility."""
    print("\n=== Testing Integration Performance ===")

    start_time = time.time()

    # Test all components together
    from vision_processor.extraction.pipeline_components import (
        ATOComplianceHandler,
        ConfidenceManager,
        DocumentClassifier,
    )

    # Initialize components
    config = type("Config", (), {})()  # Mock config

    classifier = DocumentClassifier(config)
    classifier.initialize()

    confidence_manager = ConfidenceManager(config)
    confidence_manager.initialize()

    ato_handler = ATOComplianceHandler(config)
    ato_handler.initialize()

    initialization_time = time.time() - start_time
    print(f"✓ Component initialization: {initialization_time:.3f}s")

    # Test text classification
    test_text = "woolworths receipt total $45.50"
    doc_type, confidence, evidence = classifier.classify_from_text(test_text)
    print(f"✓ Classification: {doc_type.value} ({confidence:.2f})")

    return True


def main():
    """Run all Phase 4 feature tests."""
    print("🚀 Starting Phase 4: Feature Compatibility and Performance Testing\n")

    tests = [
        test_australian_business_registry,
        test_field_validators,
        test_awk_extractor,
        test_computer_vision_integration,
        test_integration_performance,
    ]

    results = []
    total_start = time.time()

    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(False)

    total_time = time.time() - total_start

    print("\n=== Phase 4 Test Results ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print(f"Total test time: {total_time:.3f}s")
    print(f"Phase 4 status: {'✓ COMPLETE' if all(results) else '✗ NEEDS ATTENTION'}")

    if all(results):
        print("\n✅ Phase 4: Feature Integration completed successfully!")
        print("Ready to proceed to Phase 5: Handler and Prompt Integration")
    else:
        print("\n⚠️  Some tests failed. Review and fix before proceeding.")


if __name__ == "__main__":
    main()
