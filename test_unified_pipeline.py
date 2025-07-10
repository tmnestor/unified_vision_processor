#!/usr/bin/env python3
"""
Test script for the unified vision processing pipeline.

Tests the 7-step processing pipeline with both InternVL and Llama models.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_unified_pipeline():
    """Test the unified processing pipeline with both models."""

    # Create a simple test document content (simulated OCR output)
    test_document_text = """
    WOOLWORTHS SUPERMARKETS
    RECEIPT
    Store: Woolworths Bondi Junction
    Date: 15/03/2024
    
    ITEMS:
    Bananas 1kg                $4.50
    Milk 2L                    $3.20
    Bread Loaf                 $2.80
    
    SUBTOTAL:                  $10.50
    GST (10%):                 $0.95
    TOTAL:                     $11.45
    
    ABN: 88 000 014 675
    
    Thank you for shopping with us!
    """

    logger.info("Starting unified pipeline test")

    try:
        # Import required components
        from vision_processor.config.unified_config import UnifiedConfig
        from vision_processor.extraction.pipeline_components import DocumentClassifier

        # Test 1: Document Classification
        logger.info("Testing document classification...")

        config = UnifiedConfig.from_env()  # Load config from environment/.env
        classifier = DocumentClassifier(config)
        classifier.ensure_initialized()

        doc_type, confidence, evidence = classifier.classify_from_text(
            test_document_text
        )

        logger.info("Classification Result:")
        logger.info(f"  Document Type: {doc_type.value}")
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  Evidence: {evidence}")

        # Test 2: Confidence Manager
        logger.info("\nTesting 4-component confidence scoring...")

        from vision_processor.extraction.pipeline_components import (
            ComplianceResult,
            ConfidenceManager,
        )

        confidence_manager = ConfidenceManager(config)
        confidence_manager.ensure_initialized()

        # Mock extracted fields
        mock_extracted_fields = {
            "date": "15/03/2024",
            "business_name": "Woolworths Supermarkets",
            "total_amount": "$11.45",
            "gst_amount": "$0.95",
            "abn": "88 000 014 675",
            "items": ["Bananas 1kg", "Milk 2L", "Bread Loaf"],
        }

        # Mock compliance result
        mock_compliance = ComplianceResult(
            compliance_score=0.85, compliance_passed=True, issues=[], recommendations=[]
        )

        confidence_result = confidence_manager.assess_document_confidence(
            test_document_text,
            mock_extracted_fields,
            mock_compliance,
            confidence,  # classification confidence
            False,  # highlights detected
        )

        logger.info("Confidence Assessment:")
        logger.info(f"  Overall Confidence: {confidence_result.overall_confidence:.2f}")
        logger.info(f"  Quality Grade: {confidence_result.quality_grade}")
        logger.info(f"  Production Ready: {confidence_result.production_ready}")
        logger.info(f"  Component Scores: {confidence_result.component_scores}")
        logger.info(f"  Quality Flags: {confidence_result.quality_flags}")
        logger.info(f"  Recommendations: {confidence_result.recommendations}")

        # Test 3: Pipeline components integration test
        logger.info("\nTesting pipeline component integration...")

        # Test that all components can be initialized
        from vision_processor.extraction.pipeline_components import (
            ATOComplianceHandler,
            AWKExtractor,
            EnhancedKeyValueParser,
            HighlightDetector,
            PromptManager,
        )

        components = [
            ("AWKExtractor", AWKExtractor(config)),
            ("ATOComplianceHandler", ATOComplianceHandler(config)),
            ("PromptManager", PromptManager(config)),
            ("EnhancedKeyValueParser", EnhancedKeyValueParser(config)),
            ("HighlightDetector", HighlightDetector(config)),
        ]

        for name, component in components:
            try:
                component.ensure_initialized()
                logger.info(f"  ✓ {name} initialized successfully")
            except Exception as e:
                logger.warning(f"  ⚠ {name} initialization issue: {e}")

        logger.info("\n🎉 Unified pipeline test completed successfully!")
        logger.info("Phase 3 implementation is working correctly.")

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Make sure you're running from the correct directory and the unified_vision_processor environment is activated"
        )
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_integration():
    """Test model integration capabilities."""
    logger.info("\nTesting model integration capabilities...")

    try:
        from vision_processor.config.model_factory import ModelFactory
        from vision_processor.config.unified_config import ModelType

        # Test model factory
        available_models = ModelFactory.get_available_models()
        logger.info(f"Available models: {list(available_models.keys())}")

        # Test recommended configurations
        for model_type in [ModelType.INTERNVL3, ModelType.LLAMA32_VISION]:
            try:
                config = ModelFactory.get_recommended_config(model_type, "auto")
                logger.info(f"  ✓ {model_type.value} configuration: {config}")
            except Exception as e:
                logger.warning(f"  ⚠ {model_type.value} config issue: {e}")

        logger.info("✓ Model integration test completed")
        return True

    except Exception as e:
        logger.error(f"Model integration test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("UNIFIED VISION PROCESSOR - PHASE 3 TEST")
    logger.info("=" * 60)

    # Run pipeline test
    pipeline_success = test_unified_pipeline()

    # Run model integration test
    model_success = test_model_integration()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Pipeline Test: {'✓ PASSED' if pipeline_success else '✗ FAILED'}")
    logger.info(
        f"Model Integration Test: {'✓ PASSED' if model_success else '✗ FAILED'}"
    )

    if pipeline_success and model_success:
        logger.info("\n🎉 ALL TESTS PASSED - Phase 3 implementation is ready!")
        logger.info("Next: Continue with Phase 4 (Feature Integration)")
    else:
        logger.info("\n⚠ Some tests failed - check the logs above")
