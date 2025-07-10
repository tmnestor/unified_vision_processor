"""Document Classification Module

This module provides comprehensive Australian tax document classification
with specialized knowledge of Australian businesses and document formats.

Components:
- DocumentClassifier: Main classification engine with graceful degradation
- DocumentType: Unified taxonomy of 11 Australian tax document types
- ClassificationResult: Structured classification results with evidence
- Australian business knowledge base and format recognition
"""

from .australian_tax_types import (
    DOCUMENT_TYPE_METADATA,
    ClassificationResult,
    DocumentType,
    get_document_type_info,
    get_expected_fields,
    is_business_expense_type,
    requires_gst_validation,
)
from .document_classifier import (
    BasePipelineComponent,
    DocumentClassifier,
)

__all__ = [
    # Document types and results
    "DocumentType",
    "ClassificationResult",
    "DOCUMENT_TYPE_METADATA",
    # Classification engine
    "DocumentClassifier",
    "BasePipelineComponent",
    # Utility functions
    "get_document_type_info",
    "is_business_expense_type",
    "requires_gst_validation",
    "get_expected_fields",
]
