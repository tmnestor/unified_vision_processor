"""Confidence Assessment Module

This module provides 4-component confidence scoring for production readiness
assessment of Australian tax document processing.

Components:
- ConfidenceManager: Main 4-component confidence scoring system
- ComplianceResult: ATO compliance assessment results
- ConfidenceResult: Comprehensive confidence assessment results
"""

from .confidence_integration_manager import (
    BasePipelineComponent,
    ComplianceResult,
    ConfidenceManager,
    ConfidenceResult,
)

__all__ = [
    "BasePipelineComponent",
    "ComplianceResult",
    "ConfidenceManager",
    "ConfidenceResult",
]
