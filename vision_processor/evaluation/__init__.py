"""
Unified Evaluation Framework

Comprehensive evaluation system for fair model comparison using identical
Llama 7-step processing pipeline. Supports SROIE dataset evaluation,
cross-model comparison, and detailed performance analysis.

Key Components:
- UnifiedEvaluator: Cross-model evaluation with Llama pipeline
- SROIEEvaluator: SROIE dataset evaluation (enhanced)
- ModelComparator: Fair model comparison framework
- MetricsCalculator: Advanced metrics computation
- ReportGenerator: Comprehensive evaluation reporting
"""

from .metrics_calculator import MetricsCalculator
from .model_comparator import (
    ComparisonConfiguration,
    ComparisonResult,
    ModelComparator,
)
from .report_generator import ReportGenerator
from .sroie_evaluator import SROIEEvaluator
from .unified_evaluator import (
    DatasetEvaluationResult,
    EvaluationResult,
    UnifiedEvaluator,
)

__all__ = [
    # Core evaluators
    "UnifiedEvaluator",
    "SROIEEvaluator",
    "ModelComparator",
    # Utility components
    "MetricsCalculator",
    "ReportGenerator",
    # Data structures
    "EvaluationResult",
    "DatasetEvaluationResult",
    "ComparisonConfiguration",
    "ComparisonResult",
]

# Version information
__version__ = "1.0.0"
__author__ = "Unified Vision Processor Team"
__description__ = "Fair model comparison and evaluation framework"
