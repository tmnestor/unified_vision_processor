"""Prompt Optimizer

Performance optimization system for prompt selection and effectiveness tracking.
"""

import logging
from typing import Any

from ..classification.australian_tax_types import DocumentType

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimize prompt performance and selection based on extraction results.

    Features:
    - Performance tracking by document type and prompt strategy
    - A/B testing support for prompt variations
    - Automatic optimization recommendations
    - Confidence score correlation analysis
    """

    def __init__(self, config: Any = None):
        self.config = config or {}
        self.initialized = False

        # Performance tracking
        self.performance_data: dict[str, dict[str, list[float]]] = {}
        self.usage_statistics: dict[str, dict[str, int]] = {}

        # Optimization settings
        self.min_samples_for_optimization = 10
        self.performance_threshold = 0.7
        self.confidence_weight = 0.4
        self.accuracy_weight = 0.6

    def initialize(self) -> None:
        """Initialize the prompt optimizer."""
        if self.initialized:
            return

        # Initialize tracking structures
        for doc_type in DocumentType:
            self.performance_data[doc_type.value] = {}
            self.usage_statistics[doc_type.value] = {}

        logger.info("PromptOptimizer initialized")
        self.initialized = True

    def record_prompt_performance(
        self,
        document_type: DocumentType,
        prompt_strategy: str,
        confidence_score: float,
        extraction_accuracy: float,
        _processing_time: float = 0.0,
    ) -> None:
        """Record performance metrics for a prompt strategy.

        Args:
            document_type: Type of document processed
            prompt_strategy: Strategy used (ato_compliance, highlight_enhanced, etc.)
            confidence_score: Confidence in extraction (0-1)
            extraction_accuracy: Accuracy of extraction (0-1)
            processing_time: Time taken for processing (seconds)

        """
        if not self.initialized:
            self.initialize()

        doc_type_str = document_type.value

        # Initialize if not exists
        if prompt_strategy not in self.performance_data[doc_type_str]:
            self.performance_data[doc_type_str][prompt_strategy] = []
            self.usage_statistics[doc_type_str][prompt_strategy] = 0

        # Calculate composite performance score
        performance_score = (
            confidence_score * self.confidence_weight + extraction_accuracy * self.accuracy_weight
        )

        # Record performance
        self.performance_data[doc_type_str][prompt_strategy].append(performance_score)
        self.usage_statistics[doc_type_str][prompt_strategy] += 1

        logger.debug(
            f"Recorded performance for {doc_type_str}/{prompt_strategy}: {performance_score:.3f}",
        )

    def get_optimal_strategy(self, document_type: DocumentType) -> str | None:
        """Get the optimal prompt strategy for a document type.

        Args:
            document_type: Type of document

        Returns:
            Optimal prompt strategy name, or None if insufficient data

        """
        if not self.initialized:
            self.initialize()

        doc_type_str = document_type.value
        strategies = self.performance_data[doc_type_str]

        if not strategies:
            return None

        strategy_averages = {}

        for strategy, performances in strategies.items():
            if len(performances) >= self.min_samples_for_optimization:
                strategy_averages[strategy] = sum(performances) / len(performances)

        if not strategy_averages:
            return None

        # Return strategy with highest average performance
        optimal_strategy = max(strategy_averages, key=strategy_averages.get)

        # Only recommend if performance is above threshold
        if strategy_averages[optimal_strategy] >= self.performance_threshold:
            return optimal_strategy

        return None

    def get_performance_analysis(self, document_type: DocumentType) -> dict[str, Any]:
        """Get detailed performance analysis for a document type.

        Args:
            document_type: Type of document

        Returns:
            Dictionary with performance statistics

        """
        doc_type_str = document_type.value
        strategies = self.performance_data[doc_type_str]

        analysis = {
            "document_type": doc_type_str,
            "strategies": {},
            "recommendations": [],
            "total_samples": 0,
        }

        for strategy, performances in strategies.items():
            if not performances:
                continue

            strategy_stats = {
                "sample_count": len(performances),
                "average_performance": sum(performances) / len(performances),
                "min_performance": min(performances),
                "max_performance": max(performances),
                "usage_count": self.usage_statistics[doc_type_str].get(strategy, 0),
            }

            analysis["strategies"][strategy] = strategy_stats
            analysis["total_samples"] += len(performances)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        strategies = analysis["strategies"]

        if not strategies:
            recommendations.append("Insufficient data for recommendations")
            return recommendations

        # Find best and worst performing strategies
        best_strategy = max(
            strategies.items(),
            key=lambda x: x[1]["average_performance"],
        )
        worst_strategy = min(
            strategies.items(),
            key=lambda x: x[1]["average_performance"],
        )

        # Performance gap analysis
        performance_gap = best_strategy[1]["average_performance"] - worst_strategy[1]["average_performance"]

        if performance_gap > 0.2:
            recommendations.append(
                f"Significant performance gap detected. Prefer '{best_strategy[0]}' over '{worst_strategy[0]}'",
            )

        # Sample size recommendations
        for strategy, stats in strategies.items():
            if stats["sample_count"] < self.min_samples_for_optimization:
                recommendations.append(
                    f"Strategy '{strategy}' needs more samples ({stats['sample_count']}/{self.min_samples_for_optimization})",
                )

        # Performance threshold recommendations
        for strategy, stats in strategies.items():
            if stats["average_performance"] < self.performance_threshold:
                recommendations.append(
                    f"Strategy '{strategy}' below performance threshold ({stats['average_performance']:.3f} < {self.performance_threshold})",
                )

        return recommendations

    def compare_strategies(
        self,
        document_type: DocumentType,
        strategy1: str,
        strategy2: str,
    ) -> dict[str, Any]:
        """Compare performance between two strategies.

        Args:
            document_type: Type of document
            strategy1: First strategy name
            strategy2: Second strategy name

        Returns:
            Comparison results

        """
        doc_type_str = document_type.value

        comparison = {
            "document_type": doc_type_str,
            "strategy1": {"name": strategy1, "performance": None, "samples": 0},
            "strategy2": {"name": strategy2, "performance": None, "samples": 0},
            "winner": None,
            "confidence": 0.0,
            "recommendation": "",
        }

        # Get performance data
        strategy1_data = self.performance_data[doc_type_str].get(strategy1, [])
        strategy2_data = self.performance_data[doc_type_str].get(strategy2, [])

        if strategy1_data:
            comparison["strategy1"]["performance"] = sum(strategy1_data) / len(
                strategy1_data,
            )
            comparison["strategy1"]["samples"] = len(strategy1_data)

        if strategy2_data:
            comparison["strategy2"]["performance"] = sum(strategy2_data) / len(
                strategy2_data,
            )
            comparison["strategy2"]["samples"] = len(strategy2_data)

        # Determine winner
        if comparison["strategy1"]["performance"] and comparison["strategy2"]["performance"]:
            perf1 = comparison["strategy1"]["performance"]
            perf2 = comparison["strategy2"]["performance"]

            if perf1 > perf2:
                comparison["winner"] = strategy1
                comparison["confidence"] = min(
                    (perf1 - perf2) * 5,
                    1.0,
                )  # Scale difference
            elif perf2 > perf1:
                comparison["winner"] = strategy2
                comparison["confidence"] = min((perf2 - perf1) * 5, 1.0)
            else:
                comparison["winner"] = "tie"
                comparison["confidence"] = 0.0

        # Generate recommendation
        if comparison["winner"] and comparison["winner"] != "tie":
            if comparison["confidence"] > 0.6:
                comparison["recommendation"] = f"Strong preference for {comparison['winner']}"
            elif comparison["confidence"] > 0.3:
                comparison["recommendation"] = f"Moderate preference for {comparison['winner']}"
            else:
                comparison["recommendation"] = f"Weak preference for {comparison['winner']}"
        else:
            comparison["recommendation"] = "No clear preference - continue A/B testing"

        return comparison

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get comprehensive optimization summary across all document types."""
        summary = {
            "total_document_types": 0,
            "total_strategies": 0,
            "total_samples": 0,
            "optimal_strategies": {},
            "performance_leaders": {},
            "recommendations": [],
        }

        all_strategies = set()

        for doc_type in DocumentType:
            doc_type_str = doc_type.value

            if doc_type_str in self.performance_data:
                strategies = self.performance_data[doc_type_str]

                if strategies:
                    summary["total_document_types"] += 1
                    all_strategies.update(strategies.keys())

                    # Count samples
                    for strategy_performances in strategies.values():
                        summary["total_samples"] += len(strategy_performances)

                    # Get optimal strategy for this document type
                    optimal = self.get_optimal_strategy(doc_type)
                    if optimal:
                        summary["optimal_strategies"][doc_type_str] = optimal

                    # Find performance leader
                    if strategies:
                        leader = max(
                            strategies.items(),
                            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
                        )
                        if leader[1]:  # Has performance data
                            summary["performance_leaders"][doc_type_str] = {
                                "strategy": leader[0],
                                "performance": sum(leader[1]) / len(leader[1]),
                            }

        summary["total_strategies"] = len(all_strategies)

        # Generate global recommendations
        if summary["total_samples"] < 100:
            summary["recommendations"].append(
                "Collect more performance data for better optimization",
            )

        if len(summary["optimal_strategies"]) < summary["total_document_types"] / 2:
            summary["recommendations"].append(
                "Many document types lack clear optimal strategies",
            )

        return summary
