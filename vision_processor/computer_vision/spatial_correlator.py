"""Spatial Correlator for Highlight-Text Correlation

This module provides spatial analysis capabilities to correlate highlighted
regions with surrounding text content for better context understanding.
"""

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .highlight_detector import HighlightRegion
from .ocr_processor import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class SpatialRegion:
    """Represents a spatial region with text content."""

    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float
    region_type: str  # "highlight", "surrounding", "header", "footer"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x, y, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> tuple[int, int]:
        """Return center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Return area of the region."""
        return self.width * self.height


@dataclass
class SpatialCorrelation:
    """Represents correlation between highlighted region and surrounding text."""

    highlight_region: HighlightRegion
    highlighted_text: str | None
    surrounding_regions: list[SpatialRegion]
    context_text: str
    correlation_score: float
    spatial_relationships: dict[str, Any]

    def get_full_context(self) -> str:
        """Get full context including highlighted text and surroundings."""
        context_parts = []

        if self.highlighted_text:
            context_parts.append(f"HIGHLIGHTED: {self.highlighted_text}")

        if self.context_text:
            context_parts.append(f"CONTEXT: {self.context_text}")

        return " | ".join(context_parts)


class SpatialCorrelator:
    """Spatial correlator for analyzing relationships between highlighted regions and text.

    Features:
    - Spatial relationship analysis
    - Context extraction around highlights
    - Text correlation scoring
    - Layout pattern recognition
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # Spatial analysis configuration
        self.spatial_config = {
            "context_radius": 100,  # Pixels around highlight to consider
            "min_text_confidence": 0.3,
            "max_context_regions": 10,
            "spatial_weights": {
                "proximity": 0.4,  # How close text is to highlight
                "alignment": 0.3,  # How well aligned text is
                "size_similarity": 0.2,  # Similar size regions
                "text_quality": 0.1,  # OCR confidence
            },
        }

        # Layout patterns for different document types
        self.layout_patterns = {
            "bank_statement": {
                "transaction_row": {
                    "typical_height": (20, 60),
                    "typical_width": (300, 2000),
                    "text_alignment": "left",
                    "expected_fields": ["date", "description", "amount"],
                },
                "header": {
                    "position": "top",
                    "height_ratio": 0.2,
                    "contains": ["account", "statement", "period"],
                },
            },
            "receipt": {
                "item_line": {
                    "typical_height": (15, 40),
                    "alignment": "vertical",
                    "expected_fields": ["item", "price"],
                },
                "total_line": {
                    "position": "bottom",
                    "emphasis": "highlighted",
                    "expected_fields": ["total", "amount"],
                },
            },
        }

    def initialize(self) -> None:
        """Initialize spatial correlator."""
        if self.initialized:
            return

        # Load configuration overrides
        if hasattr(self.config, "spatial_config"):
            self.spatial_config.update(self.config.spatial_config)

        logger.info("SpatialCorrelator initialized")
        self.initialized = True

    def correlate_highlights_with_text(
        self,
        image_path: str | Any,
        highlights: list[HighlightRegion],
        ocr_results: list[OCRResult],
        document_type: str = "general",
    ) -> list[SpatialCorrelation]:
        """Correlate highlighted regions with surrounding text.

        Args:
            image_path: Path to original image
            highlights: List of highlight regions
            ocr_results: List of OCR results
            document_type: Type of document for layout-specific analysis

        Returns:
            List of spatial correlations

        """
        if not self.initialized:
            self.initialize()

        correlations = []

        try:
            # Load image for spatial analysis
            if isinstance(image_path, (str, type(None))):
                image = None  # We'll work with coordinates only
            else:
                image = cv2.imread(str(image_path)) if hasattr(image_path, "__str__") else None

            for highlight in highlights:
                correlation = self._analyze_highlight_correlation(
                    highlight,
                    ocr_results,
                    image,
                    document_type,
                )
                correlations.append(correlation)

            # Sort by correlation score
            correlations.sort(key=lambda c: c.correlation_score, reverse=True)

            logger.info(f"Generated {len(correlations)} spatial correlations")
            return correlations

        except Exception as e:
            logger.error(f"Error correlating highlights with text: {e}")
            return []

    def _analyze_highlight_correlation(
        self,
        highlight: HighlightRegion,
        ocr_results: list[OCRResult],
        _image: np.ndarray | None,
        document_type: str,
    ) -> SpatialCorrelation:
        """Analyze correlation for a single highlight."""
        # Find OCR text within the highlight
        highlighted_text = self._find_text_in_highlight(highlight, ocr_results)

        # Find surrounding text regions
        surrounding_regions = self._find_surrounding_regions(highlight, ocr_results)

        # Extract context text
        context_text = self._extract_context_text(surrounding_regions)

        # Calculate spatial relationships
        spatial_relationships = self._calculate_spatial_relationships(
            highlight,
            surrounding_regions,
            document_type,
        )

        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(
            highlight,
            highlighted_text,
            surrounding_regions,
            spatial_relationships,
        )

        return SpatialCorrelation(
            highlight_region=highlight,
            highlighted_text=highlighted_text,
            surrounding_regions=surrounding_regions,
            context_text=context_text,
            correlation_score=correlation_score,
            spatial_relationships=spatial_relationships,
        )

    def _find_text_in_highlight(
        self,
        highlight: HighlightRegion,
        ocr_results: list[OCRResult],
    ) -> str | None:
        """Find OCR text that falls within the highlight region."""
        highlight_bbox = highlight.bbox

        for ocr_result in ocr_results:
            ocr_bbox = ocr_result.bbox

            # Check if OCR result overlaps with highlight
            if self._rectangles_overlap(highlight_bbox, ocr_bbox):
                # Calculate overlap percentage
                overlap_area = self._calculate_overlap_area(highlight_bbox, ocr_bbox)
                ocr_area = (ocr_bbox[2] - ocr_bbox[0]) * (ocr_bbox[3] - ocr_bbox[1])

                overlap_percentage = overlap_area / ocr_area if ocr_area > 0 else 0

                # If significant overlap, consider this text as highlighted
                if overlap_percentage > 0.5:
                    return ocr_result.text

        return None

    def _find_surrounding_regions(
        self,
        highlight: HighlightRegion,
        ocr_results: list[OCRResult],
    ) -> list[SpatialRegion]:
        """Find text regions surrounding the highlight."""
        surrounding_regions = []
        context_radius = self.spatial_config["context_radius"]

        # Define extended region around highlight
        extended_bbox = (
            max(0, highlight.x - context_radius),
            max(0, highlight.y - context_radius),
            highlight.x + highlight.width + context_radius,
            highlight.y + highlight.height + context_radius,
        )

        for ocr_result in ocr_results:
            ocr_bbox = ocr_result.bbox

            # Skip if this is the highlighted text itself
            if self._rectangles_overlap(highlight.bbox, ocr_bbox):
                continue

            # Check if OCR result is in the extended region
            if self._point_in_rectangle(
                ((ocr_bbox[0] + ocr_bbox[2]) // 2, (ocr_bbox[1] + ocr_bbox[3]) // 2),
                extended_bbox,
            ):
                # Determine region type based on position relative to highlight
                region_type = self._determine_region_type(highlight, ocr_result)

                spatial_region = SpatialRegion(
                    x=ocr_bbox[0],
                    y=ocr_bbox[1],
                    width=ocr_bbox[2] - ocr_bbox[0],
                    height=ocr_bbox[3] - ocr_bbox[1],
                    text=ocr_result.text,
                    confidence=ocr_result.confidence,
                    region_type=region_type,
                )

                surrounding_regions.append(spatial_region)

        # Sort by distance from highlight
        highlight_center = highlight.center
        surrounding_regions.sort(
            key=lambda r: self._calculate_distance(highlight_center, r.center),
        )

        # Limit number of regions
        max_regions = self.spatial_config["max_context_regions"]
        return surrounding_regions[:max_regions]

    def _determine_region_type(
        self,
        highlight: HighlightRegion,
        ocr_result: OCRResult,
    ) -> str:
        """Determine the type of spatial region relative to highlight."""
        ocr_bbox = ocr_result.bbox
        highlight_bbox = highlight.bbox

        # Calculate relative position
        ocr_center_x = (ocr_bbox[0] + ocr_bbox[2]) // 2
        ocr_center_y = (ocr_bbox[1] + ocr_bbox[3]) // 2

        # Determine relative position
        if ocr_center_y < highlight_bbox[1]:
            return "above"
        if ocr_center_y > highlight_bbox[3]:
            return "below"
        if ocr_center_x < highlight_bbox[0]:
            return "left"
        if ocr_center_x > highlight_bbox[2]:
            return "right"
        return "overlapping"

    def _extract_context_text(self, surrounding_regions: list[SpatialRegion]) -> str:
        """Extract meaningful context text from surrounding regions."""
        if not surrounding_regions:
            return ""

        # Filter by confidence
        min_confidence = self.spatial_config["min_text_confidence"]
        quality_regions = [r for r in surrounding_regions if r.confidence >= min_confidence]

        if not quality_regions:
            quality_regions = surrounding_regions[:3]  # Take top 3 if none meet confidence threshold

        # Group by position
        context_parts = {
            "above": [],
            "below": [],
            "left": [],
            "right": [],
            "overlapping": [],
        }

        for region in quality_regions:
            context_parts[region.region_type].append(region.text)

        # Build context string
        context_components = []

        if context_parts["above"]:
            context_components.append(f"ABOVE: {' '.join(context_parts['above'])}")

        if context_parts["left"]:
            context_components.append(f"LEFT: {' '.join(context_parts['left'])}")

        if context_parts["right"]:
            context_components.append(f"RIGHT: {' '.join(context_parts['right'])}")

        if context_parts["below"]:
            context_components.append(f"BELOW: {' '.join(context_parts['below'])}")

        return " | ".join(context_components)

    def _calculate_spatial_relationships(
        self,
        highlight: HighlightRegion,
        surrounding_regions: list[SpatialRegion],
        document_type: str,
    ) -> dict[str, Any]:
        """Calculate spatial relationships and layout analysis."""
        relationships = {
            "document_type": document_type,
            "highlight_position": self._get_relative_position(highlight),
            "surrounding_count": len(surrounding_regions),
            "proximity_scores": {},
            "alignment_scores": {},
            "layout_patterns": {},
        }

        highlight_center = highlight.center

        for region in surrounding_regions:
            region_id = f"{region.region_type}_{hash(region.text) % 1000}"

            # Proximity score (inverse of distance)
            distance = self._calculate_distance(highlight_center, region.center)
            proximity_score = 1.0 / (1.0 + distance / 100.0)  # Normalize
            relationships["proximity_scores"][region_id] = proximity_score

            # Alignment score
            alignment_score = self._calculate_alignment_score(highlight, region)
            relationships["alignment_scores"][region_id] = alignment_score

        # Layout pattern analysis
        if document_type in self.layout_patterns:
            layout_analysis = self._analyze_layout_patterns(
                highlight,
                surrounding_regions,
                document_type,
            )
            relationships["layout_patterns"] = layout_analysis

        return relationships

    def _get_relative_position(self, highlight: HighlightRegion) -> dict[str, float]:
        """Get relative position of highlight (requires image dimensions)."""
        # This would normally use image dimensions, but we'll use default values
        default_width, default_height = 1000, 1000

        return {
            "relative_x": highlight.x / default_width,
            "relative_y": highlight.y / default_height,
            "relative_width": highlight.width / default_width,
            "relative_height": highlight.height / default_height,
        }

    def _calculate_alignment_score(
        self,
        highlight: HighlightRegion,
        region: SpatialRegion,
    ) -> float:
        """Calculate how well aligned a region is with the highlight."""
        # Horizontal alignment
        h_overlap = max(
            0,
            min(highlight.x + highlight.width, region.x + region.width) - max(highlight.x, region.x),
        )
        h_alignment = h_overlap / max(highlight.width, region.width)

        # Vertical alignment
        v_overlap = max(
            0,
            min(highlight.y + highlight.height, region.y + region.height) - max(highlight.y, region.y),
        )
        v_alignment = v_overlap / max(highlight.height, region.height)

        # Combined alignment score
        return (h_alignment + v_alignment) / 2.0

    def _analyze_layout_patterns(
        self,
        highlight: HighlightRegion,
        surrounding_regions: list[SpatialRegion],
        document_type: str,
    ) -> dict[str, Any]:
        """Analyze document-specific layout patterns."""
        patterns = self.layout_patterns.get(document_type, {})
        analysis = {}

        if document_type == "bank_statement":
            # Check if this looks like a transaction row
            transaction_pattern = patterns.get("transaction_row", {})

            height_range = transaction_pattern.get("typical_height", (20, 60))
            width_range = transaction_pattern.get("typical_width", (300, 2000))

            height_match = height_range[0] <= highlight.height <= height_range[1]
            width_match = width_range[0] <= highlight.width <= width_range[1]

            analysis["transaction_row_likelihood"] = 0.8 if (height_match and width_match) else 0.3

            # Check for expected fields
            expected_fields = transaction_pattern.get("expected_fields", [])
            field_matches = []

            for region in surrounding_regions:
                text_lower = region.text.lower()
                for field in expected_fields:
                    if field in text_lower:
                        field_matches.append(field)

            analysis["field_matches"] = field_matches
            analysis["field_coverage"] = len(set(field_matches)) / len(expected_fields)

        return analysis

    def _calculate_correlation_score(
        self,
        highlight: HighlightRegion,
        highlighted_text: str | None,
        surrounding_regions: list[SpatialRegion],
        spatial_relationships: dict[str, Any],
    ) -> float:
        """Calculate overall correlation score."""
        weights = self.spatial_config["spatial_weights"]
        score = 0.0

        # Base score from having highlighted text
        if highlighted_text and highlighted_text.strip():
            score += 0.3

        # Proximity component
        if surrounding_regions:
            proximity_scores = spatial_relationships.get("proximity_scores", {})
            if proximity_scores:
                avg_proximity = sum(proximity_scores.values()) / len(proximity_scores)
                score += avg_proximity * weights["proximity"]

        # Alignment component
        alignment_scores = spatial_relationships.get("alignment_scores", {})
        if alignment_scores:
            avg_alignment = sum(alignment_scores.values()) / len(alignment_scores)
            score += avg_alignment * weights["alignment"]

        # Size similarity component (regions of similar size are likely related)
        if surrounding_regions:
            size_similarities = []
            highlight_area = highlight.area

            for region in surrounding_regions:
                if highlight_area > 0:
                    size_ratio = min(region.area, highlight_area) / max(
                        region.area,
                        highlight_area,
                    )
                    size_similarities.append(size_ratio)

            if size_similarities:
                avg_size_similarity = sum(size_similarities) / len(size_similarities)
                score += avg_size_similarity * weights["size_similarity"]

        # Text quality component
        if surrounding_regions:
            avg_confidence = sum(r.confidence for r in surrounding_regions) / len(
                surrounding_regions,
            )
            score += avg_confidence * weights["text_quality"]

        return min(score, 1.0)

    def _rectangles_overlap(
        self,
        rect1: tuple[int, int, int, int],
        rect2: tuple[int, int, int, int],
    ) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2

        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def _calculate_overlap_area(
        self,
        rect1: tuple[int, int, int, int],
        rect2: tuple[int, int, int, int],
    ) -> int:
        """Calculate the area of overlap between two rectangles."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2

        # Calculate intersection
        left = max(x1, x3)
        right = min(x2, x4)
        top = max(y1, y3)
        bottom = min(y2, y4)

        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        return 0

    def _point_in_rectangle(
        self,
        point: tuple[int, int],
        rect: tuple[int, int, int, int],
    ) -> bool:
        """Check if a point is inside a rectangle."""
        x, y = point
        x1, y1, x2, y2 = rect

        return x1 <= x <= x2 and y1 <= y <= y2

    def _calculate_distance(
        self,
        point1: tuple[int, int],
        point2: tuple[int, int],
    ) -> float:
        """Calculate Euclidean distance between two points."""
        x1, y1 = point1
        x2, y2 = point2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def generate_correlation_report(
        self,
        correlations: list[SpatialCorrelation],
    ) -> dict[str, Any]:
        """Generate a comprehensive correlation analysis report."""
        if not correlations:
            return {"error": "No correlations to analyze"}

        report = {
            "summary": {
                "total_correlations": len(correlations),
                "high_confidence": len(
                    [c for c in correlations if c.correlation_score > 0.7],
                ),
                "medium_confidence": len(
                    [c for c in correlations if 0.4 <= c.correlation_score <= 0.7],
                ),
                "low_confidence": len(
                    [c for c in correlations if c.correlation_score < 0.4],
                ),
            },
            "average_correlation_score": sum(c.correlation_score for c in correlations) / len(correlations),
            "best_correlations": [],
            "spatial_insights": {},
        }

        # Top correlations
        top_correlations = sorted(
            correlations,
            key=lambda c: c.correlation_score,
            reverse=True,
        )[:5]

        for i, correlation in enumerate(top_correlations, 1):
            report["best_correlations"].append(
                {
                    "rank": i,
                    "score": correlation.correlation_score,
                    "highlighted_text": correlation.highlighted_text,
                    "context": correlation.context_text[:200],  # Truncate for readability
                    "surrounding_regions_count": len(correlation.surrounding_regions),
                },
            )

        # Spatial insights
        total_regions = sum(len(c.surrounding_regions) for c in correlations)
        if total_regions > 0:
            report["spatial_insights"] = {
                "average_surrounding_regions": total_regions / len(correlations),
                "context_coverage": len([c for c in correlations if c.context_text]) / len(correlations),
                "highlighted_text_detection": len(
                    [c for c in correlations if c.highlighted_text],
                )
                / len(correlations),
            }

        return report
