"""Multi-Color Highlight Detection for Bank Statements

This module implements InternVL's advanced highlight detection capabilities
for identifying highlighted regions in bank statements and other documents.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class HighlightRegion:
    """Represents a detected highlight region."""

    x: int
    y: int
    width: int
    height: int
    color: str
    confidence: float
    area: int
    text_content: str | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x, y, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> tuple[int, int]:
        """Return center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class HighlightDetector:
    """Advanced highlight detection for bank statements and documents.

    Features:
    - Multi-color highlight detection (yellow, green, pink, blue)
    - Noise filtering and region merging
    - Confidence scoring
    - Bank statement optimized detection
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # Color ranges for different highlight types (HSV)
        self.color_ranges = {
            "yellow": {
                "lower": np.array([15, 50, 50]),
                "upper": np.array([35, 255, 255]),
                "name": "yellow",
            },
            "green": {
                "lower": np.array([35, 50, 50]),
                "upper": np.array([85, 255, 255]),
                "name": "green",
            },
            "pink": {
                "lower": np.array([140, 50, 50]),
                "upper": np.array([170, 255, 255]),
                "name": "pink",
            },
            "blue": {
                "lower": np.array([90, 50, 50]),
                "upper": np.array([130, 255, 255]),
                "name": "blue",
            },
        }

        # Detection parameters
        self.detection_params = {
            "min_area": 100,  # Minimum area for a valid highlight
            "max_area": 50000,  # Maximum area for a valid highlight
            "min_confidence": 0.3,  # Minimum confidence score
            "merge_threshold": 10,  # Distance threshold for merging nearby regions
            "noise_filter_size": 3,  # Kernel size for noise filtering
            "dilate_iterations": 2,  # Dilation iterations for region enhancement
            "erode_iterations": 1,  # Erosion iterations for cleanup
        }

    def initialize(self) -> None:
        """Initialize the highlight detector."""
        if self.initialized:
            return

        # Load configuration overrides
        if hasattr(self.config, "highlight_detection_params"):
            self.detection_params.update(self.config.highlight_detection_params)

        logger.info("HighlightDetector initialized with multi-color detection")
        self.initialized = True

    def detect_highlights(
        self,
        image_path: str | Path | Image.Image,
    ) -> list[HighlightRegion]:
        """Detect highlights in an image.

        Args:
            image_path: Path to image file or PIL Image

        Returns:
            List of detected highlight regions

        """
        if not self.initialized:
            self.initialize()

        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return []
            elif isinstance(image_path, Image.Image):
                image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
            else:
                logger.error(f"Unsupported image type: {type(image_path)}")
                return []

            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            all_highlights = []

            # Detect highlights for each color
            for _color_name, color_info in self.color_ranges.items():
                highlights = self._detect_color_highlights(hsv, color_info)
                all_highlights.extend(highlights)

            # Filter and merge overlapping regions
            filtered_highlights = self._filter_and_merge_highlights(all_highlights)

            # Calculate confidence scores
            final_highlights = self._calculate_confidence_scores(
                filtered_highlights,
                image,
            )

            logger.info(f"Detected {len(final_highlights)} highlight regions")
            return final_highlights

        except Exception as e:
            logger.error(f"Error detecting highlights: {e}")
            return []

    def _detect_color_highlights(
        self,
        hsv_image: np.ndarray,
        color_info: dict[str, Any],
    ) -> list[HighlightRegion]:
        """Detect highlights of a specific color."""
        # Create color mask
        mask = cv2.inRange(hsv_image, color_info["lower"], color_info["upper"])

        # Apply morphological operations to clean up the mask
        kernel = np.ones(
            (
                self.detection_params["noise_filter_size"],
                self.detection_params["noise_filter_size"],
            ),
            np.uint8,
        )

        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Dilate to enhance regions
        mask = cv2.dilate(
            mask,
            kernel,
            iterations=self.detection_params["dilate_iterations"],
        )

        # Erode to clean up
        mask = cv2.erode(
            mask,
            kernel,
            iterations=self.detection_params["erode_iterations"],
        )

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlights = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by area
            if (
                self.detection_params["min_area"]
                <= area
                <= self.detection_params["max_area"]
            ):
                highlight = HighlightRegion(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    color=color_info["name"],
                    confidence=0.5,  # Will be calculated later
                    area=int(area),
                )
                highlights.append(highlight)

        return highlights

    def _filter_and_merge_highlights(
        self,
        highlights: list[HighlightRegion],
    ) -> list[HighlightRegion]:
        """Filter and merge overlapping highlight regions."""
        if not highlights:
            return []

        # Sort by area (largest first)
        highlights.sort(key=lambda h: h.area, reverse=True)

        merged_highlights = []
        used_indices = set()

        for i, highlight in enumerate(highlights):
            if i in used_indices:
                continue

            # Find nearby highlights to merge
            merge_group = [highlight]
            used_indices.add(i)

            for j, other_highlight in enumerate(highlights[i + 1 :], i + 1):
                if j in used_indices:
                    continue

                # Check if highlights are close enough to merge
                if self._should_merge_highlights(highlight, other_highlight):
                    merge_group.append(other_highlight)
                    used_indices.add(j)

            # Merge the group
            if len(merge_group) == 1:
                merged_highlights.append(merge_group[0])
            else:
                merged_highlight = self._merge_highlight_group(merge_group)
                merged_highlights.append(merged_highlight)

        return merged_highlights

    def _should_merge_highlights(
        self,
        h1: HighlightRegion,
        h2: HighlightRegion,
    ) -> bool:
        """Check if two highlights should be merged."""
        # Calculate distance between centers
        center1 = h1.center
        center2 = h2.center
        distance = np.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2,
        )

        # Check if they overlap or are very close
        threshold = self.detection_params["merge_threshold"]
        return distance < threshold or self._rectangles_overlap(h1.bbox, h2.bbox)

    def _rectangles_overlap(
        self,
        rect1: tuple[int, int, int, int],
        rect2: tuple[int, int, int, int],
    ) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2

        return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

    def _merge_highlight_group(
        self,
        highlights: list[HighlightRegion],
    ) -> HighlightRegion:
        """Merge a group of highlights into a single region."""
        # Find bounding box that encompasses all highlights
        min_x = min(h.x for h in highlights)
        min_y = min(h.y for h in highlights)
        max_x = max(h.x + h.width for h in highlights)
        max_y = max(h.y + h.height for h in highlights)

        # Calculate merged properties
        total_area = sum(h.area for h in highlights)
        avg_confidence = sum(h.confidence for h in highlights) / len(highlights)

        # Use the most common color
        colors = [h.color for h in highlights]
        merged_color = max(set(colors), key=colors.count)

        return HighlightRegion(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            color=merged_color,
            confidence=avg_confidence,
            area=total_area,
        )

    def _calculate_confidence_scores(
        self,
        highlights: list[HighlightRegion],
        image: np.ndarray,
    ) -> list[HighlightRegion]:
        """Calculate confidence scores for detected highlights."""
        if not highlights:
            return []

        # Calculate confidence based on various factors
        for highlight in highlights:
            confidence = 0.0

            # Factor 1: Area (medium-sized regions are most likely highlights)
            area_score = self._calculate_area_score(highlight.area)
            confidence += area_score * 0.3

            # Factor 2: Aspect ratio (reasonable width/height ratio)
            aspect_ratio = highlight.width / highlight.height
            aspect_score = self._calculate_aspect_score(aspect_ratio)
            confidence += aspect_score * 0.2

            # Factor 3: Color intensity (well-defined color regions)
            color_score = self._calculate_color_score(highlight, image)
            confidence += color_score * 0.3

            # Factor 4: Position (bank statements often have highlights in specific areas)
            position_score = self._calculate_position_score(highlight, image.shape)
            confidence += position_score * 0.2

            # Update confidence
            highlight.confidence = max(0.0, min(1.0, confidence))

        # Filter by minimum confidence
        filtered_highlights = [
            h
            for h in highlights
            if h.confidence >= self.detection_params["min_confidence"]
        ]

        return filtered_highlights

    def _calculate_area_score(self, area: int) -> float:
        """Calculate score based on area (medium areas are preferred)."""
        optimal_area = 1000  # Optimal area for highlights
        max_area = self.detection_params["max_area"]

        if area < optimal_area:
            return area / optimal_area
        return max(0.0, 1.0 - (area - optimal_area) / (max_area - optimal_area))

    def _calculate_aspect_score(self, aspect_ratio: float) -> float:
        """Calculate score based on aspect ratio."""
        # Prefer roughly rectangular regions (not too thin or too wide)
        if 0.2 <= aspect_ratio <= 5.0:
            return 1.0
        if 0.1 <= aspect_ratio <= 10.0:
            return 0.5
        return 0.1

    def _calculate_color_score(
        self,
        highlight: HighlightRegion,
        image: np.ndarray,
    ) -> float:
        """Calculate score based on color consistency."""
        # Extract region from image
        region = image[
            highlight.y : highlight.y + highlight.height,
            highlight.x : highlight.x + highlight.width,
        ]

        if region.size == 0:
            return 0.0

        # Convert to HSV for color analysis
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Calculate color consistency
        std_color = np.std(hsv_region, axis=(0, 1))

        # Lower standard deviation means more consistent color
        consistency_score = max(0.0, 1.0 - np.mean(std_color) / 128.0)

        return consistency_score

    def _calculate_position_score(
        self,
        highlight: HighlightRegion,
        image_shape: tuple[int, int, int],
    ) -> float:
        """Calculate score based on position in image."""
        height, width, _ = image_shape

        # Get relative position
        rel_x = highlight.x / width
        rel_y = highlight.y / height

        # For bank statements, highlights are often in the center-left area
        # This is a heuristic that can be adjusted
        if 0.1 <= rel_x <= 0.8 and 0.2 <= rel_y <= 0.8:
            return 1.0
        return 0.5

    def detect_bank_statement_highlights(
        self,
        image_path: str | Path | Image.Image,
    ) -> list[HighlightRegion]:
        """Specialized highlight detection for bank statements.

        Args:
            image_path: Path to bank statement image

        Returns:
            List of detected highlight regions optimized for bank statements

        """
        # Use standard detection but with bank-specific parameters
        original_params = self.detection_params.copy()

        try:
            # Optimize parameters for bank statements
            self.detection_params.update(
                {
                    "min_area": 200,  # Larger minimum area for bank statement highlights
                    "max_area": 30000,  # Reasonable maximum for statement rows
                    "min_confidence": 0.4,  # Higher confidence threshold
                    "merge_threshold": 5,  # Smaller merge threshold for precise detection
                },
            )

            highlights = self.detect_highlights(image_path)

            # Additional filtering for bank statements
            bank_highlights = []
            for highlight in highlights:
                # Filter by aspect ratio (bank statement highlights are usually wide)
                aspect_ratio = highlight.width / highlight.height
                if 1.5 <= aspect_ratio <= 20.0:  # Wide rectangular regions
                    bank_highlights.append(highlight)

            logger.info(f"Detected {len(bank_highlights)} bank statement highlights")
            return bank_highlights

        finally:
            # Restore original parameters
            self.detection_params = original_params

    def export_highlighted_regions(
        self,
        image_path: str | Path | Image.Image,
        highlights: list[HighlightRegion],
        output_dir: str | Path,
    ) -> list[Path]:
        """Export highlighted regions as separate image files.

        Args:
            image_path: Original image path
            highlights: List of highlight regions
            output_dir: Directory to save extracted regions

        Returns:
            List of paths to exported region images

        """
        if not highlights:
            return []

        try:
            # Load original image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
            elif isinstance(image_path, Image.Image):
                image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
            else:
                return []

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            exported_paths = []

            for i, highlight in enumerate(highlights):
                # Extract region
                region = image[
                    highlight.y : highlight.y + highlight.height,
                    highlight.x : highlight.x + highlight.width,
                ]

                # Generate filename
                filename = f"highlight_{i:03d}_{highlight.color}_{highlight.confidence:.2f}.png"
                output_path = output_dir / filename

                # Save region
                cv2.imwrite(str(output_path), region)
                exported_paths.append(output_path)

            logger.info(
                f"Exported {len(exported_paths)} highlight regions to {output_dir}",
            )
            return exported_paths

        except Exception as e:
            logger.error(f"Error exporting highlight regions: {e}")
            return []
