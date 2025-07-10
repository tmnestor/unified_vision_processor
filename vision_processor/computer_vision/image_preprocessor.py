"""Image Preprocessing and Optimization

This module provides image preprocessing capabilities to optimize images
for better computer vision and OCR performance.
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
class ImageMetadata:
    """Metadata about an image."""

    width: int
    height: int
    channels: int
    file_size: int | None = None
    format: str | None = None
    dpi: tuple[int, int] | None = None
    color_space: str | None = None

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return (self.width * self.height) / 1_000_000


@dataclass
class PreprocessingResult:
    """Result from image preprocessing."""

    processed_image: np.ndarray
    original_metadata: ImageMetadata
    processed_metadata: ImageMetadata
    operations_applied: list[str]
    quality_score: float
    preprocessing_time: float


class ImagePreprocessor:
    """Advanced image preprocessing for computer vision tasks.

    Features:
    - Image quality assessment
    - Noise reduction and enhancement
    - Resolution optimization
    - Format conversion and standardization
    - Document-specific preprocessing
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # Preprocessing configuration
        self.preprocessing_config = {
            "target_dpi": 300,  # Target DPI for OCR
            "max_dimension": 4000,  # Maximum width/height
            "min_dimension": 100,  # Minimum width/height
            "quality_threshold": 0.7,  # Minimum quality score
            "noise_reduction": True,
            "contrast_enhancement": True,
            "sharpening": True,
            "deskewing": True,
            "border_removal": True,
        }

        # Image quality parameters
        self.quality_params = {
            "blur_threshold": 100,  # Laplacian variance threshold
            "brightness_range": (50, 200),  # Acceptable brightness range
            "contrast_threshold": 50,  # Minimum contrast
            "noise_threshold": 0.1,  # Maximum noise level
        }

    def initialize(self) -> None:
        """Initialize image preprocessor."""
        if self.initialized:
            return

        # Load configuration overrides
        if hasattr(self.config, "preprocessing_config"):
            self.preprocessing_config.update(self.config.preprocessing_config)

        if hasattr(self.config, "quality_params"):
            self.quality_params.update(self.config.quality_params)

        logger.info("ImagePreprocessor initialized")
        self.initialized = True

    def preprocess_image(
        self,
        image_path: str | Path | Image.Image,
        target_type: str = "document",
    ) -> PreprocessingResult:
        """Preprocess image for optimal computer vision performance.

        Args:
            image_path: Path to image file or PIL Image
            target_type: Type of processing ("document", "bank_statement", "receipt")

        Returns:
            PreprocessingResult with processed image and metadata

        """
        if not self.initialized:
            self.initialize()

        import time

        start_time = time.time()

        try:
            # Load image and get metadata
            image, original_metadata = self._load_image_with_metadata(image_path)

            operations_applied = []
            processed_image = image.copy()

            # Apply preprocessing based on target type
            if target_type == "document":
                processed_image, ops = self._preprocess_document(processed_image)
                operations_applied.extend(ops)
            elif target_type == "bank_statement":
                processed_image, ops = self._preprocess_bank_statement(processed_image)
                operations_applied.extend(ops)
            elif target_type == "receipt":
                processed_image, ops = self._preprocess_receipt(processed_image)
                operations_applied.extend(ops)
            else:
                processed_image, ops = self._preprocess_general(processed_image)
                operations_applied.extend(ops)

            # Get processed metadata
            processed_metadata = self._get_image_metadata(processed_image)

            # Calculate quality score
            quality_score = self._calculate_quality_score(processed_image)

            processing_time = time.time() - start_time

            return PreprocessingResult(
                processed_image=processed_image,
                original_metadata=original_metadata,
                processed_metadata=processed_metadata,
                operations_applied=operations_applied,
                quality_score=quality_score,
                preprocessing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}") from e

    def _load_image_with_metadata(
        self,
        image_path: str | Path | Image.Image,
    ) -> tuple[np.ndarray, ImageMetadata]:
        """Load image and extract metadata."""
        if isinstance(image_path, (str, Path)):
            # Load with PIL to get metadata
            pil_image = Image.open(image_path)

            # Get file info
            file_size = (
                Path(image_path).stat().st_size if Path(image_path).exists() else None
            )
            format_info = pil_image.format
            dpi = pil_image.info.get("dpi")

            # Convert to OpenCV format
            if pil_image.mode == "RGBA":
                # Convert RGBA to RGB
                rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = rgb_image
            elif pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        elif isinstance(image_path, Image.Image):
            pil_image = image_path
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            file_size = None
            format_info = None
            dpi = pil_image.info.get("dpi")
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")

        # Create metadata
        height, width, channels = cv_image.shape
        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            format=format_info,
            dpi=dpi,
            color_space="BGR",
        )

        return cv_image, metadata

    def _get_image_metadata(self, image: np.ndarray) -> ImageMetadata:
        """Get metadata from OpenCV image."""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        return ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            color_space="BGR" if channels == 3 else "GRAY",
        )

    def _preprocess_document(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Preprocess image for general document processing."""
        operations = []
        processed = image.copy()

        # 1. Resize if needed
        processed, resize_op = self._resize_image(processed)
        if resize_op:
            operations.append(resize_op)

        # 2. Deskew
        if self.preprocessing_config["deskewing"]:
            processed = self._deskew_image(processed)
            operations.append("deskew")

        # 3. Remove borders
        if self.preprocessing_config["border_removal"]:
            processed = self._remove_borders(processed)
            operations.append("border_removal")

        # 4. Enhance contrast
        if self.preprocessing_config["contrast_enhancement"]:
            processed = self._enhance_contrast(processed)
            operations.append("contrast_enhancement")

        # 5. Reduce noise
        if self.preprocessing_config["noise_reduction"]:
            processed = self._reduce_noise(processed)
            operations.append("noise_reduction")

        # 6. Sharpen
        if self.preprocessing_config["sharpening"]:
            processed = self._sharpen_image(processed)
            operations.append("sharpening")

        return processed, operations

    def _preprocess_bank_statement(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Preprocess image specifically for bank statements."""
        operations = []
        processed = image.copy()

        # Bank statements benefit from specific preprocessing
        processed, resize_op = self._resize_image(processed, target_width=2000)
        if resize_op:
            operations.append(resize_op)

        # Enhance contrast for better text visibility
        processed = self._enhance_contrast(processed, strength=1.2)
        operations.append("contrast_enhancement")

        # Reduce noise but preserve fine details
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        operations.append("bilateral_filtering")

        # Slight sharpening for text clarity
        processed = self._sharpen_image(processed, strength=0.8)
        operations.append("sharpening")

        return processed, operations

    def _preprocess_receipt(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Preprocess image specifically for receipts."""
        operations = []
        processed = image.copy()

        # Receipts often need more aggressive preprocessing
        processed, resize_op = self._resize_image(processed)
        if resize_op:
            operations.append(resize_op)

        # Deskew (receipts are often at angles)
        processed = self._deskew_image(processed)
        operations.append("deskew")

        # Strong contrast enhancement
        processed = self._enhance_contrast(processed, strength=1.5)
        operations.append("contrast_enhancement")

        # Noise reduction
        processed = self._reduce_noise(processed)
        operations.append("noise_reduction")

        # Strong sharpening for small text
        processed = self._sharpen_image(processed, strength=1.2)
        operations.append("sharpening")

        return processed, operations

    def _preprocess_general(self, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """General preprocessing for any image type."""
        operations = []
        processed = image.copy()

        # Basic preprocessing
        processed, resize_op = self._resize_image(processed)
        if resize_op:
            operations.append(resize_op)

        processed = self._enhance_contrast(processed)
        operations.append("contrast_enhancement")

        processed = self._reduce_noise(processed)
        operations.append("noise_reduction")

        return processed, operations

    def _resize_image(
        self,
        image: np.ndarray,
        target_width: int | None = None,
    ) -> tuple[np.ndarray, str | None]:
        """Resize image to optimal dimensions."""
        height, width = image.shape[:2]

        max_dim = self.preprocessing_config["max_dimension"]
        min_dim = self.preprocessing_config["min_dimension"]

        # Check if resizing is needed
        if (
            width <= max_dim
            and height <= max_dim
            and width >= min_dim
            and height >= min_dim
        ):
            if target_width is None or abs(width - target_width) < 100:
                return image, None

        # Calculate new dimensions
        if target_width:
            new_width = target_width
            new_height = int(height * (target_width / width))
        # Scale based on max dimension
        elif width > height:
            new_width = min(width, max_dim)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(height, max_dim)
            new_width = int(width * (new_height / height))

        # Ensure minimum dimensions
        if new_width < min_dim:
            new_width = min_dim
            new_height = int(height * (new_width / width))
        if new_height < min_dim:
            new_height = min_dim
            new_width = int(width * (new_height / height))

        # Resize with high-quality interpolation
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

        return resized, f"resize_{width}x{height}_to_{new_width}x{new_height}"

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image to correct rotation."""
        try:
            # Convert to grayscale
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )

            # Find skew angle using Hough line transform
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

            if lines is None:
                return image

            # Calculate average angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                # Convert to [-90, 90] range
                if angle > 90:
                    angle -= 180
                # Only consider significant angles
                if abs(angle) > 1:
                    angles.append(angle)

            if not angles:
                return image

            # Use median angle to avoid outliers
            skew_angle = np.median(angles)

            # Only correct if angle is significant
            if abs(skew_angle) > 0.5:
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

                # Calculate new image size to fit rotated image
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_width = int((height * sin_angle) + (width * cos_angle))
                new_height = int((height * cos_angle) + (width * sin_angle))

                # Adjust translation
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]

                rotated = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (new_width, new_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
                return rotated

            return image

        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove borders/margins from image."""
        try:
            # Convert to grayscale for border detection
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )

            # Find contours
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                return image

            # Find the largest contour (likely the document)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add small margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)

            # Crop image
            cropped = image[y : y + h, x : x + w]

            # Only return cropped if it's significantly smaller
            original_area = image.shape[0] * image.shape[1]
            cropped_area = cropped.shape[0] * cropped.shape[1]

            if cropped_area > 0.7 * original_area:  # At least 70% of original
                return cropped
            return image

        except Exception as e:
            logger.warning(f"Border removal failed: {e}")
            return image

    def _enhance_contrast(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Enhance image contrast."""
        try:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(lightness)

            # Merge channels back
            enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in image."""
        try:
            # Use Non-local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

            return denoised

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image

    def _sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Sharpen image for better text clarity."""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength

            # Normalize kernel
            kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel

            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)

            return sharpened

        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image

    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """Calculate overall quality score for image."""
        try:
            score = 0.0
            weights = {
                "sharpness": 0.3,
                "brightness": 0.2,
                "contrast": 0.3,
                "noise": 0.2,
            }

            # Convert to grayscale for analysis
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )

            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(
                laplacian_var / self.quality_params["blur_threshold"],
                1.0,
            )
            score += sharpness_score * weights["sharpness"]

            # 2. Brightness
            mean_brightness = np.mean(gray)
            brightness_range = self.quality_params["brightness_range"]
            if brightness_range[0] <= mean_brightness <= brightness_range[1]:
                brightness_score = 1.0
            else:
                brightness_score = max(
                    0.0,
                    1.0 - abs(mean_brightness - np.mean(brightness_range)) / 128.0,
                )
            score += brightness_score * weights["brightness"]

            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(
                contrast / self.quality_params["contrast_threshold"],
                1.0,
            )
            score += contrast_score * weights["contrast"]

            # 4. Noise (inverse of standard deviation of Laplacian)
            noise_level = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            noise_score = max(0.0, 1.0 - noise_level / 100.0)  # Normalize to 0-1
            score += noise_score * weights["noise"]

            return min(score, 1.0)

        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Default moderate score

    def assess_image_quality(
        self,
        image_path: str | Path | Image.Image,
    ) -> dict[str, Any]:
        """Assess image quality and provide recommendations."""
        try:
            image, metadata = self._load_image_with_metadata(image_path)
            quality_score = self._calculate_quality_score(image)

            # Convert to grayscale for detailed analysis
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )

            # Detailed metrics
            metrics = {
                "overall_quality": quality_score,
                "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "noise_level": float(np.std(cv2.Laplacian(gray, cv2.CV_64F))),
                "resolution": {"width": metadata.width, "height": metadata.height},
                "aspect_ratio": metadata.aspect_ratio,
                "megapixels": metadata.megapixels,
            }

            # Generate recommendations
            recommendations = []

            if quality_score < self.quality_params["quality_threshold"]:
                recommendations.append("Image quality is below recommended threshold")

            if metrics["sharpness"] < self.quality_params["blur_threshold"]:
                recommendations.append("Image appears blurry - consider resharpening")

            brightness_range = self.quality_params["brightness_range"]
            if not (
                brightness_range[0] <= metrics["brightness"] <= brightness_range[1]
            ):
                if metrics["brightness"] < brightness_range[0]:
                    recommendations.append(
                        "Image is too dark - consider brightness enhancement",
                    )
                else:
                    recommendations.append(
                        "Image is too bright - consider brightness reduction",
                    )

            if metrics["contrast"] < self.quality_params["contrast_threshold"]:
                recommendations.append("Low contrast - consider contrast enhancement")

            if metadata.width < 800 or metadata.height < 600:
                recommendations.append(
                    "Low resolution - consider using higher resolution image",
                )

            return {
                "quality_assessment": metrics,
                "recommendations": recommendations,
                "preprocessing_suggested": len(recommendations) > 0,
            }

        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {"error": str(e)}
