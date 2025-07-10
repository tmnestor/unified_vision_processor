"""OCR Processor for Highlighted Regions

This module provides OCR capabilities specifically optimized for processing
text from highlighted regions detected by the HighlightDetector.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .highlight_detector import HighlightRegion

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float
    bbox: tuple
    highlight_region: HighlightRegion | None = None
    preprocessed: bool = False

    def __post_init__(self):
        """Clean up extracted text."""
        self.text = self.text.strip()


class OCRProcessor:
    """OCR processor optimized for highlighted regions.

    Features:
    - Image preprocessing for better OCR accuracy
    - Multiple OCR engine support (tesseract, easyocr)
    - Confidence scoring and filtering
    - Bank statement text recognition optimization
    """

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

        # OCR configuration
        self.ocr_config = {
            "engine": "tesseract",  # tesseract | easyocr | both
            "min_confidence": 0.5,
            "languages": ["eng"],
            "tesseract_config": "--oem 3 --psm 6",  # OCR Engine Mode and Page Segmentation Mode
            "preprocessing": True,
            "deskew": True,
            "denoise": True,
        }

        # Text filtering patterns
        self.text_filters = {
            "min_length": 2,
            "max_length": 200,
            "exclude_patterns": [
                r"^[^a-zA-Z0-9]*$",  # Only special characters
                r"^\s*$",  # Only whitespace
            ],
            "include_patterns": [
                r"[a-zA-Z0-9]",  # Must contain alphanumeric characters
            ],
        }

        self.ocr_engine = None

    def initialize(self) -> None:
        """Initialize OCR processor."""
        if self.initialized:
            return

        # Load configuration overrides
        if hasattr(self.config, "ocr_config"):
            self.ocr_config.update(self.config.ocr_config)

        # Initialize OCR engine
        self._initialize_ocr_engine()

        logger.info(f"OCRProcessor initialized with {self.ocr_config['engine']} engine")
        self.initialized = True

    def _initialize_ocr_engine(self) -> None:
        """Initialize the selected OCR engine."""
        engine = self.ocr_config["engine"]

        if engine == "tesseract":
            self._initialize_tesseract()
        elif engine == "easyocr":
            self._initialize_easyocr()
        elif engine == "both":
            self._initialize_tesseract()
            self._initialize_easyocr()
        else:
            logger.warning(f"Unknown OCR engine: {engine}, defaulting to tesseract")
            self._initialize_tesseract()

    def _initialize_tesseract(self) -> None:
        """Initialize Tesseract OCR engine."""
        try:
            import pytesseract

            self.tesseract = pytesseract
            logger.info("Tesseract OCR engine initialized")
        except ImportError:
            logger.warning(
                "pytesseract not available, OCR functionality will be limited",
            )
            self.tesseract = None

    def _initialize_easyocr(self) -> None:
        """Initialize EasyOCR engine."""
        try:
            import easyocr

            self.easyocr = easyocr.Reader(self.ocr_config["languages"])
            logger.info("EasyOCR engine initialized")
        except ImportError:
            logger.warning("easyocr not available, using tesseract only")
            self.easyocr = None

    def process_highlighted_regions(
        self,
        image_path: str | Path | Image.Image,
        highlights: list[HighlightRegion],
    ) -> list[OCRResult]:
        """Process OCR on highlighted regions.

        Args:
            image_path: Path to original image
            highlights: List of highlight regions to process

        Returns:
            List of OCR results for each highlight

        """
        if not self.initialized:
            self.initialize()

        if not highlights:
            return []

        try:
            # Load original image
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

            ocr_results = []

            for highlight in highlights:
                result = self._process_single_highlight(image, highlight)
                if result:
                    ocr_results.append(result)

            logger.info(f"Processed OCR for {len(ocr_results)} highlight regions")
            return ocr_results

        except Exception as e:
            logger.error(f"Error processing highlighted regions: {e}")
            return []

    def _process_single_highlight(
        self,
        image: np.ndarray,
        highlight: HighlightRegion,
    ) -> OCRResult | None:
        """Process OCR for a single highlight region."""
        try:
            # Extract region from image
            region = image[
                highlight.y : highlight.y + highlight.height,
                highlight.x : highlight.x + highlight.width,
            ]

            if region.size == 0:
                return None

            # Preprocess region for better OCR
            if self.ocr_config["preprocessing"]:
                processed_region = self._preprocess_region(region)
                preprocessed = True
            else:
                processed_region = region
                preprocessed = False

            # Run OCR
            text, confidence = self._run_ocr(processed_region)

            # Filter and validate text
            if not self._is_valid_text(text):
                return None

            return OCRResult(
                text=text,
                confidence=confidence,
                bbox=highlight.bbox,
                highlight_region=highlight,
                preprocessed=preprocessed,
            )

        except Exception as e:
            logger.error(f"Error processing highlight region: {e}")
            return None

    def _preprocess_region(self, region: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR accuracy."""
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Resize if too small
        height, width = gray.shape
        if height < 30 or width < 30:
            scale_factor = max(30 / height, 30 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(
                gray,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC,
            )

        # Deskew if enabled
        if self.ocr_config["deskew"]:
            gray = self._deskew_image(gray)

        # Denoise if enabled
        if self.ocr_config["denoise"]:
            gray = cv2.fastNlMeansDenoising(gray)

        # Enhance contrast
        gray = self._enhance_contrast(gray)

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image to improve OCR accuracy."""
        try:
            # Find skew angle
            coords = np.column_stack(np.where(image > 0))
            if len(coords) < 10:  # Not enough points
                return image

            angle = cv2.minAreaRect(coords)[-1]

            # Correct angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Only correct if angle is significant
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return rotated

            return image

        except Exception:
            # If deskewing fails, return original
            return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def _run_ocr(self, image: np.ndarray) -> tuple[str, float]:
        """Run OCR on preprocessed image."""
        engine = self.ocr_config["engine"]

        if engine == "tesseract" and self.tesseract:
            return self._run_tesseract_ocr(image)
        if engine == "easyocr" and self.easyocr:
            return self._run_easyocr(image)
        if engine == "both":
            # Try both engines and use the result with higher confidence
            tesseract_result = (
                self._run_tesseract_ocr(image) if self.tesseract else ("", 0.0)
            )
            easyocr_result = self._run_easyocr(image) if self.easyocr else ("", 0.0)

            if tesseract_result[1] >= easyocr_result[1]:
                return tesseract_result
            return easyocr_result
        logger.warning("No OCR engine available")
        return "", 0.0

    def _run_tesseract_ocr(self, image: np.ndarray) -> tuple[str, float]:
        """Run Tesseract OCR."""
        if not self.tesseract:
            return "", 0.0

        try:
            # Get OCR data with confidence scores
            data = self.tesseract.image_to_data(
                image,
                config=self.ocr_config["tesseract_config"],
                output_type=self.tesseract.Output.DICT,
            )

            # Filter out low-confidence words
            confidences = data["conf"]
            words = data["text"]

            filtered_words = []
            total_confidence = 0
            valid_words = 0

            for _i, (word, conf) in enumerate(zip(words, confidences, strict=False)):
                if conf > 0 and word.strip():  # Valid confidence and non-empty word
                    if (
                        conf >= self.ocr_config["min_confidence"] * 100
                    ):  # Tesseract uses 0-100 scale
                        filtered_words.append(word.strip())
                        total_confidence += conf
                        valid_words += 1

            if valid_words > 0:
                text = " ".join(filtered_words)
                avg_confidence = total_confidence / (
                    valid_words * 100
                )  # Normalize to 0-1
                return text, avg_confidence
            return "", 0.0

        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return "", 0.0

    def _run_easyocr(self, image: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR."""
        if not self.easyocr:
            return "", 0.0

        try:
            results = self.easyocr.readtext(image)

            if not results:
                return "", 0.0

            # Combine all detected text with confidence weighting
            all_text = []
            total_confidence = 0
            total_length = 0

            for _bbox, text, confidence in results:
                if confidence >= self.ocr_config["min_confidence"] and text.strip():
                    all_text.append(text.strip())
                    total_confidence += confidence * len(text)
                    total_length += len(text)

            if all_text and total_length > 0:
                combined_text = " ".join(all_text)
                avg_confidence = total_confidence / total_length
                return combined_text, avg_confidence
            return "", 0.0

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0

    def _is_valid_text(self, text: str) -> bool:
        """Validate extracted text."""
        if not text or not text.strip():
            return False

        text = text.strip()

        # Check length constraints
        if len(text) < self.text_filters["min_length"]:
            return False
        if len(text) > self.text_filters["max_length"]:
            return False

        # Check exclude patterns
        import re

        for pattern in self.text_filters["exclude_patterns"]:
            if re.match(pattern, text):
                return False

        # Check include patterns
        has_required = False
        for pattern in self.text_filters["include_patterns"]:
            if re.search(pattern, text):
                has_required = True
                break

        return has_required

    def process_bank_statement_highlights(
        self,
        image_path: str | Path | Image.Image,
        highlights: list[HighlightRegion],
    ) -> dict[str, Any]:
        """Process OCR specifically for bank statement highlights.

        Args:
            image_path: Path to bank statement image
            highlights: List of highlight regions

        Returns:
            Dictionary with extracted bank statement data

        """
        ocr_results = self.process_highlighted_regions(image_path, highlights)

        if not ocr_results:
            return {
                "transactions": [],
                "total_highlights": 0,
                "processed_highlights": 0,
            }

        # Parse bank statement specific information
        transactions = []
        account_info = {}

        for result in ocr_results:
            text = result.text

            # Try to parse as transaction
            transaction = self._parse_transaction_text(text, result)
            if transaction:
                transactions.append(transaction)

            # Try to extract account information
            account_data = self._extract_account_info(text)
            if account_data:
                account_info.update(account_data)

        return {
            "transactions": transactions,
            "account_info": account_info,
            "total_highlights": len(highlights),
            "processed_highlights": len(ocr_results),
            "ocr_results": ocr_results,
        }

    def _parse_transaction_text(
        self,
        text: str,
        ocr_result: OCRResult,
    ) -> dict[str, Any] | None:
        """Parse transaction information from OCR text."""
        import re

        # Common transaction patterns
        amount_pattern = r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
        date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"

        # Find amounts
        amounts = re.findall(amount_pattern, text)
        dates = re.findall(date_pattern, text)

        if amounts:
            # Clean amount (remove commas)
            amount_str = amounts[-1].replace(
                ",",
                "",
            )  # Take last amount (usually the final amount)

            try:
                amount = float(amount_str)

                transaction = {
                    "text": text,
                    "amount": amount,
                    "highlight_region": ocr_result.highlight_region,
                    "confidence": ocr_result.confidence,
                }

                if dates:
                    transaction["date"] = dates[0]

                # Extract description (text without amounts and dates)
                description = text
                for amount in amounts:
                    description = description.replace(f"${amount}", "").replace(
                        amount,
                        "",
                    )
                for date in dates:
                    description = description.replace(date, "")

                transaction["description"] = description.strip()

                return transaction

            except ValueError:
                pass

        return None

    def _extract_account_info(self, text: str) -> dict[str, Any]:
        """Extract account information from OCR text."""
        import re

        account_info = {}

        # BSB pattern (XXX-XXX)
        bsb_pattern = r"(\d{3}-\d{3})"
        bsb_match = re.search(bsb_pattern, text)
        if bsb_match:
            account_info["bsb"] = bsb_match.group(1)

        # Account number pattern
        account_pattern = r"(?:account|acc)[\s:]*(\d{6,12})"
        account_match = re.search(account_pattern, text, re.IGNORECASE)
        if account_match:
            account_info["account_number"] = account_match.group(1)

        # Balance patterns
        balance_pattern = r"(?:balance|bal)[\s:]*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
        balance_match = re.search(balance_pattern, text, re.IGNORECASE)
        if balance_match:
            try:
                balance = float(balance_match.group(1).replace(",", ""))
                account_info["balance"] = balance
            except ValueError:
                pass

        return account_info
