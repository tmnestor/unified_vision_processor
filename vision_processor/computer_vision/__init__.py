"""Computer Vision Module for Advanced Image Processing

This module provides InternVL's advanced computer vision capabilities:
- Multi-color highlight detection
- OCR from highlighted regions
- Bank statement computer vision
- Image preprocessing and optimization
- Spatial correlation between highlights and text
"""

from .bank_statement_cv import BankStatementCV
from .highlight_detector import HighlightDetector
from .image_preprocessor import ImagePreprocessor
from .ocr_processor import OCRProcessor
from .spatial_correlator import SpatialCorrelator

__all__ = [
    "BankStatementCV",
    "HighlightDetector",
    "ImagePreprocessor",
    "OCRProcessor",
    "SpatialCorrelator",
]
