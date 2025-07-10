"""
Accommodation Handler

Specialized handler for accommodation receipts following the Llama 7-step pipeline.
"""

import logging
from typing import Any, Dict, List

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class AccommodationHandler(BaseATOHandler):
    """Handler for accommodation receipts."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "hotel_name", "total_amount"]
        self.optional_fields = ["check_in", "check_out", "nights", "room_number"]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (50.0, 5000.0)}

    def _extract_document_specific_fields(self, _text: str) -> Dict[str, Any]:
        return {}

    def _validate_document_specific_fields(self, _fields: Dict[str, Any]) -> List[str]:
        return []
