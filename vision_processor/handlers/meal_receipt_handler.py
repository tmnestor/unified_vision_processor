"""
Meal Receipt Handler

Specialized handler for meal receipts following the Llama 7-step pipeline
with Australian restaurant and food service recognition.
"""

import logging
from typing import Any, Dict, List

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class MealReceiptHandler(BaseATOHandler):
    """Handler for meal receipts with Australian food service expertise."""

    def _load_field_requirements(self) -> None:
        """Load meal receipt specific field requirements."""
        self.required_fields = ["date", "restaurant_name", "total_amount"]
        self.optional_fields = [
            "gst_amount",
            "items",
            "table_number",
            "covers",
            "meal_type",
        ]

    def _load_validation_rules(self) -> None:
        """Load meal receipt validation rules."""
        self.validation_rules = {
            "total_amount_range": (5.0, 1000.0),
            "meal_types": ["breakfast", "lunch", "dinner", "snack"],
        }

    def _extract_document_specific_fields(self, text: str) -> Dict[str, Any]:
        """Extract meal receipt specific fields."""
        return {"restaurant_name": self._extract_australian_business_name(text)}

    def _validate_document_specific_fields(self, _fields: Dict[str, Any]) -> List[str]:
        """Validate meal receipt specific fields."""
        return []
