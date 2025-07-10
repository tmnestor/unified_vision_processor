"""Equipment Supplies Handler

Specialized handler for equipment and supplies receipts following the Llama 7-step pipeline.
"""

import logging
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class EquipmentSuppliesHandler(BaseATOHandler):
    """Handler for equipment and supplies receipts."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "supplier", "total_amount"]
        self.optional_fields = ["items", "model_numbers", "warranty", "serial_numbers"]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (10.0, 100000.0)}

    def _extract_document_specific_fields(self, _text: str) -> dict[str, Any]:
        return {}

    def _validate_document_specific_fields(self, _fields: dict[str, Any]) -> list[str]:
        return []
