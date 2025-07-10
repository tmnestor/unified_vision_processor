"""Parking Toll Handler

Specialized handler for parking and toll receipts following the Llama 7-step pipeline.
"""

import logging
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class ParkingTollHandler(BaseATOHandler):
    """Handler for parking and toll receipts."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "location", "total_amount"]
        self.optional_fields = ["duration", "entry_time", "exit_time", "vehicle_reg"]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (1.0, 200.0)}

    def _extract_document_specific_fields(self, _text: str) -> dict[str, Any]:
        return {}

    def _validate_document_specific_fields(self, _fields: dict[str, Any]) -> list[str]:
        return []
