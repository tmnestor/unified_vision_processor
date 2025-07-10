"""Travel Document Handler

Specialized handler for travel documents following the Llama 7-step pipeline.
"""

import logging
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class TravelDocumentHandler(BaseATOHandler):
    """Handler for travel documents."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "airline", "total_amount"]
        self.optional_fields = ["flight_number", "departure", "arrival", "passenger"]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (50.0, 10000.0)}

    def _extract_document_specific_fields(self, _text: str) -> dict[str, Any]:
        return {}

    def _validate_document_specific_fields(self, _fields: dict[str, Any]) -> list[str]:
        return []
