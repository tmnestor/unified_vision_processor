"""
Other Document Handler

Fallback handler for unclassified documents following the Llama 7-step pipeline.
"""

import logging
from typing import Any, Dict, List

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class OtherDocumentHandler(BaseATOHandler):
    """Fallback handler for unclassified documents."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "total_amount"]
        self.optional_fields = ["business_name", "description", "reference"]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (0.01, 100000.0)}

    def _extract_document_specific_fields(self, text: str) -> Dict[str, Any]:
        return {"business_name": self._extract_australian_business_name(text)}

    def _validate_document_specific_fields(self, _fields: Dict[str, Any]) -> List[str]:
        return []
