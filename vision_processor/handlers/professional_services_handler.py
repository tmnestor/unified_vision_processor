"""
Professional Services Handler

Specialized handler for professional services invoices following the Llama 7-step pipeline.
"""

import logging
from typing import Any, Dict, List

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class ProfessionalServicesHandler(BaseATOHandler):
    """Handler for professional services invoices."""

    def _load_field_requirements(self) -> None:
        self.required_fields = ["date", "service_provider", "total_amount"]
        self.optional_fields = [
            "service_description",
            "hours",
            "rate",
            "matter_reference",
        ]

    def _load_validation_rules(self) -> None:
        self.validation_rules = {"total_amount_range": (100.0, 50000.0)}

    def _extract_document_specific_fields(self, _text: str) -> Dict[str, Any]:
        return {}

    def _validate_document_specific_fields(self, _fields: Dict[str, Any]) -> List[str]:
        return []
