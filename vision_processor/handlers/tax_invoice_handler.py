"""Tax Invoice Handler

Specialized handler for tax invoices following the Llama 7-step pipeline
with Australian tax compliance and professional services recognition.
"""

import logging
import re
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class TaxInvoiceHandler(BaseATOHandler):
    """Handler for tax invoices with Australian tax compliance expertise.

    ATO Requirements for Tax Invoices:
    - Words "Tax Invoice" prominently displayed
    - Identity and ABN of supplier
    - Date of issue
    - Brief description of goods/services
    - GST amount (if applicable)
    - Total amount payable
    - Extent to which goods/services are taxable
    """

    def _load_field_requirements(self) -> None:
        """Load tax invoice specific field requirements."""
        self.required_fields = [
            "supplier_name",
            "supplier_abn",
            "invoice_date",
            "invoice_number",
            "total_amount",
            "gst_amount",
        ]

        self.optional_fields = [
            "customer_name",
            "customer_abn",
            "description",
            "subtotal",
            "due_date",
            "payment_terms",
            "supplier_address",
            "tax_invoice_indicator",
        ]

    def _load_validation_rules(self) -> None:
        """Load tax invoice validation rules."""
        self.validation_rules = {
            "total_amount_range": (1.0, 100000.0),
            "gst_rate": 0.10,
            "gst_tolerance": 0.05,
            "required_text": ["tax invoice", "gst invoice"],
        }

    def _extract_document_specific_fields(self, text: str) -> dict[str, Any]:
        """Extract tax invoice specific fields."""
        fields = {}

        # Extract supplier information
        supplier_name = self._extract_supplier_name(text)
        if supplier_name:
            fields["supplier_name"] = supplier_name

        supplier_abn = self._extract_supplier_abn(text)
        if supplier_abn:
            fields["supplier_abn"] = supplier_abn

        # Extract invoice details
        invoice_number = self._extract_invoice_number(text)
        if invoice_number:
            fields["invoice_number"] = invoice_number

        invoice_date = self._extract_invoice_date(text)
        if invoice_date:
            fields["invoice_date"] = invoice_date

        # Extract customer information
        customer_name = self._extract_customer_name(text)
        if customer_name:
            fields["customer_name"] = customer_name

        # Extract description
        description = self._extract_description(text)
        if description:
            fields["description"] = description

        # Extract subtotal
        subtotal = self._extract_subtotal(text)
        if subtotal:
            fields["subtotal"] = subtotal

        # Extract due date
        due_date = self._extract_due_date(text)
        if due_date:
            fields["due_date"] = due_date

        return fields

    def _extract_supplier_name(self, text: str) -> str:
        """Extract supplier name."""
        # Look for business name patterns at the top of the invoice
        lines = text.split("\n")
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and not re.match(r"^\d|^tax\s*invoice", line.lower()):
                return line
        return ""

    def _extract_supplier_abn(self, text: str) -> str:
        """Extract supplier ABN."""
        abn_patterns = [
            r"(?:abn|australian business number)\s*:?\s*(\d{2}\s+\d{3}\s+\d{3}\s+\d{3})",
            r"(?:abn|australian business number)\s*:?\s*(\d{11})",
        ]

        for pattern in abn_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_invoice_number(self, text: str) -> str:
        """Extract invoice number."""
        invoice_patterns = [
            r"(?:invoice|inv)\s*(?:no\.?|number|#)\s*:?\s*([A-Za-z0-9\-]+)",
            r"(?:tax\s*)?invoice\s*:?\s*([A-Za-z0-9\-]+)",
        ]

        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_invoice_date(self, text: str) -> str:
        """Extract invoice date."""
        date_patterns = [
            r"(?:invoice\s*date|date\s*of\s*issue)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:issued|date)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_customer_name(self, text: str) -> str:
        """Extract customer name."""
        customer_patterns = [
            r"(?:bill\s*to|customer|client)\s*:?\s*([A-Za-z\s&]+)",
            r"(?:to|for)\s*:?\s*([A-Za-z\s&]+)",
        ]

        for pattern in customer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_description(self, text: str) -> str:
        """Extract service/goods description."""
        description_patterns = [
            r"(?:description|services?|goods)\s*:?\s*([A-Za-z\s,.-]+)",
            r"(?:for|re)\s*:?\s*([A-Za-z\s,.-]+)",
        ]

        for pattern in description_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_subtotal(self, text: str) -> float:
        """Extract subtotal amount."""
        subtotal_patterns = [
            r"(?:subtotal|sub.?total|net)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"(?:amount\s*before\s*gst)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
        ]

        for pattern in subtotal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

    def _extract_due_date(self, text: str) -> str:
        """Extract payment due date."""
        due_patterns = [
            r"(?:due\s*date|payment\s*due)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"(?:payable\s*by|due)\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        ]

        for pattern in due_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _validate_document_specific_fields(self, fields: dict[str, Any]) -> list[str]:
        """Validate tax invoice specific fields."""
        issues = []

        # Check for "Tax Invoice" text
        if "tax_invoice_indicator" not in fields:
            issues.append("Document should contain 'Tax Invoice' text")

        # Validate supplier ABN
        if fields.get("supplier_abn"):
            abn = str(fields["supplier_abn"]).replace(" ", "")
            if len(abn) != 11 or not abn.isdigit():
                issues.append("Invalid supplier ABN format")

        # Validate GST calculation
        if all(field in fields and fields[field] for field in ["subtotal", "gst_amount", "total_amount"]):
            try:
                subtotal = float(fields["subtotal"])
                gst_amount = float(fields["gst_amount"])
                total_amount = float(fields["total_amount"])

                expected_gst = subtotal * self.validation_rules["gst_rate"]
                expected_total = subtotal + gst_amount

                if abs(gst_amount - expected_gst) > self.validation_rules["gst_tolerance"]:
                    issues.append(
                        f"GST amount may be incorrect (expected ${expected_gst:.2f})",
                    )

                if abs(total_amount - expected_total) > self.validation_rules["gst_tolerance"]:
                    issues.append(
                        f"Total amount may be incorrect (expected ${expected_total:.2f})",
                    )

            except (ValueError, TypeError):
                issues.append("Cannot validate GST calculation due to invalid values")

        return issues

    def enhance_with_highlights(
        self,
        fields: dict[str, Any],
        highlights: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Enhance tax invoice extraction using InternVL highlight detection.

        Tax invoices benefit from highlight detection for:
        - Total amount and GST highlighting (critical for compliance)
        - ABN and supplier details highlighting
        - Invoice number and date highlighting
        - Service description highlighting
        """
        if not highlights or not self.supports_highlights:
            return fields

        enhanced_fields = fields.copy()
        logger.info(f"Enhancing tax invoice with {len(highlights)} highlights")

        # Priority order for tax invoice fields
        priority_fields = [
            "total_amount",
            "gst_amount",
            "supplier_abn",
            "invoice_number",
            "invoice_date",
            "subtotal",
        ]

        for highlight in highlights:
            if highlight.get("text"):
                highlight_text = highlight["text"]
                highlight_confidence = highlight.get("confidence", 0.8)

                # Extract tax invoice specific information from highlighted regions
                highlight_fields = self._extract_from_highlight(highlight_text)

                # Merge with enhanced preference for highlighted data
                for field, value in highlight_fields.items():
                    if value and (field not in enhanced_fields or not enhanced_fields[field]):
                        enhanced_fields[field] = value
                        enhanced_fields[f"{field}_highlight_confidence"] = highlight_confidence
                        logger.info(f"Enhanced field {field} from highlight: {value}")
                    elif field in priority_fields and value:
                        # Override existing value if this is a priority field from highlights
                        if highlight_confidence > 0.7:
                            enhanced_fields[field] = value
                            enhanced_fields[f"{field}_highlight_confidence"] = highlight_confidence
                            logger.info(
                                f"Override field {field} from high-confidence highlight: {value}",
                            )

        # Validate GST calculations after highlight enhancement
        enhanced_fields = self._validate_gst_calculations(enhanced_fields)

        # Check for tax invoice indicator in highlights
        enhanced_fields = self._detect_tax_invoice_indicator(
            enhanced_fields,
            highlights,
        )

        return enhanced_fields

    def _extract_from_highlight(self, highlight_text: str) -> dict[str, Any]:
        """Extract tax invoice information from highlighted text region."""
        fields = {}

        # Try all extraction methods on the highlighted text
        supplier_name = self._extract_supplier_name(highlight_text)
        if supplier_name:
            fields["supplier_name"] = supplier_name

        supplier_abn = self._extract_supplier_abn(highlight_text)
        if supplier_abn:
            fields["supplier_abn"] = supplier_abn

        invoice_number = self._extract_invoice_number(highlight_text)
        if invoice_number:
            fields["invoice_number"] = invoice_number

        invoice_date = self._extract_invoice_date(highlight_text)
        if invoice_date:
            fields["invoice_date"] = invoice_date

        # Look for amount patterns specifically in highlights
        amount_patterns = [
            (
                r"(?:total|amount due|amount payable)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
                "total_amount",
            ),
            (r"(?:gst|tax)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)", "gst_amount"),
            (r"(?:subtotal|sub.?total|net)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)", "subtotal"),
            (r"\$\s*(\d+(?:\.\d{2})?)", "amount"),  # Generic amount
        ]

        for pattern, field_name in amount_patterns:
            match = re.search(pattern, highlight_text, re.IGNORECASE)
            if match:
                try:
                    amount = float(match.group(1))
                    if field_name == "amount":
                        # For generic amounts, try to determine context
                        if "total" in highlight_text.lower():
                            fields["total_amount"] = amount
                        elif "gst" in highlight_text.lower() or "tax" in highlight_text.lower():
                            fields["gst_amount"] = amount
                        elif "sub" in highlight_text.lower():
                            fields["subtotal"] = amount
                    else:
                        fields[field_name] = amount
                    break
                except ValueError:
                    continue

        # Extract description from highlights
        description = self._extract_description(highlight_text)
        if description:
            fields["description"] = description

        # Extract customer information from highlights
        customer_name = self._extract_customer_name(highlight_text)
        if customer_name:
            fields["customer_name"] = customer_name

        return fields

    def _validate_gst_calculations(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Validate and correct GST calculations after highlight enhancement."""
        # If we have subtotal and total, calculate GST
        if (
            "subtotal" in fields
            and fields["subtotal"]
            and "total_amount" in fields
            and fields["total_amount"]
        ):
            try:
                subtotal = float(fields["subtotal"])
                total_amount = float(fields["total_amount"])
                calculated_gst = total_amount - subtotal

                # If we don't have GST or it's significantly wrong, use calculated
                if "gst_amount" not in fields or not fields["gst_amount"]:
                    fields["gst_amount"] = calculated_gst
                    fields["calculated_gst"] = True
                    logger.info(
                        f"Calculated GST from highlights: ${calculated_gst:.2f}",
                    )
                else:
                    existing_gst = float(fields["gst_amount"])
                    difference = abs(calculated_gst - existing_gst)

                    # If highlighted calculation is more accurate, use it
                    if difference > 0.10 and calculated_gst > 0:
                        # Validate against 10% GST rate
                        expected_gst = subtotal * 0.10
                        if abs(calculated_gst - expected_gst) < abs(
                            existing_gst - expected_gst,
                        ):
                            fields["gst_amount"] = calculated_gst
                            fields["corrected_gst"] = True
                            logger.info(
                                f"Corrected GST using calculation: ${calculated_gst:.2f}",
                            )

            except (ValueError, TypeError):
                logger.warning("Could not validate GST calculations from highlights")

        # If we have subtotal and GST, calculate total
        elif (
            "subtotal" in fields and fields["subtotal"] and "gst_amount" in fields and fields["gst_amount"]
        ):
            try:
                subtotal = float(fields["subtotal"])
                gst_amount = float(fields["gst_amount"])
                calculated_total = subtotal + gst_amount

                if "total_amount" not in fields or not fields["total_amount"]:
                    fields["total_amount"] = calculated_total
                    fields["calculated_total"] = True
                    logger.info(
                        f"Calculated total from highlights: ${calculated_total:.2f}",
                    )

            except (ValueError, TypeError):
                logger.warning("Could not calculate total from subtotal and GST")

        return fields

    def _detect_tax_invoice_indicator(
        self,
        fields: dict[str, Any],
        highlights: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect if 'Tax Invoice' text appears in highlights."""
        for highlight in highlights:
            if highlight.get("text"):
                highlight_text = highlight["text"].lower()
                if "tax invoice" in highlight_text or "gst invoice" in highlight_text:
                    fields["tax_invoice_indicator"] = True
                    fields["tax_invoice_confidence"] = highlight.get("confidence", 0.8)
                    logger.info("Tax Invoice indicator found in highlights")
                    break

        return fields

    def _apply_enhanced_parsing(self, text: str) -> dict[str, Any]:
        """Apply InternVL enhanced parsing techniques for tax invoices."""
        if not self.supports_enhanced_parsing:
            return {}

        enhanced_fields = {}

        # Enhanced ABN validation and formatting
        abn = self._enhanced_abn_extraction(text)
        if abn:
            enhanced_fields["supplier_abn"] = abn

        # Enhanced business name recognition
        business_name = self._enhanced_business_recognition(text)
        if business_name:
            enhanced_fields["supplier_name"] = business_name

        # Enhanced professional services detection
        service_type = self._detect_professional_service_type(text)
        if service_type:
            enhanced_fields["service_type"] = service_type

        # Enhanced payment terms extraction
        payment_terms = self._extract_payment_terms(text)
        if payment_terms:
            enhanced_fields["payment_terms"] = payment_terms

        # Enhanced supplier address extraction
        supplier_address = self._extract_supplier_address(text)
        if supplier_address:
            enhanced_fields["supplier_address"] = supplier_address

        return enhanced_fields

    def _enhanced_abn_extraction(self, text: str) -> str:
        """Enhanced ABN extraction with validation."""
        # Standard extraction first
        abn = self._extract_supplier_abn(text)
        if abn:
            # Format and validate ABN
            abn_digits = re.sub(r"\D", "", abn)
            if len(abn_digits) == 11:
                # Format as XX XXX XXX XXX
                formatted_abn = f"{abn_digits[:2]} {abn_digits[2:5]} {abn_digits[5:8]} {abn_digits[8:]}"
                return formatted_abn

        # Try additional patterns
        enhanced_patterns = [
            r"(?:australian business number|a\.?b\.?n\.?)\s*:?\s*(\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3})",
            r"(?:abn|australian business number)\s*[:#-]?\s*(\d{11})",
        ]

        for pattern in enhanced_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                abn_digits = re.sub(r"\D", "", match.group(1))
                if len(abn_digits) == 11:
                    return f"{abn_digits[:2]} {abn_digits[2:5]} {abn_digits[5:8]} {abn_digits[8:]}"

        return ""

    def _enhanced_business_recognition(self, text: str) -> str:
        """Enhanced business name recognition with Australian business knowledge."""
        # Standard extraction first
        business_name = self._extract_supplier_name(text)
        if business_name:
            return business_name

        # Look for professional services patterns
        professional_patterns = [
            r"([A-Za-z\s&]+)\s+(?:lawyers?|solicitors?|barristers?)",
            r"([A-Za-z\s&]+)\s+(?:accountants?|accounting)",
            r"([A-Za-z\s&]+)\s+(?:consultants?|consulting)",
            r"([A-Za-z\s&]+)\s+(?:pty\.?\s*ltd\.?|limited|corp\.?)",
        ]

        for pattern in professional_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _detect_professional_service_type(self, text: str) -> str:
        """Detect the type of professional service."""
        service_types = {
            "legal": ["lawyer", "solicitor", "barrister", "legal", "law firm"],
            "accounting": ["accountant", "accounting", "bookkeeper", "tax agent"],
            "consulting": ["consultant", "consulting", "advisory", "advisor"],
            "engineering": ["engineer", "engineering", "technical"],
            "medical": ["doctor", "medical", "dental", "clinic"],
            "it": ["technology", "software", "it services", "computer"],
        }

        text_lower = text.lower()
        for service_type, keywords in service_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return service_type

        return ""

    def _extract_payment_terms(self, text: str) -> str:
        """Extract payment terms from invoice."""
        terms_patterns = [
            r"(?:payment terms|terms)\s*:?\s*([A-Za-z0-9\s]+)",
            r"(?:net|due in)\s+(\d+)\s*days?",
            r"payment\s+due\s+on\s+receipt",
        ]

        for pattern in terms_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "receipt" in match.group(0).lower():
                    return "Due on receipt"
                if "days" in match.group(0).lower():
                    return f"Net {match.group(1)} days"
                return match.group(1).strip()

        return ""

    def _extract_supplier_address(self, text: str) -> str:
        """Extract supplier address with Australian address patterns."""
        lines = text.split("\n")
        address_lines = []

        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()

            # Look for Australian address patterns
            if re.search(r"\b\d{4}\b", line):  # Contains postcode
                # Check surrounding lines for address components
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 2)

                for addr_line in lines[start_idx:end_idx]:
                    addr_line = addr_line.strip()
                    if (
                        addr_line
                        and len(addr_line) > 5
                        and not re.match(r"^(?:abn|phone|email|fax)", addr_line.lower())
                    ):
                        address_lines.append(addr_line)

                if address_lines:
                    return ", ".join(address_lines[:3])  # Limit to 3 lines

        return ""
