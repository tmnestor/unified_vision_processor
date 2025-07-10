"""
Business Receipt Handler

Specialized handler for general business receipts following the Llama 7-step pipeline
with Australian retail chain recognition and item-level extraction.
"""

import logging
import re
from typing import Any, Dict, List

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class BusinessReceiptHandler(BaseATOHandler):
    """
    Handler for general business receipts with Australian retail expertise.

    Supports major Australian retailers:
    - Woolworths, Coles, ALDI, Target, Kmart
    - Bunnings, Officeworks, Harvey Norman, JB Hi-Fi
    - Chemist Warehouse, Priceline, Big W, Myer
    - Electronics, hardware, and general merchandise

    ATO Requirements:
    - Date of purchase
    - Business name and ABN
    - Total amount including GST
    - GST amount (if GST registered)
    - Item descriptions (for business expense validation)
    - Payment method
    """

    def _load_field_requirements(self) -> None:
        """Load business receipt specific field requirements."""
        self.required_fields = [
            "date",
            "business_name",
            "total_amount",
        ]

        self.optional_fields = [
            "gst_amount",
            "subtotal",
            "abn",
            "items",
            "payment_method",
            "receipt_number",
            "store_location",
            "cashier_id",
            "discount_amount",
        ]

    def _load_validation_rules(self) -> None:
        """Load business receipt validation rules."""
        self.validation_rules = {
            "total_amount_range": (0.01, 10000.0),  # Reasonable purchase range
            "gst_rate": 0.10,  # 10% GST in Australia
            "gst_tolerance": 0.05,  # 5 cent tolerance for GST calculations
            "payment_methods": [
                "cash",
                "credit",
                "debit",
                "eftpos",
                "card",
                "paywave",
                "contactless",
                "afterpay",
                "zip",
                "gift card",
            ],
        }

        # Australian retail chain patterns
        self.retail_chain_patterns = {
            "Woolworths": r"\b(?:woolworths|woolies)\b",
            "Coles": r"\bcoles\s*(?:supermarket|express)?\b",
            "ALDI": r"\baldi\s*(?:australia)?\b",
            "Target": r"\btarget\s*(?:australia)?\b",
            "Kmart": r"\bk?mart\b",
            "Bunnings": r"\bbunnings\s*(?:warehouse)?\b",
            "Officeworks": r"\bofficeworks\b",
            "Harvey Norman": r"\bharvey\s*norman\b",
            "JB Hi-Fi": r"\bjb\s*hi.?fi\b",
            "Big W": r"\bbig\s*w\b",
            "Myer": r"\bmyer\b",
            "David Jones": r"\bdavid\s*jones\b",
            "Chemist Warehouse": r"\bchemist\s*warehouse\b",
            "Priceline": r"\bpriceline\s*pharmacy\b",
            "Spotlight": r"\bspotlight\b",
            "Rebel Sport": r"\brebel\s*sport\b",
            "IKEA": r"\bikea\b",
            "Costco": r"\bcostco\b",
        }

    def _extract_document_specific_fields(self, text: str) -> Dict[str, Any]:
        """Extract business receipt specific fields."""
        fields = {}

        # Extract business name using Australian retail patterns
        business_name = self._extract_retail_business_name(text)
        if business_name:
            fields["business_name"] = business_name

        # Extract subtotal
        subtotal = self._extract_subtotal(text)
        if subtotal:
            fields["subtotal"] = subtotal

        # Extract payment method
        payment_method = self._extract_payment_method(text)
        if payment_method:
            fields["payment_method"] = payment_method

        # Extract receipt number
        receipt_number = self._extract_receipt_number(text)
        if receipt_number:
            fields["receipt_number"] = receipt_number

        # Extract store location
        store_location = self._extract_store_location(text)
        if store_location:
            fields["store_location"] = store_location

        # Extract items (if detailed receipt)
        items = self._extract_items(text)
        if items:
            fields["items"] = items

        # Extract discount amount
        discount = self._extract_discount_amount(text)
        if discount:
            fields["discount_amount"] = discount

        return fields

    def _extract_retail_business_name(self, text: str) -> str:
        """Extract Australian retail business name."""
        for business_name, pattern in self.retail_chain_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return business_name

        # Fallback to generic business name extraction
        lines = text.split("\n")
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if len(line) > 2 and not re.match(r"^\d|^[\*\-\=]", line):
                # Potential business name (not starting with number or special chars)
                return line

        return ""

    def _extract_subtotal(self, text: str) -> float:
        """Extract subtotal amount (before GST)."""
        subtotal_patterns = [
            r"(?:subtotal|sub.?total)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"(?:net|nett)\s*(?:total|amount)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"(?:before\s*gst)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
        ]

        for pattern in subtotal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return 0.0

    def _extract_payment_method(self, text: str) -> str:
        """Extract payment method."""
        payment_patterns = [
            r"(?:payment|paid|tender)\s*(?:method|type)?\s*:?\s*(\w+)",
            r"(\w+)\s*(?:card|payment)",
            r"(?:eftpos|credit|debit)\s*(\w*)",
        ]

        # Check for specific payment methods
        for payment_method in self.validation_rules["payment_methods"]:
            if re.search(rf"\b{payment_method}\b", text, re.IGNORECASE):
                return payment_method

        # Try pattern matching
        for pattern in payment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                payment_type = match.group(1).lower()
                if payment_type in [
                    pm.lower() for pm in self.validation_rules["payment_methods"]
                ]:
                    return payment_type

        return ""

    def _extract_receipt_number(self, text: str) -> str:
        """Extract receipt or transaction number."""
        receipt_patterns = [
            r"(?:receipt|transaction|ref|reference)\s*(?:no\.?|number|#)?\s*:?\s*([A-Za-z0-9\-]+)",
            r"(?:txn|trans)\s*(?:no\.?|#)?\s*:?\s*([A-Za-z0-9\-]+)",
            r"(?:invoice|inv)\s*(?:no\.?|#)?\s*:?\s*([A-Za-z0-9\-]+)",
        ]

        for pattern in receipt_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""

    def _extract_store_location(self, text: str) -> str:
        """Extract store location or branch."""
        # Australian states and cities
        location_patterns = [
            r"\b(?:sydney|melbourne|brisbane|perth|adelaide|hobart|canberra|darwin)\b",
            r"\b(?:nsw|vic|qld|wa|sa|tas|act|nt)\b",
            r"\b(\d{4})\b",  # Postcode
            r"(?:store|branch)\s*(?:no\.?|#)?\s*:?\s*(\w+)",
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0) if pattern.endswith(r"\b") else match.group(1)

        return ""

    def _extract_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual items from detailed receipt."""
        items = []
        lines = text.split("\n")

        # Look for item lines (typically have price at the end)
        item_pattern = r"^(.{3,40})\s+\$?(\d+(?:\.\d{2})?)\s*$"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(item_pattern, line)
            if match:
                description = match.group(1).strip()
                price = float(match.group(2))

                # Filter out non-item lines
                if not re.match(
                    r"(?:total|subtotal|gst|tax|payment|change|tender)",
                    description.lower(),
                ):
                    items.append({"description": description, "amount": price})

        return items

    def _extract_discount_amount(self, text: str) -> float:
        """Extract discount amount if any."""
        discount_patterns = [
            r"(?:discount|savings?|off)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"\-\s*\$?\s*(\d+(?:\.\d{2})?)\s*(?:discount|off)",
            r"(?:member|loyalty)\s*(?:discount|savings?)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
        ]

        for pattern in discount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return 0.0

    def _validate_document_specific_fields(self, fields: Dict[str, Any]) -> List[str]:
        """Validate business receipt specific fields."""
        issues = []

        # Validate total amount range
        if "total_amount" in fields and fields["total_amount"]:
            try:
                total = float(fields["total_amount"])
                min_total, max_total = self.validation_rules["total_amount_range"]
                if not (min_total <= total <= max_total):
                    issues.append(
                        f"Total amount ${total:.2f} outside reasonable range (${min_total}-${max_total})"
                    )
            except (ValueError, TypeError):
                issues.append("Invalid total amount format")

        # Validate payment method
        if "payment_method" in fields and fields["payment_method"]:
            payment_method = fields["payment_method"].lower()
            valid_methods = [
                pm.lower() for pm in self.validation_rules["payment_methods"]
            ]
            if payment_method not in valid_methods:
                issues.append(
                    f"Unrecognized payment method: {fields['payment_method']}"
                )

        # Validate GST calculation consistency
        if all(
            field in fields and fields[field]
            for field in ["subtotal", "gst_amount", "total_amount"]
        ):
            try:
                subtotal = float(fields["subtotal"])
                gst_amount = float(fields["gst_amount"])
                total_amount = float(fields["total_amount"])

                # Check if subtotal + GST = total
                calculated_total = subtotal + gst_amount
                total_difference = abs(calculated_total - total_amount)

                if total_difference > self.validation_rules["gst_tolerance"]:
                    issues.append(
                        f"Subtotal + GST (${calculated_total:.2f}) does not equal total (${total_amount:.2f})"
                    )

                # Check if GST is approximately 10% of subtotal
                expected_gst = subtotal * self.validation_rules["gst_rate"]
                gst_difference = abs(gst_amount - expected_gst)

                if gst_difference > self.validation_rules["gst_tolerance"]:
                    issues.append(
                        f"GST amount ${gst_amount:.2f} does not match expected 10% (${expected_gst:.2f})"
                    )

            except (ValueError, TypeError):
                issues.append("Cannot validate GST calculation due to invalid values")

        # Validate items total consistency (if items are present)
        if "items" in fields and fields["items"] and "total_amount" in fields:
            try:
                items_total = sum(item.get("amount", 0) for item in fields["items"])
                receipt_total = float(fields["total_amount"])
                difference = abs(items_total - receipt_total)

                # Allow for taxes, fees, and rounding
                tolerance = max(receipt_total * 0.20, 5.0)  # 20% or $5 tolerance
                if difference > tolerance:
                    issues.append(
                        f"Items total ${items_total:.2f} differs significantly from receipt total ${receipt_total:.2f}"
                    )
            except (ValueError, TypeError):
                issues.append("Cannot validate items total due to invalid values")

        # Validate business name recognition
        if "business_name" in fields and fields["business_name"]:
            business_name = fields["business_name"].lower()
            known_businesses = [
                name.lower() for name in self.retail_chain_patterns.keys()
            ]
            if not any(
                known_business in business_name for known_business in known_businesses
            ):
                # Not necessarily an error, but worth noting
                logger.info(
                    f"Business name '{fields['business_name']}' not in known Australian retailers"
                )

        return issues

    def enhance_with_highlights(
        self, fields: Dict[str, Any], highlights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhance business receipt extraction using InternVL highlight detection.

        Business receipts benefit from highlight detection for:
        - Total amount highlighting (most important for expense claims)
        - GST/tax amount highlighting (critical for ATO compliance)
        - Business name and ABN highlighting
        - Item-level highlighting for detailed expense tracking
        - Receipt number highlighting for reference
        """
        if not highlights or not self.supports_highlights:
            return fields

        enhanced_fields = fields.copy()
        logger.info(f"Enhancing business receipt with {len(highlights)} highlights")

        # Priority order for business receipt fields
        priority_fields = [
            "total_amount",
            "gst_amount",
            "business_name",
            "date",
            "receipt_number",
            "subtotal",
        ]

        highlighted_items = []

        for highlight in highlights:
            if "text" in highlight and highlight["text"]:
                highlight_text = highlight["text"]
                highlight_confidence = highlight.get("confidence", 0.8)

                # Extract business receipt specific information from highlighted regions
                highlight_fields = self._extract_from_highlight(highlight_text)

                # Merge with enhanced preference for highlighted data
                for field, value in highlight_fields.items():
                    if value and (
                        field not in enhanced_fields or not enhanced_fields[field]
                    ):
                        enhanced_fields[field] = value
                        enhanced_fields[f"{field}_highlight_confidence"] = (
                            highlight_confidence
                        )
                        logger.info(f"Enhanced field {field} from highlight: {value}")
                    elif field in priority_fields and value:
                        # Override existing value if this is a priority field from highlights
                        if highlight_confidence > 0.7:
                            enhanced_fields[field] = value
                            enhanced_fields[f"{field}_highlight_confidence"] = (
                                highlight_confidence
                            )
                            logger.info(
                                f"Override field {field} from high-confidence highlight: {value}"
                            )

                # Extract items from highlighted regions
                highlight_items = self._extract_items(highlight_text)
                if highlight_items:
                    # Mark items as highlighted for priority
                    for item in highlight_items:
                        item["highlighted"] = True
                        item["highlight_confidence"] = highlight_confidence
                    highlighted_items.extend(highlight_items)

        # Merge highlighted items with existing items
        if highlighted_items:
            existing_items = enhanced_fields.get("items", [])
            all_items = existing_items + highlighted_items

            # Remove duplicates and prioritize highlighted items
            unique_items = self._deduplicate_items(all_items)
            enhanced_fields["items"] = unique_items

        # Validate calculations after highlight enhancement
        enhanced_fields = self._validate_receipt_calculations(enhanced_fields)

        return enhanced_fields

    def _extract_from_highlight(self, highlight_text: str) -> Dict[str, Any]:
        """Extract business receipt information from highlighted text region."""
        fields = {}

        # Try all extraction methods on the highlighted text
        business_name = self._extract_retail_business_name(highlight_text)
        if business_name:
            fields["business_name"] = business_name

        # Look for amount patterns specifically in highlights
        amount_patterns = [
            (r"(?:total|amount|pay)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)", "total_amount"),
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
                        elif (
                            "gst" in highlight_text.lower()
                            or "tax" in highlight_text.lower()
                        ):
                            fields["gst_amount"] = amount
                        elif "sub" in highlight_text.lower():
                            fields["subtotal"] = amount
                        else:
                            # Could be an item price or total
                            fields["highlight_amount"] = amount
                    else:
                        fields[field_name] = amount
                    break
                except ValueError:
                    continue

        # Extract receipt number from highlights
        receipt_number = self._extract_receipt_number(highlight_text)
        if receipt_number:
            fields["receipt_number"] = receipt_number

        # Extract payment method from highlights
        payment_method = self._extract_payment_method(highlight_text)
        if payment_method:
            fields["payment_method"] = payment_method

        # Extract date from highlights
        date_match = self._extract_first_match(highlight_text, self.date_patterns)
        if date_match:
            fields["date"] = date_match

        # Extract ABN from highlights
        abn_match = self._extract_first_match(highlight_text, self.abn_patterns)
        if abn_match:
            fields["abn"] = abn_match

        return fields

    def _deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate items and prioritize highlighted items."""
        seen = set()
        unique_items = []

        for item in items:
            # Create key based on description and amount
            key = (item.get("description", "").lower().strip(), item.get("amount", 0))

            if key not in seen:
                seen.add(key)
                unique_items.append(item)
            else:
                # If duplicate, prefer highlighted version
                if item.get("highlighted", False):
                    # Replace existing with highlighted version
                    for i, existing in enumerate(unique_items):
                        existing_key = (
                            existing.get("description", "").lower().strip(),
                            existing.get("amount", 0),
                        )
                        if existing_key == key:
                            unique_items[i] = item
                            break

        return unique_items

    def _validate_receipt_calculations(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct receipt calculations after highlight enhancement."""

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
                    if calculated_gst > 0:
                        fields["gst_amount"] = calculated_gst
                        fields["calculated_gst"] = True
                        logger.info(
                            f"Calculated GST from highlights: ${calculated_gst:.2f}"
                        )
                else:
                    existing_gst = float(fields["gst_amount"])
                    difference = abs(calculated_gst - existing_gst)

                    # If highlighted calculation is more accurate, use it
                    if difference > 0.10 and calculated_gst > 0:
                        # Validate against 10% GST rate
                        expected_gst = subtotal * 0.10
                        if abs(calculated_gst - expected_gst) < abs(
                            existing_gst - expected_gst
                        ):
                            fields["gst_amount"] = calculated_gst
                            fields["corrected_gst"] = True
                            logger.info(
                                f"Corrected GST using calculation: ${calculated_gst:.2f}"
                            )

            except (ValueError, TypeError):
                logger.warning("Could not validate GST calculations from highlights")

        # If we have subtotal and GST, calculate total
        elif (
            "subtotal" in fields
            and fields["subtotal"]
            and "gst_amount" in fields
            and fields["gst_amount"]
        ):
            try:
                subtotal = float(fields["subtotal"])
                gst_amount = float(fields["gst_amount"])
                calculated_total = subtotal + gst_amount

                if "total_amount" not in fields or not fields["total_amount"]:
                    fields["total_amount"] = calculated_total
                    fields["calculated_total"] = True
                    logger.info(
                        f"Calculated total from highlights: ${calculated_total:.2f}"
                    )

            except (ValueError, TypeError):
                logger.warning("Could not calculate total from subtotal and GST")

        # Calculate totals from items if available
        if "items" in fields and fields["items"] and len(fields["items"]) > 0:
            try:
                items_total = sum(item.get("amount", 0) for item in fields["items"])

                # If we don't have a total amount, use items total
                if "total_amount" not in fields or not fields["total_amount"]:
                    # Add estimated GST (10% of items total)
                    estimated_total_with_gst = items_total * 1.10
                    fields["total_amount"] = estimated_total_with_gst
                    fields["estimated_from_items"] = True
                    logger.info(
                        f"Estimated total from items: ${estimated_total_with_gst:.2f}"
                    )

            except (ValueError, TypeError):
                logger.warning("Could not calculate total from items")

        return fields

    def _apply_enhanced_parsing(self, text: str) -> Dict[str, Any]:
        """Apply InternVL enhanced parsing techniques for business receipts."""
        if not self.supports_enhanced_parsing:
            return {}

        enhanced_fields = {}

        # Enhanced business name recognition with fuzzy matching
        business_name = self._enhanced_business_recognition(text)
        if business_name:
            enhanced_fields["business_name"] = business_name

        # Enhanced item extraction with business expense categorization
        categorized_items = self._categorize_business_items(text)
        if categorized_items:
            enhanced_fields["categorized_items"] = categorized_items

        # Extract loyalty program information
        loyalty_info = self._extract_loyalty_program_info(text)
        if loyalty_info:
            enhanced_fields.update(loyalty_info)

        # Enhanced store location extraction
        enhanced_location = self._enhanced_store_location_extraction(text)
        if enhanced_location:
            enhanced_fields["store_location"] = enhanced_location

        # Extract staff/cashier information
        staff_info = self._extract_staff_information(text)
        if staff_info:
            enhanced_fields["staff_id"] = staff_info

        return enhanced_fields

    def _enhanced_business_recognition(self, text: str) -> str:
        """Enhanced business name recognition with fuzzy matching."""
        # Standard extraction first
        business_name = self._extract_retail_business_name(text)
        if business_name:
            return business_name

        # Fuzzy matching for variations
        text_lower = text.lower()
        fuzzy_matches = {
            "Woolworths": ["woolies", "ww", "fresh food people"],
            "Coles": ["coles supermarket", "coles express", "red hand"],
            "Target": ["target australia", "expect more pay less"],
            "Bunnings": ["bunnings warehouse", "lowest prices are just the beginning"],
            "Officeworks": ["officeworks", "office works"],
            "JB Hi-Fi": ["jb hifi", "jb hi fi", "for the love of tech"],
        }

        for business, variants in fuzzy_matches.items():
            if any(variant in text_lower for variant in variants):
                return business

        return ""

    def _categorize_business_items(self, text: str) -> List[Dict[str, Any]]:
        """Categorize items for business expense purposes."""
        items = self._extract_items(text)
        if not items:
            return []

        # Business expense categories
        categories = {
            "office_supplies": [
                "pen",
                "paper",
                "folder",
                "stapler",
                "ink",
                "toner",
                "notebook",
            ],
            "computer_equipment": [
                "laptop",
                "mouse",
                "keyboard",
                "monitor",
                "cable",
                "usb",
                "hard drive",
            ],
            "software": [
                "software",
                "license",
                "subscription",
                "antivirus",
                "office 365",
            ],
            "stationery": ["envelope", "stamp", "business card", "letterhead"],
            "cleaning": ["cleaning", "sanitizer", "wipes", "soap", "detergent"],
            "catering": ["coffee", "tea", "biscuits", "water", "lunch", "meeting"],
            "marketing": ["banner", "sign", "brochure", "flyer", "promotional"],
            "tools": ["drill", "hammer", "screwdriver", "tape measure", "ladder"],
            "safety": ["helmet", "gloves", "vest", "first aid", "safety"],
        }

        categorized_items = []
        for item in items:
            description = item.get("description", "").lower()
            item_category = "other"

            for category, keywords in categories.items():
                if any(keyword in description for keyword in keywords):
                    item_category = category
                    break

            categorized_item = item.copy()
            categorized_item["business_category"] = item_category
            categorized_item["business_expense_likelihood"] = (
                self._calculate_business_likelihood(description, item_category)
            )
            categorized_items.append(categorized_item)

        return categorized_items

    def _calculate_business_likelihood(self, _description: str, category: str) -> float:
        """Calculate likelihood that an item is a business expense."""
        if category in [
            "office_supplies",
            "computer_equipment",
            "software",
            "stationery",
        ]:
            return 0.9
        elif category in ["cleaning", "tools", "safety", "marketing"]:
            return 0.8
        elif category in ["catering"]:
            return 0.6  # Depends on context
        else:
            return 0.3

    def _extract_loyalty_program_info(self, text: str) -> Dict[str, Any]:
        """Extract loyalty program information."""
        loyalty_info = {}

        # Common Australian loyalty programs
        loyalty_patterns = {
            "everyday_rewards": r"everyday\s*rewards?\s*(?:card|member)?\s*:?\s*(\d+)",
            "flybuys": r"flybuys?\s*(?:card|member)?\s*:?\s*(\d+)",
            "target_circle": r"target\s*circle\s*(?:card|member)?\s*:?\s*(\d+)",
            "bunnings_membership": r"powerpass\s*(?:card|member)?\s*:?\s*(\d+)",
        }

        for program, pattern in loyalty_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                loyalty_info["loyalty_program"] = program
                loyalty_info["loyalty_member_id"] = match.group(1)
                break

        return loyalty_info

    def _enhanced_store_location_extraction(self, text: str) -> str:
        """Enhanced store location extraction with Australian context."""
        location = self._extract_store_location(text)
        if location:
            return location

        # Look for specific Australian location patterns
        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()

            # Look for store number with location
            store_match = re.search(
                r"store\s*(?:no\.?|#)?\s*(\d+)\s*([A-Za-z\s]+)", line, re.IGNORECASE
            )
            if store_match:
                return f"Store {store_match.group(1)} - {store_match.group(2).strip()}"

            # Look for address patterns
            if re.search(
                r"\d+\s+[A-Za-z\s]+(?:st|street|rd|road|ave|avenue)",
                line,
                re.IGNORECASE,
            ):
                return line

        return ""

    def _extract_staff_information(self, text: str) -> str:
        """Extract staff/cashier information."""
        staff_patterns = [
            r"(?:cashier|operator|staff|served by)\s*:?\s*([A-Za-z\s]+)",
            r"(?:op|operator)\s*:?\s*(\w+)",
            r"(?:till|register)\s*(?:no\.?|#)?\s*(\d+)",
        ]

        for pattern in staff_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""
