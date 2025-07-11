"""Fuel Receipt Handler

Specialized handler for fuel receipts following the Llama 7-step pipeline
with Australian fuel station recognition and vehicle expense validation.
"""

import logging
import re
from typing import Any

from .base_ato_handler import BaseATOHandler

logger = logging.getLogger(__name__)


class FuelReceiptHandler(BaseATOHandler):
    """Handler for fuel receipts with Australian fuel station expertise.

    Supports major Australian fuel stations:
    - BP, Shell, Caltex/Ampol, Mobil, 7-Eleven
    - United Petroleum, Liberty, Metro Petroleum
    - Independent stations

    ATO Requirements:
    - Date of purchase
    - Fuel type and quantity
    - Cost per litre
    - Total amount including GST
    - Station name and location
    - Vehicle expense business purpose
    """

    def _load_field_requirements(self) -> None:
        """Load fuel receipt specific field requirements."""
        self.required_fields = [
            "date",
            "fuel_type",
            "litres",
            "total_amount",
            "station_name",
        ]

        self.optional_fields = [
            "price_per_litre",
            "gst_amount",
            "pump_number",
            "location",
            "abn",
            "odometer_reading",
            "vehicle_registration",
        ]

    def _load_validation_rules(self) -> None:
        """Load fuel receipt validation rules."""
        self.validation_rules = {
            "litres_range": (1.0, 200.0),  # Reasonable fuel purchase range
            "price_per_litre_range": (1.0, 3.0),  # Realistic fuel prices in Australia
            "total_amount_range": (5.0, 500.0),  # Reasonable total purchase range
            "fuel_types": [
                "unleaded",
                "premium unleaded",
                "diesel",
                "e10",
                "e85",
                "lpg",
                "premium 95",
                "premium 98",
            ],
        }

        # Australian fuel station patterns
        self.fuel_station_patterns = [
            r"(?:bp|shell|caltex|ampol|mobil|7.?eleven)\s*(?:fuel|petrol|service)",
            r"(?:united|liberty|metro)\s*petroleum",
            r"(?:speedway|puma|gull)\s*(?:fuel|petrol)",
            r"(?:costco|woolworths|coles)\s*fuel",
        ]

    def _extract_document_specific_fields(self, text: str) -> dict[str, Any]:
        """Extract fuel receipt specific fields."""
        fields = {}

        # Extract fuel type
        fuel_type = self._extract_fuel_type(text)
        if fuel_type:
            fields["fuel_type"] = fuel_type

        # Extract litres
        litres = self._extract_litres(text)
        if litres:
            fields["litres"] = litres

        # Extract price per litre
        price_per_litre = self._extract_price_per_litre(text)
        if price_per_litre:
            fields["price_per_litre"] = price_per_litre

        # Extract pump number
        pump_number = self._extract_pump_number(text)
        if pump_number:
            fields["pump_number"] = pump_number

        # Extract station name
        station_name = self._extract_station_name(text)
        if station_name:
            fields["station_name"] = station_name

        # Extract location
        location = self._extract_location(text)
        if location:
            fields["location"] = location

        # Extract odometer reading if present
        odometer = self._extract_odometer(text)
        if odometer:
            fields["odometer_reading"] = odometer

        return fields

    def _extract_fuel_type(self, text: str) -> str:
        """Extract fuel type from receipt text."""
        fuel_type_patterns = [
            r"(?:unleaded|ulp)\s*(?:91|95|98)?",
            r"premium\s*(?:unleaded|ulp)?\s*(?:95|98)",
            r"(?:diesel|dsl)",
            r"e10",
            r"e85",
            r"lpg",
            r"supreme\s*(?:unleaded|plus)",
        ]

        for pattern in fuel_type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).lower().strip()

        return ""

    def _extract_litres(self, text: str) -> float:
        """Extract fuel quantity in litres."""
        litre_patterns = [
            r"(\d+(?:\.\d{1,3})?)\s*(?:l|litres?|lts?)\b",
            r"(?:quantity|qty|litres?)\s*:?\s*(\d+(?:\.\d{1,3})?)",
            r"(\d+(?:\.\d{1,3})?)\s*(?:litres?|l)\s*@",
        ]

        for pattern in litre_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return 0.0

    def _extract_price_per_litre(self, text: str) -> float:
        """Extract price per litre."""
        price_patterns = [
            r"@\s*\$?\s*(\d+(?:\.\d{1,3})?)\s*(?:per\s*)?(?:l|litre)",
            r"(\d+(?:\.\d{1,3})?)\s*(?:c|cents?)\s*(?:per\s*)?(?:l|litre)",
            r"price\s*(?:per\s*)?(?:l|litre)\s*:?\s*\$?\s*(\d+(?:\.\d{1,3})?)",
        ]

        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    # Convert cents to dollars if necessary
                    if price > 10.0:  # Likely in cents
                        price = price / 100.0
                    return price
                except ValueError:
                    continue

        return 0.0

    def _extract_pump_number(self, text: str) -> str:
        """Extract pump number."""
        pump_patterns = [
            r"pump\s*(?:no\.?|number|#)?\s*(\d+)",
            r"(?:pump|pos)\s*(\d+)",
        ]

        for pattern in pump_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""

    def _extract_station_name(self, text: str) -> str:
        """Extract fuel station name."""
        # Check for known Australian fuel stations
        station_patterns = {
            "BP": r"\b(?:bp|british petroleum)\b",
            "Shell": r"\bshell\b",
            "Caltex": r"\bcaltex\b",
            "Ampol": r"\bampol\b",
            "Mobil": r"\bmobil\b",
            "7-Eleven": r"\b7.?eleven\b",
            "United Petroleum": r"\bunited\s*petroleum\b",
            "Liberty": r"\bliberty\s*(?:fuel|oil)?\b",
            "Metro Petroleum": r"\bmetro\s*petroleum\b",
            "Speedway": r"\bspeedway\b",
            "Puma Energy": r"\bpuma\s*energy\b",
            "Gull": r"\bgull\b",
            "Costco Fuel": r"\bcostco\s*fuel\b",
            "Woolworths Petrol": r"\bwoolworths\s*(?:petrol|fuel)\b",
            "Coles Express": r"\bcoles\s*express\b",
        }

        for station_name, pattern in station_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return station_name

        # Try to extract station name from header lines
        lines = text.split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 3 and not re.match(r"^\d", line):
                # Potential station name
                return line

        return ""

    def _extract_location(self, text: str) -> str:
        """Extract station location."""
        # Australian states and territories
        state_patterns = [
            r"\b(?:nsw|new south wales)\b",
            r"\b(?:vic|victoria)\b",
            r"\b(?:qld|queensland)\b",
            r"\b(?:wa|western australia)\b",
            r"\b(?:sa|south australia)\b",
            r"\b(?:tas|tasmania)\b",
            r"\b(?:act|australian capital territory)\b",
            r"\b(?:nt|northern territory)\b",
        ]

        # Look for Australian postcodes (4 digits)
        postcode_pattern = r"\b(\d{4})\b"
        postcode_match = re.search(postcode_pattern, text)

        # Look for state abbreviations
        for pattern in state_patterns:
            state_match = re.search(pattern, text, re.IGNORECASE)
            if state_match:
                location = state_match.group(0).upper()
                if postcode_match:
                    location = f"{postcode_match.group(1)}, {location}"
                return location

        if postcode_match:
            return postcode_match.group(1)

        return ""

    def _extract_odometer(self, text: str) -> str:
        """Extract odometer reading if present."""
        odometer_patterns = [
            r"(?:odometer|odo|km)\s*:?\s*(\d{1,7})",
            r"(\d{1,7})\s*(?:km|kilometres)",
        ]

        for pattern in odometer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""

    def _validate_document_specific_fields(self, fields: dict[str, Any]) -> list[str]:
        """Validate fuel receipt specific fields."""
        issues = []

        # Validate fuel type
        if fields.get("fuel_type"):
            fuel_type = fields["fuel_type"].lower()
            valid_types = [ft.lower() for ft in self.validation_rules["fuel_types"]]
            if not any(vt in fuel_type for vt in valid_types):
                issues.append(f"Unrecognized fuel type: {fields['fuel_type']}")

        # Validate litres range
        if fields.get("litres"):
            try:
                litres = float(fields["litres"])
                min_litres, max_litres = self.validation_rules["litres_range"]
                if not (min_litres <= litres <= max_litres):
                    issues.append(
                        f"Fuel quantity {litres}L outside reasonable range ({min_litres}-{max_litres}L)",
                    )
            except (ValueError, TypeError):
                issues.append("Invalid fuel quantity format")

        # Validate price per litre
        if fields.get("price_per_litre"):
            try:
                price = float(fields["price_per_litre"])
                min_price, max_price = self.validation_rules["price_per_litre_range"]
                if not (min_price <= price <= max_price):
                    issues.append(
                        f"Price per litre ${price:.3f} outside reasonable range (${min_price}-${max_price})",
                    )
            except (ValueError, TypeError):
                issues.append("Invalid price per litre format")

        # Validate total amount
        if fields.get("total_amount"):
            try:
                total = float(fields["total_amount"])
                min_total, max_total = self.validation_rules["total_amount_range"]
                if not (min_total <= total <= max_total):
                    issues.append(
                        f"Total amount ${total:.2f} outside reasonable range (${min_total}-${max_total})",
                    )
            except (ValueError, TypeError):
                issues.append("Invalid total amount format")

        # Cross-validate litres and price calculation
        if all(
            field in fields and fields[field] for field in ["litres", "price_per_litre", "total_amount"]
        ):
            try:
                litres = float(fields["litres"])
                price_per_litre = float(fields["price_per_litre"])
                total_amount = float(fields["total_amount"])

                calculated_total = litres * price_per_litre
                difference = abs(calculated_total - total_amount)

                # Allow for GST and rounding differences
                tolerance = max(calculated_total * 0.15, 2.0)  # 15% or $2 tolerance
                if difference > tolerance:
                    issues.append(
                        f"Calculated total ${calculated_total:.2f} differs significantly from stated total ${total_amount:.2f}",
                    )
            except (ValueError, TypeError):
                issues.append("Cannot validate fuel calculation due to invalid values")

        return issues

    def enhance_with_highlights(
        self,
        fields: dict[str, Any],
        highlights: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Enhance fuel receipt extraction using InternVL highlight detection.

        Fuel receipts benefit from highlight detection for:
        - Total amount highlighting (most important)
        - Fuel type and quantity highlighting
        - Price per litre highlighting
        - Pump number identification
        """
        if not highlights or not self.supports_highlights:
            return fields

        enhanced_fields = fields.copy()
        logger.info(f"Enhancing fuel receipt with {len(highlights)} highlights")

        # Priority order for fuel receipt fields
        priority_fields = [
            "total_amount",
            "litres",
            "price_per_litre",
            "fuel_type",
            "pump_number",
            "date",
        ]

        for highlight in highlights:
            if highlight.get("text"):
                highlight_text = highlight["text"]
                highlight_confidence = highlight.get("confidence", 0.8)

                # Extract fuel-specific information from highlighted regions
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

        # Validate calculations after highlight enhancement
        enhanced_fields = self._validate_fuel_calculations(enhanced_fields)

        return enhanced_fields

    def _extract_from_highlight(self, highlight_text: str) -> dict[str, Any]:
        """Extract fuel receipt information from highlighted text region."""
        fields = {}

        # Try all extraction methods on the highlighted text
        fuel_type = self._extract_fuel_type(highlight_text)
        if fuel_type:
            fields["fuel_type"] = fuel_type

        litres = self._extract_litres(highlight_text)
        if litres:
            fields["litres"] = litres

        price_per_litre = self._extract_price_per_litre(highlight_text)
        if price_per_litre:
            fields["price_per_litre"] = price_per_litre

        pump_number = self._extract_pump_number(highlight_text)
        if pump_number:
            fields["pump_number"] = pump_number

        # Look for total amount patterns specifically in highlights
        total_patterns = [
            r"(?:total|amount|pay)\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)",
            r"\$\s*(\d+(?:\.\d{2})?)",
            r"(\d+\.\d{2})\s*$",  # Amount at end of line
        ]

        for pattern in total_patterns:
            match = re.search(pattern, highlight_text, re.IGNORECASE)
            if match:
                try:
                    fields["total_amount"] = float(match.group(1))
                    break
                except ValueError:
                    continue

        # Extract date from highlights
        date_match = self._extract_first_match(highlight_text, self.date_patterns)
        if date_match:
            fields["date"] = date_match

        return fields

    def _validate_fuel_calculations(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Validate and correct fuel calculations after highlight enhancement."""
        if all(field in fields and fields[field] for field in ["litres", "price_per_litre"]):
            try:
                litres = float(fields["litres"])
                price_per_litre = float(fields["price_per_litre"])
                calculated_total = litres * price_per_litre

                # If we don't have a total or it's significantly wrong, use calculated
                if "total_amount" not in fields or not fields["total_amount"]:
                    fields["total_amount"] = calculated_total
                    fields["calculated_total"] = True
                    logger.info(
                        f"Calculated total from highlights: ${calculated_total:.2f}",
                    )
                else:
                    existing_total = float(fields["total_amount"])
                    difference = abs(calculated_total - existing_total)

                    # If highlighted calculation is more accurate, use it
                    if difference > 2.0 and calculated_total > 0:
                        # Check if calculation is more reasonable
                        if 5.0 <= calculated_total <= 500.0:  # Reasonable fuel purchase range
                            fields["total_amount"] = calculated_total
                            fields["corrected_from_calculation"] = True
                            logger.info(
                                f"Corrected total using calculation: ${calculated_total:.2f}",
                            )

            except (ValueError, TypeError):
                logger.warning("Could not validate fuel calculations from highlights")

        return fields

    def _apply_enhanced_parsing(self, text: str) -> dict[str, Any]:
        """Apply InternVL enhanced parsing techniques for fuel receipts."""
        if not self.supports_enhanced_parsing:
            return {}

        enhanced_fields = {}

        # Enhanced fuel station recognition using fuzzy matching
        station_name = self._enhanced_station_recognition(text)
        if station_name:
            enhanced_fields["station_name"] = station_name

        # Enhanced fuel type detection with context
        fuel_type = self._enhanced_fuel_type_detection(text)
        if fuel_type:
            enhanced_fields["fuel_type"] = fuel_type

        # Enhanced location extraction with Australian context
        location = self._enhanced_location_extraction(text)
        if location:
            enhanced_fields["location"] = location

        # Vehicle registration extraction
        vehicle_reg = self._extract_vehicle_registration(text)
        if vehicle_reg:
            enhanced_fields["vehicle_registration"] = vehicle_reg

        return enhanced_fields

    def _enhanced_station_recognition(self, text: str) -> str:
        """Enhanced fuel station recognition with fuzzy matching."""
        # First try exact patterns
        station_name = self._extract_station_name(text)
        if station_name:
            return station_name

        # Try fuzzy matching for Australian stations
        station_keywords = {
            "BP": ["british petroleum", "bp"],
            "Shell": ["shell"],
            "Caltex": ["caltex"],
            "Ampol": ["ampol"],
            "Mobil": ["mobil", "exxonmobil"],
            "7-Eleven": ["7-eleven", "seven eleven", "711"],
            "United Petroleum": ["united", "united petroleum"],
            "Liberty": ["liberty", "liberty oil"],
            "Metro Petroleum": ["metro", "metro petroleum"],
        }

        text_lower = text.lower()
        for station, keywords in station_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return station

        return ""

    def _enhanced_fuel_type_detection(self, text: str) -> str:
        """Enhanced fuel type detection with context awareness."""
        # Standard extraction first
        fuel_type = self._extract_fuel_type(text)
        if fuel_type:
            return fuel_type

        # Context-aware detection
        fuel_context_patterns = [
            r"(?:grade|octane)\s*(\d+)",  # Octane rating
            r"(?:regular|standard)\s*unleaded",
            r"(?:super|supreme|premium)",
            r"(?:automotive|auto)\s*diesel",
        ]

        for pattern in fuel_context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "91" in match.group(0) or "regular" in match.group(0).lower():
                    return "unleaded 91"
                if "95" in match.group(0) or "premium" in match.group(0).lower():
                    return "premium unleaded 95"
                if "98" in match.group(0) or "supreme" in match.group(0).lower():
                    return "premium unleaded 98"
                if "diesel" in match.group(0).lower():
                    return "diesel"

        return ""

    def _enhanced_location_extraction(self, text: str) -> str:
        """Enhanced location extraction with Australian context."""
        # Standard extraction first
        location = self._extract_location(text)
        if location:
            return location

        # Enhanced extraction with suburb names
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Look for lines with Australian address patterns
            if re.search(r"\b\d{4}\b", line):  # Contains postcode
                # Check if line contains state abbreviation
                if re.search(
                    r"\b(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)\b",
                    line,
                    re.IGNORECASE,
                ) or re.search(
                    r"\b\d+\s+[A-Za-z\s]+(?:st|street|rd|road|ave|avenue|dr|drive)\b",
                    line,
                    re.IGNORECASE,
                ):
                    return line

        return ""

    def _extract_vehicle_registration(self, text: str) -> str:
        """Extract vehicle registration if present on receipt."""
        # Australian vehicle registration patterns
        rego_patterns = [
            r"(?:rego|registration|vehicle|plate)\s*:?\s*([A-Z0-9]{3,8})",
            r"\b([A-Z]{2,3}\s*[0-9]{2,3}[A-Z]?)\b",  # NSW style: ABC123
            r"\b([0-9]{3}\s*[A-Z]{3})\b",  # VIC style: 123ABC
        ]

        for pattern in rego_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper().replace(" ", "")

        return ""
