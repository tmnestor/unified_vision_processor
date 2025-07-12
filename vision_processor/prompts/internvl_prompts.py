"""InternVL Prompts

Collection of 47 specialized prompts from the InternVL system optimized for
document processing with computer vision and highlight detection capabilities.
"""

import logging

from ..classification.australian_tax_types import DocumentType

logger = logging.getLogger(__name__)


class InternVLPrompts:
    """InternVL specialized prompt library with 47 optimized prompts.

    Features:
    - Document-specific extraction prompts
    - Highlight-aware variations
    - Computer vision optimized instructions
    - Australian business context
    - Technical precision focus
    """

    def __init__(self):
        self.initialized = False
        self.prompts: dict[str, dict[str, str]] = {}

    def initialize(self) -> None:
        """Initialize all InternVL prompts."""
        if self.initialized:
            return

        self._load_base_prompts()
        self._load_highlight_prompts()
        self._load_specialized_prompts()

        logger.info(
            f"InternVL prompts initialized with {self.get_prompt_count()} prompts",
        )
        self.initialized = True

    def _load_base_prompts(self) -> None:
        """Load base document processing prompts (11 prompts)."""
        self.prompts["base"] = {
            DocumentType.FUEL_RECEIPT.value: """You are an expert in fuel receipt analysis. Extract all key information from this fuel receipt image.

Focus on:
- Date of purchase
- Fuel station name
- Pump number
- Fuel type (unleaded, diesel, premium)
- Quantity in litres
- Price per litre
- Total amount
- Payment method

Provide the extracted information in a structured format with confidence scores.""",
            DocumentType.TAX_INVOICE.value: """You are an expert in tax invoice processing. Analyze this tax invoice image and extract all relevant business information.

Key fields to extract:
- Invoice number and date
- Supplier details and ABN
- Customer details
- Service/goods description
- Subtotal, GST amount, total
- Payment terms and due date

Ensure accuracy for tax compliance purposes.""",
            DocumentType.BUSINESS_RECEIPT.value: """You are an expert in retail receipt analysis. Process this business receipt and extract comprehensive transaction details.

Extract:
- Store name and location
- Transaction date and time
- Item details and quantities
- Individual prices and total
- Payment method
- Receipt/transaction number
- Any discounts or promotions

Focus on accuracy and completeness.""",
            DocumentType.BANK_STATEMENT.value: """You are an expert in bank statement analysis. Examine this bank statement and extract account and transaction information.

Key information:
- Bank name and branch
- Account holder and number
- Statement period
- Opening and closing balances
- Transaction details (date, description, amount)
- Transaction types (debit/credit)

Pay attention to transaction patterns and business expenses.""",
            DocumentType.MEAL_RECEIPT.value: """You are an expert in restaurant receipt processing. Analyze this meal receipt for dining transaction details.

Extract:
- Restaurant name and location
- Date and time of meal
- Menu items and prices
- Total amount including taxes/tips
- Payment method
- Number of guests/covers
- Meal type (breakfast/lunch/dinner)

Focus on business expense validation.""",
            DocumentType.ACCOMMODATION.value: """You are an expert in accommodation receipt analysis. Process this hotel/lodging receipt for booking details.

Key information:
- Hotel/accommodation name
- Guest details and booking reference
- Check-in and check-out dates
- Room type and number
- Nightly rate and total charges
- Additional services/fees
- Payment details

Ensure accuracy for travel expense reporting.""",
            DocumentType.TRAVEL_DOCUMENT.value: """You are an expert in travel document processing. Analyze this travel-related document for journey details.

Extract:
- Transportation provider (airline, rail, etc.)
- Travel dates and times
- Origin and destination
- Passenger details
- Ticket/booking reference
- Fare breakdown and total cost
- Travel class/service type

Focus on business travel expense compliance.""",
            DocumentType.PARKING_TOLL.value: """You are an expert in parking and toll receipt analysis. Process this parking/toll receipt for transaction details.

Key fields:
- Location/facility name
- Entry and exit times
- Duration of parking/toll usage
- Vehicle registration (if shown)
- Rate structure and calculations
- Total amount charged
- Payment method

Extract all temporal and cost information accurately.""",
            DocumentType.PROFESSIONAL_SERVICES.value: """You are an expert in professional services invoice processing. Analyze this invoice for legal, accounting, or consulting services.

Extract:
- Service provider details
- Client/matter information
- Service period and description
- Time entries and hourly rates
- Expense details and disbursements
- Subtotal, taxes, and total
- Payment terms

Ensure accuracy for professional expense tracking.""",
            DocumentType.EQUIPMENT_SUPPLIES.value: """You are an expert in equipment and supplies receipt processing. Analyze this purchase receipt for business assets.

Key information:
- Supplier/vendor details
- Purchase date and order number
- Item descriptions and model numbers
- Quantities and unit prices
- Total purchase amount
- Warranty information
- Delivery/installation details

Focus on asset tracking and expense categorization.""",
            DocumentType.OTHER.value: """You are an expert in general document processing. Analyze this document and extract all relevant business information.

Extract any available:
- Business/organization name
- Document date and reference
- Transaction or service details
- Monetary amounts and calculations
- Contact information
- Any regulatory or compliance identifiers

Provide structured output with confidence indicators.""",
        }

    def _load_highlight_prompts(self) -> None:
        """Load highlight-aware prompts (11 prompts)."""
        self.prompts["highlight"] = {
            DocumentType.FUEL_RECEIPT.value: """You are an expert in fuel receipt analysis with HIGHLIGHT DETECTION capabilities.

HIGHLIGHTED REGIONS contain key information - prioritize these areas:
- Highlighted totals, prices, or amounts
- Highlighted fuel quantities or pump numbers
- Highlighted station names or dates

Standard extraction fields:
- Date, fuel station, pump number
- Fuel type and quantity in litres
- Price per litre and total amount
- Payment method and receipt number

Cross-reference highlighted information with surrounding text for validation.""",
            DocumentType.BANK_STATEMENT.value: """You are an expert in bank statement analysis with ADVANCED HIGHLIGHT DETECTION.

HIGHLIGHTED AREAS indicate important transactions or balances:
- Highlighted transactions (often work-related expenses)
- Highlighted balances (opening/closing amounts)
- Highlighted account details or references

Extract comprehensive information:
- Account details and statement period
- All transactions with dates, descriptions, amounts
- Balance information and calculations
- Identify potential business expenses in highlighted regions

Pay special attention to highlighted transactions as they may indicate business-critical information.""",
            DocumentType.BUSINESS_RECEIPT.value: """You are an expert in business receipt processing with HIGHLIGHT AWARENESS.

HIGHLIGHTED SECTIONS may contain:
- Key totals or important amounts
- Specific items or product details
- Store information or promotional offers

Extract all receipt information:
- Store name, date, transaction details
- Item lists with prices and quantities
- Subtotals, taxes, discounts, final total
- Payment method and receipt identifiers

Use highlighted areas to prioritize and validate extracted information.""",
            DocumentType.TAX_INVOICE.value: """You are an expert in tax invoice processing with HIGHLIGHT DETECTION capabilities.

HIGHLIGHTED REGIONS often contain critical tax information:
- Tax invoice indicators and ABN numbers
- GST amounts and calculations
- Total amounts and payment terms

Core extraction requirements:
- Invoice details (number, date, terms)
- Supplier and customer information
- Service/goods descriptions
- Financial breakdown (subtotal, GST, total)

Prioritize information from highlighted areas while maintaining tax compliance accuracy.""",
            # Simplified highlight prompts for other document types
            DocumentType.MEAL_RECEIPT.value: """Extract meal receipt details with highlight awareness. Highlighted areas contain key information like totals, restaurant names, or special items.""",
            DocumentType.ACCOMMODATION.value: """Process accommodation receipt with highlight detection. Focus on highlighted booking details, dates, and charges.""",
            DocumentType.TRAVEL_DOCUMENT.value: """Analyze travel document with highlight awareness. Highlighted sections contain important journey or fare information.""",
            DocumentType.PARKING_TOLL.value: """Extract parking/toll information with highlight detection. Highlighted areas show key timing or cost details.""",
            DocumentType.PROFESSIONAL_SERVICES.value: """Process professional services invoice with highlight awareness. Focus on highlighted service details and charges.""",
            DocumentType.EQUIPMENT_SUPPLIES.value: """Analyze equipment/supplies receipt with highlight detection. Highlighted items may show key products or totals.""",
            DocumentType.OTHER.value: """Process document with highlight awareness. Prioritize information from highlighted regions for key business details.""",
        }

    def _load_specialized_prompts(self) -> None:
        """Load specialized technical prompts (25 prompts)."""
        self.prompts["specialized"] = {
            # Fuel receipt specialized prompts (3 prompts)
            f"{DocumentType.FUEL_RECEIPT.value}_detailed": """FUEL RECEIPT SPECIALIST - Advanced Extraction

You are a specialized fuel transaction processor. Perform comprehensive analysis:

PRIMARY EXTRACTION:
1. Temporal Data: Extract precise date/time with timezone awareness
2. Station Identification: Recognize Australian fuel brands (BP, Shell, Caltex, Ampol, Mobil, 7-Eleven, United, Liberty, Metro)
3. Fuel Specifications: Type (ULP91/95/98, Premium, Diesel, E10, E85, LPG), octane rating
4. Quantity Metrics: Litres with precision to 3 decimal places
5. Pricing Analysis: Rate per litre, total fuel cost, price validation
6. Transaction Details: Pump number, payment method, receipt/transaction ID

VALIDATION PROTOCOLS:
- Cross-verify quantity × rate = total calculations
- Validate price ranges (Australian fuel prices: $1.20-$2.50/L)
- Confirm fuel type consistency with pricing tier
- Check for promotional pricing or discounts

OUTPUT: Structured JSON with confidence scoring for each field.""",
            f"{DocumentType.FUEL_RECEIPT.value}_australian": """AUSTRALIAN FUEL STATION EXPERT

Specialized in Australian fuel retail ecosystem:

STATION CLASSIFICATION:
- Major Chains: BP, Shell, Caltex (now Ampol), Mobil, 7-Eleven
- Independent Networks: United Petroleum, Liberty Oil, Metro Petroleum, Puma Energy
- Supermarket Fuel: Woolworths Petrol, Coles Express, Costco Fuel
- Regional Operators: Speedway, Gull, local independents

AUSTRALIAN FUEL STANDARDS:
- Regular Unleaded (ULP91): Most common grade
- Premium Unleaded (95/98 RON): Higher octane options
- Diesel: Standard automotive diesel
- E10: 10% ethanol blend (cheaper option)
- LPG: Liquid petroleum gas (alternative fuel)

PRICING CONTEXT:
- Metropolitan vs regional pricing variations
- Fuel cycle patterns (weekly price movements)
- Government fuel taxation components
- Discount programs (shopper dockets, loyalty cards)

Extract with Australian market knowledge and validate against typical patterns.""",
            f"{DocumentType.FUEL_RECEIPT.value}_validation": """FUEL RECEIPT VALIDATION SPECIALIST

Advanced validation engine for fuel transaction accuracy:

CALCULATION VERIFICATION:
1. Primary: Litres × Price/Litre = Fuel Cost
2. Additional: Pump fees, transaction fees, loyalty discounts
3. Payment: Total amount vs payment method reconciliation

ANOMALY DETECTION:
- Price outliers (too high/low for market conditions)
- Quantity anomalies (vehicle tank capacity validation)
- Date/time inconsistencies
- Station location vs pricing validation

COMPLIANCE CHECKS:
- Receipt format completeness
- Regulatory requirement adherence
- Tax component identification (fuel excise)
- Business expense eligibility assessment

CONFIDENCE SCORING:
- Field extraction confidence (0-1 scale)
- Calculation accuracy rating
- Overall document authenticity assessment
- Business expense categorization reliability""",
            # Bank statement specialized prompts (5 prompts)
            f"{DocumentType.BANK_STATEMENT.value}_transactions": """BANK TRANSACTION SPECIALIST

Expert in Australian banking transaction analysis:

TRANSACTION CATEGORIZATION:
1. Work Expenses: Fuel, parking, meals, accommodation, transport, equipment
2. Business Services: Professional fees, software subscriptions, office supplies
3. Personal Expenses: Personal shopping, entertainment, personal travel
4. Transfers: Inter-account, loan payments, investments
5. Income: Salary, business income, interest, dividends

AUSTRALIAN BANKING CONTEXT:
- Big Four Banks: ANZ, Commonwealth Bank, Westpac, NAB
- Digital Banks: ING Direct, Macquarie, Bendigo Bank, Suncorp
- Credit Unions: Various regional providers
- Payment Systems: EFTPOS, PayWave, BPAY, Osko/PayID

WORK EXPENSE IDENTIFICATION:
- Business travel patterns (flights, accommodation, car rental)
- Professional development (courses, conferences, subscriptions)
- Office operations (supplies, equipment, utilities)
- Client entertainment (meals, venues within limits)

Extract and categorize with business tax implications.""",
            f"{DocumentType.BANK_STATEMENT.value}_compliance": """BANK STATEMENT COMPLIANCE EXPERT

ATO compliance specialist for business banking records:

RECORD KEEPING REQUIREMENTS:
- Transaction substantiation for business expenses
- Personal vs business expense segregation
- Capital vs operational expense classification
- GST input tax credit eligibility

DOCUMENTATION STANDARDS:
- Minimum 5-year retention period
- Digital record acceptance criteria
- Supporting documentation requirements
- Audit trail maintenance

BUSINESS EXPENSE VALIDATION:
- Wholly business purpose test
- Incidental personal use allowances
- Entertainment expense limitations (FBT implications)
- Home office expense apportionment

TAX DEDUCTION CATEGORIES:
- Immediate deductions (operational expenses)
- Capital allowances (depreciation over time)
- Prepaid expenses (timing considerations)
- Mixed-purpose asset apportionment

Analyze with tax law compliance perspective.""",
            f"{DocumentType.BANK_STATEMENT.value}_patterns": """BANKING PATTERN ANALYST

Advanced pattern recognition for financial behavior analysis:

SPENDING PATTERNS:
- Recurring transactions (subscriptions, utilities, loan payments)
- Seasonal variations (quarterly payments, annual fees)
- Business cycle correlations (project-based income/expenses)
- Geographic spending analysis (business travel routes)

CASH FLOW ANALYSIS:
- Income timing and regularity assessment
- Expense scheduling and planning indicators
- Liquidity management patterns
- Investment vs operational fund allocation

RISK INDICATORS:
- Unusual transaction amounts or frequencies
- Off-pattern geographic transactions
- Timing anomalies (business hours vs transaction times)
- Vendor relationship changes

BUSINESS INSIGHTS:
- Revenue trend analysis
- Expense category optimization opportunities
- Cash flow forecasting data points
- Financial health indicators

Generate comprehensive financial behavior profile.""",
            f"{DocumentType.BANK_STATEMENT.value}_reconciliation": """BANK RECONCILIATION SPECIALIST

Expert in financial reconciliation and accuracy verification:

BALANCE VERIFICATION:
- Opening balance + Credits - Debits = Closing balance
- Running balance calculations throughout statement period
- Interest calculations and fee assessments
- Currency conversion accuracy (if applicable)

TRANSACTION MATCHING:
- Cross-reference with business accounting records
- Invoice matching for payment verification
- Duplicate transaction identification
- Timing difference reconciliation (processing delays)

DISCREPANCY ANALYSIS:
- Missing transaction identification
- Amount variances and explanations
- Date discrepancies and corrections
- Fee and charge validation

REPORTING ACCURACY:
- Statement period completeness
- Account detail accuracy (BSB, account numbers)
- Customer information verification
- Regulatory compliance markers

Ensure 100% accuracy in financial reconciliation.""",
            f"{DocumentType.BANK_STATEMENT.value}_ato": """ATO BANK STATEMENT PROCESSOR

Australian Taxation Office compliance specialist:

ATO REQUIREMENTS:
- Business Activity Statement (BAS) supporting documentation
- Income tax return substantiation records
- Fringe Benefits Tax (FBT) calculation support
- Goods and Services Tax (GST) credit validation

BUSINESS EXPENSE EVIDENCE:
- Date, amount, supplier, business purpose documentation
- Mixed-purpose expense apportionment calculations
- Entertainment expense limitations and FBT implications
- Travel expense substantiation requirements

RECORD KEEPING COMPLIANCE:
- English language requirement (translations if needed)
- 5-year retention period compliance
- Digital format acceptance standards
- Audit accessibility requirements

TAX CALCULATION SUPPORT:
- Assessable income identification
- Allowable deduction categorization
- Capital gains/losses recognition
- Timing difference adjustments

Process with ATO audit readiness focus.""",
            # Business receipt specialized prompts (4 prompts)
            f"{DocumentType.BUSINESS_RECEIPT.value}_retail": """AUSTRALIAN RETAIL SPECIALIST

Expert in Australian retail ecosystem and receipt processing:

MAJOR RETAIL CHAINS:
- Supermarkets: Woolworths, Coles, ALDI, IGA
- Department Stores: Target, Kmart, Big W, Myer, David Jones
- Hardware: Bunnings Warehouse, Masters (historical)
- Electronics: JB Hi-Fi, Harvey Norman, Officeworks
- Pharmacy: Chemist Warehouse, Priceline, Terry White

RECEIPT ANALYSIS:
- Store identification and location verification
- Product categorization (business vs personal)
- Discount and promotion recognition
- Payment method validation
- Loyalty program integration

BUSINESS EXPENSE CATEGORIZATION:
- Office supplies and stationery
- Business equipment and tools
- Professional services and subscriptions
- Maintenance and cleaning supplies
- Business meal and entertainment

GST COMPLIANCE:
- GST-inclusive pricing recognition
- Tax invoice requirements (>$75 purchases)
- ABN verification for supplier payments
- Input tax credit eligibility assessment

Extract with Australian retail market knowledge.""",
            f"{DocumentType.BUSINESS_RECEIPT.value}_items": """ITEMIZED RECEIPT SPECIALIST

Advanced item-level analysis for detailed expense tracking:

ITEM CATEGORIZATION:
- Product codes and SKU analysis
- Brand recognition and categorization
- Quantity and unit price validation
- Discount application verification
- Bundle pricing breakdown

BUSINESS RELEVANCE SCORING:
- Office supplies: High business relevance
- Food/beverages: Context-dependent (meetings vs personal)
- Electronics: Equipment classification required
- Clothing: Uniform vs personal determination
- Services: Professional vs personal assessment

TAX IMPLICATIONS:
- Capital asset identification (>$300 threshold)
- Immediate deduction eligibility
- FBT implications for personal use items
- Input tax credit calculations
- Depreciation scheduling requirements

ACCURACY VERIFICATION:
- Mathematical validation (quantity × price)
- Discount calculation verification
- Tax calculation accuracy
- Payment reconciliation
- Receipt total validation

Generate detailed expense breakdown with tax implications.""",
            f"{DocumentType.BUSINESS_RECEIPT.value}_compliance": """RETAIL COMPLIANCE EXPERT

ATO compliance specialist for retail purchases:

TAX INVOICE REQUIREMENTS:
- Supplier ABN and business name
- Tax invoice wording (purchases >$75)
- Date of supply identification
- Recipient business details
- GST amount separately identified

DEDUCTION ELIGIBILITY:
- Business purpose test application
- Private/domestic use exclusions
- Income-producing activity connection
- Ordinary and necessary business expense criteria
- Capital vs revenue distinction

RECORD KEEPING:
- 5-year retention requirement
- Electronic storage acceptance
- Supporting documentation links
- Business purpose annotations
- Audit trail maintenance

EXPENSE CLASSIFICATION:
- Immediate deductions vs capital allowances
- Repairs vs improvements determination
- Staff amenities and welfare expenses
- Entertainment expense limitations
- Home office expense apportionment

Process with tax compliance accuracy.""",
            f"{DocumentType.BUSINESS_RECEIPT.value}_gst": """GST RECEIPT SPECIALIST

Goods and Services Tax compliance expert:

GST CALCULATION VERIFICATION:
- 10% GST rate application (standard)
- GST-inclusive vs GST-exclusive pricing
- GST-free items identification (basic food, medical)
- Input tax credit eligibility assessment
- Rounding rules application (nearest 5 cents)

TAX INVOICE VALIDATION:
- ABN format verification (XX XXX XXX XXX)
- "Tax Invoice" wording presence (>$75)
- Supplier business name matching
- GST amount separate identification
- Date and amount accuracy

BUSINESS ACTIVITY STATEMENT (BAS):
- Input tax credit calculations
- GST collected reconciliation
- Timing difference adjustments
- Cash vs accrual accounting implications
- Annual turnover threshold monitoring

COMPLIANCE DOCUMENTATION:
- Record substantiation requirements
- Mixed-purpose purchase apportionment
- Entertainment expense GST restrictions
- Motor vehicle expense calculations
- Overseas purchase GST obligations

Ensure GST compliance accuracy.""",
            # Simplified specialized prompts for other document types (8 prompts)
            f"{DocumentType.TAX_INVOICE.value}_ato": """ATO Tax Invoice Specialist: Ensure compliance with Australian tax invoice requirements including ABN, GST calculations, and proper formatting.""",
            f"{DocumentType.TAX_INVOICE.value}_professional": """Professional Services Invoice Expert: Specialized in legal, accounting, and consulting invoice formats with time-based billing analysis.""",
            f"{DocumentType.MEAL_RECEIPT.value}_business": """Business Meal Specialist: Focus on entertainment expense compliance, FBT implications, and substantiation requirements.""",
            f"{DocumentType.ACCOMMODATION.value}_travel": """Travel Accommodation Expert: Specialized in business travel substantiation and overnight allowance calculations.""",
            f"{DocumentType.TRAVEL_DOCUMENT.value}_business": """Business Travel Specialist: Focus on substantiation requirements and travel expense categorization.""",
            f"{DocumentType.PARKING_TOLL.value}_vehicle": """Vehicle Expense Specialist: Analyze parking and toll expenses for business vehicle usage calculations.""",
            f"{DocumentType.PROFESSIONAL_SERVICES.value}_billing": """Professional Billing Expert: Time-based billing analysis with hourly rate validation and expense reconciliation.""",
            f"{DocumentType.EQUIPMENT_SUPPLIES.value}_assets": """Business Asset Specialist: Capital vs revenue determination, depreciation classification, and asset register integration.""",
        }

    def get_base_prompt(self, document_type: DocumentType) -> str:
        """Get base prompt for a document type."""
        return self.prompts["base"].get(
            document_type.value,
            self.prompts["base"][DocumentType.OTHER.value],
        )

    def get_highlight_prompt(self, document_type: DocumentType) -> str:
        """Get highlight-aware prompt for a document type."""
        return self.prompts["highlight"].get(
            document_type.value,
            self.prompts["highlight"][DocumentType.OTHER.value],
        )

    def get_specialized_prompt(
        self,
        document_type: DocumentType,
        variant: str = "detailed",
    ) -> str:
        """Get specialized prompt for a document type and variant."""
        key = f"{document_type.value}_{variant}"

        # Try specific variant first
        if key in self.prompts["specialized"]:
            return self.prompts["specialized"][key]

        # Fallback to base specialized prompt
        base_key = f"{document_type.value}_detailed"
        if base_key in self.prompts["specialized"]:
            return self.prompts["specialized"][base_key]

        # Final fallback to base prompt
        return self.get_base_prompt(document_type)

    def get_prompt_count(self) -> int:
        """Get total number of InternVL prompts."""
        count = 0
        for category in self.prompts.values():
            count += len(category)
        return count

    def get_available_variants(self, document_type: DocumentType) -> list[str]:
        """Get available specialized variants for a document type."""
        variants = []
        doc_type_prefix = f"{document_type.value}_"

        for key in self.prompts["specialized"]:
            if key.startswith(doc_type_prefix):
                variant = key[len(doc_type_prefix) :]
                variants.append(variant)

        return variants

    def get_category_statistics(self) -> dict[str, int]:
        """Get prompt count by category."""
        return {category: len(prompts) for category, prompts in self.prompts.items()}
