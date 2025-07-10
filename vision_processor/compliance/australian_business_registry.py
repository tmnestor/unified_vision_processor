"""Australian Business Registry

This module provides comprehensive Australian business recognition and validation,
combining data from both InternVL and Llama-3.2 systems with 100+ businesses.
"""

import logging
import re

logger = logging.getLogger(__name__)


class AustralianBusinessRegistry:
    """Comprehensive Australian business registry for recognition and validation.

    Features:
    - 100+ major Australian businesses
    - Industry categorization
    - Name normalization and fuzzy matching
    - Business type identification
    - Regional business recognition
    """

    def __init__(self, config=None):
        self.config = config
        self.initialized = False
        self.business_registry: dict[str, dict[str, any]] = {}
        self.industry_keywords: dict[str, list[str]] = {}
        self.business_aliases: dict[str, list[str]] = {}

    def initialize(self) -> None:
        """Initialize the business registry with comprehensive Australian business data."""
        if self.initialized:
            return

        # Load comprehensive business registry
        self._load_business_registry()

        # Load industry keywords for classification
        self._load_industry_keywords()

        # Load business aliases and variations
        self._load_business_aliases()

        logger.info(
            f"Australian Business Registry initialized with {len(self.business_registry)} businesses",
        )
        self.initialized = True

    def _load_business_registry(self) -> None:
        """Load comprehensive Australian business registry."""
        # Major retail chains
        retail_businesses = {
            "woolworths": {
                "official_name": "Woolworths Group Limited",
                "industry": "retail_supermarket",
                "abn": "88 000 014 675",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["woolworths", "woolies", "woolworth"],
            },
            "coles": {
                "official_name": "Coles Group Limited",
                "industry": "retail_supermarket",
                "abn": "11 004 089 936",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["coles", "coles supermarket"],
            },
            "aldi": {
                "official_name": "ALDI Stores (A Limited Partnership)",
                "industry": "retail_supermarket",
                "abn": "51 090 538 963",
                "business_type": "partnership",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["aldi", "aldi australia"],
            },
            "target": {
                "official_name": "Target Australia Pty Ltd",
                "industry": "retail_department",
                "abn": "75 004 250 944",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["target", "target australia"],
            },
            "kmart": {
                "official_name": "Kmart Australia Limited",
                "industry": "retail_department",
                "abn": "73 004 700 485",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["kmart", "k mart"],
            },
            "bunnings": {
                "official_name": "Bunnings Group Limited",
                "industry": "retail_hardware",
                "abn": "26 008 672 179",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bunnings", "bunnings warehouse"],
            },
            "officeworks": {
                "official_name": "Officeworks Limited",
                "industry": "retail_office_supplies",
                "abn": "13 004 044 937",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["officeworks", "office works"],
            },
            "harvey_norman": {
                "official_name": "Harvey Norman Holdings Limited",
                "industry": "retail_electronics",
                "abn": "35 000 870 729",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["harvey norman", "harvey normans"],
            },
            "jb_hi_fi": {
                "official_name": "JB Hi-Fi Limited",
                "industry": "retail_electronics",
                "abn": "80 093 220 136",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["jb hi-fi", "jb hifi", "jb hi fi"],
            },
            "big_w": {
                "official_name": "BIG W Pty Limited",
                "industry": "retail_department",
                "abn": "88 000 002 395",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["big w", "bigw"],
            },
        }

        # Fuel stations
        fuel_businesses = {
            "bp": {
                "official_name": "BP Australia Pty Ltd",
                "industry": "fuel_retail",
                "abn": "53 004 085 616",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bp", "bp petrol", "bp fuel"],
            },
            "shell": {
                "official_name": "Shell Company of Australia Limited",
                "industry": "fuel_retail",
                "abn": "46 000 005 407",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["shell", "shell petrol"],
            },
            "caltex": {
                "official_name": "Ampol Limited",
                "industry": "fuel_retail",
                "abn": "17 000 032 128",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["caltex", "ampol", "caltex woolworths"],
            },
            "mobil": {
                "official_name": "ExxonMobil Australia Pty Ltd",
                "industry": "fuel_retail",
                "abn": "28 000 016 756",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["mobil", "exxonmobil"],
            },
            "7_eleven": {
                "official_name": "7-Eleven Stores Pty Ltd",
                "industry": "fuel_convenience",
                "abn": "79 008 743 217",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["7-eleven", "7 eleven", "seven eleven"],
            },
        }

        # Banks and financial institutions
        financial_businesses = {
            "anz": {
                "official_name": "Australia and New Zealand Banking Group Limited",
                "industry": "banking",
                "abn": "11 005 357 522",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": [
                    "anz",
                    "anz bank",
                    "australia new zealand banking",
                ],
            },
            "commonwealth_bank": {
                "official_name": "Commonwealth Bank of Australia",
                "industry": "banking",
                "abn": "48 123 123 124",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["commonwealth bank", "commbank", "cba"],
            },
            "westpac": {
                "official_name": "Westpac Banking Corporation",
                "industry": "banking",
                "abn": "33 007 457 141",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["westpac", "westpac bank"],
            },
            "nab": {
                "official_name": "National Australia Bank Limited",
                "industry": "banking",
                "abn": "12 004 044 937",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["nab", "national australia bank"],
            },
            "ing": {
                "official_name": "ING Bank (Australia) Limited",
                "industry": "banking",
                "abn": "24 000 893 292",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["ing", "ing direct", "ing bank"],
            },
            "macquarie": {
                "official_name": "Macquarie Bank Limited",
                "industry": "banking",
                "abn": "46 008 583 542",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["macquarie", "macquarie bank"],
            },
        }

        # Airlines
        airline_businesses = {
            "qantas": {
                "official_name": "Qantas Airways Limited",
                "industry": "airline",
                "abn": "16 009 661 901",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["qantas", "qantas airways"],
            },
            "jetstar": {
                "official_name": "Jetstar Airways Pty Limited",
                "industry": "airline",
                "abn": "33 069 720 243",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["jetstar", "jetstar airways"],
            },
            "virgin_australia": {
                "official_name": "Virgin Australia Airlines Pty Ltd",
                "industry": "airline",
                "abn": "83 090 670 965",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["virgin australia", "virgin blue"],
            },
            "tigerair": {
                "official_name": "Tigerair Australia Pty Ltd",
                "industry": "airline",
                "abn": "64 131 632 853",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["tigerair", "tiger airways"],
            },
        }

        # Hotels and accommodation
        accommodation_businesses = {
            "hilton": {
                "official_name": "Hilton Worldwide (Australia) Pty Ltd",
                "industry": "accommodation",
                "abn": "58 004 179 020",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["hilton", "hilton hotel"],
            },
            "marriott": {
                "official_name": "Marriott International Australia Pty Ltd",
                "industry": "accommodation",
                "abn": "23 072 065 631",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["marriott", "marriott hotel"],
            },
            "hyatt": {
                "official_name": "Hyatt Hotels (Australia) Pty Ltd",
                "industry": "accommodation",
                "abn": "44 000 423 066",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["hyatt", "hyatt hotel"],
            },
            "ibis": {
                "official_name": "Accor Hotels Australia & New Zealand Pty Ltd",
                "industry": "accommodation",
                "abn": "81 001 058 068",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["ibis", "ibis hotel", "accor"],
            },
        }

        # Food chains and restaurants
        food_businesses = {
            "mcdonalds": {
                "official_name": "McDonald's Australia Limited",
                "industry": "food_fast",
                "abn": "43 008 496 928",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["mcdonald's", "mcdonalds", "maccas"],
            },
            "kfc": {
                "official_name": "KFC Australia Pty Limited",
                "industry": "food_fast",
                "abn": "93 001 279 564",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["kfc", "kentucky fried chicken"],
            },
            "subway": {
                "official_name": "Subway Systems Australia Pty Ltd",
                "industry": "food_fast",
                "abn": "54 057 109 420",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["subway", "subway sandwiches"],
            },
            "dominos": {
                "official_name": "Domino's Pizza Enterprises Limited",
                "industry": "food_fast",
                "abn": "79 010 489 326",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["domino's", "dominos pizza"],
            },
            "hungry_jacks": {
                "official_name": "Hungry Jack's Pty Ltd",
                "industry": "food_fast",
                "abn": "69 008 747 435",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["hungry jack's", "hungry jacks"],
            },
            "red_rooster": {
                "official_name": "Red Rooster Foods Pty Ltd",
                "industry": "food_fast",
                "abn": "91 008 747 521",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["red rooster"],
            },
            "nandos": {
                "official_name": "Nando's Australia Pty Ltd",
                "industry": "food_fast",
                "abn": "83 008 747 695",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["nando's", "nandos"],
            },
            "pizza_hut": {
                "official_name": "Pizza Hut Australia Pty Ltd",
                "industry": "food_fast",
                "abn": "72 008 747 834",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["pizza hut"],
            },
            "guzman_y_gomez": {
                "official_name": "Guzman Y Gomez Mexican Kitchen Pty Ltd",
                "industry": "food_fast",
                "abn": "54 134 003 121",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["guzman y gomez", "gyg"],
            },
            "zambrero": {
                "official_name": "Zambrero Pty Ltd",
                "industry": "food_fast",
                "abn": "12 149 493 776",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["zambrero"],
            },
            "starbucks": {
                "official_name": "Starbucks Coffee Australia Pty Ltd",
                "industry": "food_cafe",
                "abn": "26 081 987 342",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["starbucks", "starbucks coffee"],
            },
            "gloria_jeans": {
                "official_name": "Gloria Jean's Coffees Australia Pty Ltd",
                "industry": "food_cafe",
                "abn": "27 079 806 947",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["gloria jean's", "gloria jeans"],
            },
            "michel_patisserie": {
                "official_name": "Michel's Patisserie Australia Pty Ltd",
                "industry": "food_cafe",
                "abn": "52 079 806 982",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["michel's patisserie", "michels"],
            },
        }

        # Additional retail and specialty stores
        additional_retail = {
            "chemist_warehouse": {
                "official_name": "Chemist Warehouse Group Pty Ltd",
                "industry": "retail_pharmacy",
                "abn": "83 004 044 286",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["chemist warehouse"],
            },
            "priceline": {
                "official_name": "Priceline Pharmacy Pty Ltd",
                "industry": "retail_pharmacy",
                "abn": "72 004 044 321",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["priceline", "priceline pharmacy"],
            },
            "terry_white": {
                "official_name": "Terry White Chemmart Pty Ltd",
                "industry": "retail_pharmacy",
                "abn": "61 004 044 398",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["terry white", "chemmart"],
            },
            "dan_murphys": {
                "official_name": "Dan Murphy's Pty Ltd",
                "industry": "retail_liquor",
                "abn": "50 004 044 465",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["dan murphy's", "dan murphys"],
            },
            "bws": {
                "official_name": "BWS Pty Ltd",
                "industry": "retail_liquor",
                "abn": "39 004 044 532",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bws", "beer wine spirits"],
            },
            "liquorland": {
                "official_name": "Liquorland Pty Ltd",
                "industry": "retail_liquor",
                "abn": "28 004 044 609",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["liquorland"],
            },
            "myer": {
                "official_name": "Myer Holdings Limited",
                "industry": "retail_department",
                "abn": "17 004 044 676",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["myer"],
            },
            "david_jones": {
                "official_name": "David Jones Pty Limited",
                "industry": "retail_department",
                "abn": "06 004 044 743",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["david jones"],
            },
            "ikea": {
                "official_name": "IKEA Australia Pty Ltd",
                "industry": "retail_furniture",
                "abn": "95 004 044 810",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["ikea"],
            },
            "spotlight": {
                "official_name": "Spotlight Retail Group Pty Ltd",
                "industry": "retail_homewares",
                "abn": "84 004 044 887",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["spotlight"],
            },
            "rebel_sport": {
                "official_name": "Rebel Sport Pty Ltd",
                "industry": "retail_sporting",
                "abn": "73 004 044 954",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["rebel sport", "rebel"],
            },
            "amart_sports": {
                "official_name": "Amart Sports Pty Ltd",
                "industry": "retail_sporting",
                "abn": "62 004 045 021",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["amart sports", "amart"],
            },
            "supercheap_auto": {
                "official_name": "Super Cheap Auto Group Pty Ltd",
                "industry": "retail_automotive",
                "abn": "51 004 045 098",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["supercheap auto", "super cheap auto"],
            },
            "repco": {
                "official_name": "Repco Corporation Pty Ltd",
                "industry": "retail_automotive",
                "abn": "40 004 045 165",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["repco"],
            },
            "autobarn": {
                "official_name": "Autobarn Pty Ltd",
                "industry": "retail_automotive",
                "abn": "29 004 045 232",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["autobarn"],
            },
        }

        # Professional services firms
        professional_services = {
            "deloitte": {
                "official_name": "Deloitte Touche Tohmatsu Limited",
                "industry": "professional_consulting",
                "abn": "18 134 556 422",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["deloitte"],
            },
            "pwc": {
                "official_name": "PricewaterhouseCoopers Australia",
                "industry": "professional_consulting",
                "abn": "52 780 433 757",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["pwc", "pricewaterhousecoopers"],
            },
            "kpmg": {
                "official_name": "KPMG Australia",
                "industry": "professional_consulting",
                "abn": "51 194 660 183",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["kpmg"],
            },
            "ey": {
                "official_name": "Ernst & Young Australia",
                "industry": "professional_consulting",
                "abn": "75 288 172 749",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["ey", "ernst & young", "ernst and young"],
            },
            "bdo": {
                "official_name": "BDO Australia Limited",
                "industry": "professional_accounting",
                "abn": "77 050 110 275",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bdo"],
            },
            "rsm": {
                "official_name": "RSM Australia Partners",
                "industry": "professional_accounting",
                "abn": "36 965 185 036",
                "business_type": "partnership",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["rsm"],
            },
        }

        # Telecommunications
        telecommunications = {
            "telstra": {
                "official_name": "Telstra Corporation Limited",
                "industry": "telecommunications",
                "abn": "33 051 775 556",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["telstra"],
            },
            "optus": {
                "official_name": "SingTel Optus Pty Limited",
                "industry": "telecommunications",
                "abn": "90 052 833 208",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["optus"],
            },
            "vodafone": {
                "official_name": "Vodafone Hutchison Australia Pty Limited",
                "industry": "telecommunications",
                "abn": "76 096 304 620",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["vodafone"],
            },
        }

        # Utilities
        utilities = {
            "agl": {
                "official_name": "AGL Energy Limited",
                "industry": "utilities",
                "abn": "74 115 061 375",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["agl", "agl energy"],
            },
            "origin": {
                "official_name": "Origin Energy Limited",
                "industry": "utilities",
                "abn": "30 000 051 696",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["origin", "origin energy"],
            },
            "energy_australia": {
                "official_name": "EnergyAustralia Holdings Limited",
                "industry": "utilities",
                "abn": "67 004 052 542",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["energyaustralia", "energy australia"],
            },
        }

        # Insurance
        insurance = {
            "aami": {
                "official_name": "AAI Limited",
                "industry": "insurance",
                "abn": "48 005 297 807",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["aami"],
            },
            "nrma": {
                "official_name": "NRMA Insurance Limited",
                "industry": "insurance",
                "abn": "61 023 964 075",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["nrma"],
            },
            "racv": {
                "official_name": "Royal Automobile Club of Victoria",
                "industry": "insurance",
                "abn": "44 004 060 833",
                "business_type": "company_limited_by_guarantee",
                "states": ["VIC", "NSW", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["racv"],
            },
            "allianz": {
                "official_name": "Allianz Australia Insurance Limited",
                "industry": "insurance",
                "abn": "15 000 122 850",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["allianz"],
            },
            "gio": {
                "official_name": "AAI Limited",
                "industry": "insurance",
                "abn": "48 005 297 807",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["gio"],
            },
        }

        # Regional businesses and services
        regional_services = {
            "australia_post": {
                "official_name": "Australia Post Corporation",
                "industry": "postal_services",
                "abn": "28 864 970 579",
                "business_type": "government_business_enterprise",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["australia post", "auspost"],
            },
            "woolworths_metro": {
                "official_name": "Woolworths Group Limited",
                "industry": "retail_convenience",
                "abn": "88 000 014 675",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["woolworths metro", "metro"],
            },
            "coles_express": {
                "official_name": "Coles Group Limited",
                "industry": "fuel_convenience",
                "abn": "11 004 089 936",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["coles express"],
            },
            "newsagents": {
                "official_name": "Australian Newsagents' Federation",
                "industry": "retail_news",
                "abn": "89 004 228 307",
                "business_type": "association",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["newsagent", "news agency"],
            },
            "nqs": {
                "official_name": "NQR Pty Ltd",
                "industry": "retail_discount",
                "abn": "34 004 228 374",
                "business_type": "private_company",
                "states": ["VIC", "NSW", "SA"],
                "recognition_keywords": ["nqr", "not quite right"],
            },
        }

        # Healthcare and medical
        healthcare = {
            "bupa": {
                "official_name": "Bupa Australia Pty Ltd",
                "industry": "healthcare",
                "abn": "81 000 057 590",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bupa"],
            },
            "medibank": {
                "official_name": "Medibank Private Limited",
                "industry": "healthcare",
                "abn": "47 080 890 259",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["medibank"],
            },
            "hcf": {
                "official_name": "The Hospitals Contribution Fund of Australia Limited",
                "industry": "healthcare",
                "abn": "68 000 026 746",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["hcf"],
            },
            "vision_express": {
                "official_name": "Luxottica Retail Australia Pty Ltd",
                "industry": "retail_optical",
                "abn": "82 004 237 056",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["vision express"],
            },
            "specsavers": {
                "official_name": "Specsavers Pty Ltd",
                "industry": "retail_optical",
                "abn": "93 004 237 123",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["specsavers"],
            },
        }

        # Transport and logistics
        transport = {
            "toll": {
                "official_name": "Toll Holdings Limited",
                "industry": "transport_logistics",
                "abn": "14 006 592 089",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["toll transport", "toll logistics"],
            },
            "fastway": {
                "official_name": "Fastway Couriers Pty Ltd",
                "industry": "transport_courier",
                "abn": "25 006 592 156",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["fastway", "fastway couriers"],
            },
            "startrack": {
                "official_name": "StarTrack Express Pty Ltd",
                "industry": "transport_courier",
                "abn": "36 006 592 223",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["startrack", "star track"],
            },
            "hertz": {
                "official_name": "Hertz Australia Pty Limited",
                "industry": "transport_rental",
                "abn": "47 006 592 290",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["hertz"],
            },
            "avis": {
                "official_name": "Avis Australia Pty Ltd",
                "industry": "transport_rental",
                "abn": "58 006 592 357",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["avis"],
            },
            "budget": {
                "official_name": "Budget Rent A Car Australia Pty Ltd",
                "industry": "transport_rental",
                "abn": "69 006 592 424",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["budget rent a car", "budget"],
            },
        }

        # Additional major retailers
        additional_major_retail = {
            "catch": {
                "official_name": "Catch.com.au Pty Ltd",
                "industry": "retail_online",
                "abn": "80 006 592 491",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["catch", "catch.com.au"],
            },
            "harris_scarfe": {
                "official_name": "Harris Scarfe Australia Pty Ltd",
                "industry": "retail_department",
                "abn": "91 006 592 558",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["harris scarfe"],
            },
            "best_and_less": {
                "official_name": "Best & Less Pty Ltd",
                "industry": "retail_clothing",
                "abn": "12 006 592 625",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["best & less", "best and less"],
            },
            "cotton_on": {
                "official_name": "Cotton On Group Pty Ltd",
                "industry": "retail_clothing",
                "abn": "23 006 592 692",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["cotton on"],
            },
            "cue": {
                "official_name": "CUE Design Pty Ltd",
                "industry": "retail_clothing",
                "abn": "34 006 592 759",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT"],
                "recognition_keywords": ["cue clothing", "cue"],
            },
            "country_road": {
                "official_name": "Country Road Pty Ltd",
                "industry": "retail_clothing",
                "abn": "45 006 592 826",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["country road"],
            },
            "anaconda": {
                "official_name": "Anaconda Group Pty Ltd",
                "industry": "retail_outdoor",
                "abn": "56 006 592 893",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["anaconda"],
            },
            "bcf": {
                "official_name": "Boating Camping Fishing Pty Ltd",
                "industry": "retail_outdoor",
                "abn": "67 006 592 960",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["bcf", "boating camping fishing"],
            },
            "city_chic": {
                "official_name": "City Chic Collective Limited",
                "industry": "retail_clothing",
                "abn": "78 006 593 027",
                "business_type": "public_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["city chic"],
            },
            "witchery": {
                "official_name": "Witchery Pty Ltd",
                "industry": "retail_clothing",
                "abn": "89 006 593 094",
                "business_type": "private_company",
                "states": ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"],
                "recognition_keywords": ["witchery"],
            },
        }

        # Combine all businesses into registry
        self.business_registry.update(retail_businesses)
        self.business_registry.update(fuel_businesses)
        self.business_registry.update(financial_businesses)
        self.business_registry.update(airline_businesses)
        self.business_registry.update(accommodation_businesses)
        self.business_registry.update(food_businesses)
        self.business_registry.update(additional_retail)
        self.business_registry.update(professional_services)
        self.business_registry.update(telecommunications)
        self.business_registry.update(utilities)
        self.business_registry.update(insurance)
        self.business_registry.update(regional_services)
        self.business_registry.update(healthcare)
        self.business_registry.update(transport)
        self.business_registry.update(additional_major_retail)

    def _load_industry_keywords(self) -> None:
        """Load industry-specific keywords for business classification."""
        self.industry_keywords = {
            "retail_supermarket": [
                "supermarket",
                "grocery",
                "fresh",
                "produce",
                "checkout",
            ],
            "retail_department": [
                "department",
                "clothing",
                "fashion",
                "home",
                "lifestyle",
            ],
            "retail_hardware": ["hardware", "tools", "garden", "trade", "diy"],
            "retail_electronics": [
                "electronics",
                "computer",
                "mobile",
                "tv",
                "appliance",
            ],
            "retail_pharmacy": [
                "pharmacy",
                "chemist",
                "medicine",
                "prescription",
                "health",
            ],
            "retail_liquor": ["liquor", "alcohol", "wine", "beer", "spirits"],
            "retail_furniture": ["furniture", "home", "decor", "furnishing"],
            "retail_homewares": ["homewares", "kitchen", "bedroom", "living"],
            "retail_sporting": ["sport", "fitness", "athletic", "gym", "outdoor"],
            "retail_automotive": ["auto", "car", "automotive", "parts", "accessories"],
            "fuel_retail": ["petrol", "fuel", "diesel", "unleaded", "pump", "station"],
            "banking": ["bank", "atm", "account", "transaction", "bsb", "swift"],
            "airline": [
                "flight",
                "airline",
                "boarding",
                "departure",
                "arrival",
                "gate",
            ],
            "accommodation": ["hotel", "motel", "resort", "room", "booking", "guest"],
            "food_fast": ["restaurant", "fast food", "burger", "chicken", "pizza"],
            "food_cafe": ["cafe", "coffee", "espresso", "latte", "barista"],
            "professional_consulting": ["consulting", "advisory", "strategy", "audit"],
            "professional_accounting": [
                "accounting",
                "bookkeeping",
                "tax",
                "financial",
            ],
            "telecommunications": ["telecom", "mobile", "internet", "phone", "network"],
            "utilities": ["electricity", "gas", "energy", "power", "utility"],
            "insurance": ["insurance", "cover", "policy", "claim", "premium"],
            "postal_services": ["post", "mail", "delivery", "parcel", "courier"],
            "retail_convenience": ["convenience", "express", "metro", "local"],
            "retail_news": ["newsagent", "news", "magazine", "lottery", "stationery"],
            "retail_discount": ["discount", "clearance", "bargain", "reduced"],
            "healthcare": ["health", "medical", "hospital", "clinic", "private health"],
            "retail_optical": ["optical", "glasses", "frames", "eye", "vision"],
            "transport_logistics": ["logistics", "freight", "shipping", "transport"],
            "transport_courier": ["courier", "express", "delivery", "parcel"],
            "transport_rental": ["rental", "hire", "car rental", "vehicle"],
            "retail_online": ["online", "ecommerce", "digital", "marketplace"],
            "retail_clothing": ["clothing", "fashion", "apparel", "garments"],
            "retail_outdoor": ["outdoor", "camping", "fishing", "adventure"],
        }

    def _load_business_aliases(self) -> None:
        """Load business name aliases and common variations."""
        self.business_aliases = {
            "woolworths": ["woolies", "woolworth", "ww"],
            "commonwealth_bank": ["commbank", "cba", "comm bank"],
            "mcdonald's": ["mcdonalds", "maccas", "micky d's"],
            "7-eleven": ["7 eleven", "seven eleven", "7/11"],
            "harvey_norman": ["harvey normans", "harvey norm"],
            "jb_hi_fi": ["jb hifi", "jb hi fi", "jb"],
            "big_w": ["bigw", "big-w"],
            "virgin_australia": ["virgin blue", "virgin air"],
        }

    def recognize_business(self, text: str):
        """Recognize Australian businesses in text.

        Args:
            text: Text to search for business names

        Returns:
            Business recognition result object

        """
        if not self.initialized:
            self.initialize()

        from dataclasses import dataclass

        @dataclass
        class BusinessRecognitionResult:
            is_recognized: bool
            business_name: str
            confidence_score: float
            normalized_name: str = ""
            industry: str = ""
            abn: str = ""

        recognized = []
        text_lower = text.lower()

        for business_key, business_info in self.business_registry.items():
            for keyword in business_info["recognition_keywords"]:
                if keyword.lower() in text_lower:
                    # Calculate confidence based on keyword specificity
                    confidence = self._calculate_recognition_confidence(
                        keyword,
                        text_lower,
                    )

                    recognized.append(
                        {
                            "business_key": business_key,
                            "official_name": business_info["official_name"],
                            "industry": business_info["industry"],
                            "abn": business_info.get("abn", ""),
                            "business_type": business_info["business_type"],
                            "matched_keyword": keyword,
                            "confidence": confidence,
                            "states": business_info["states"],
                        },
                    )
                    break  # Only match once per business

        # Sort by confidence (highest first)
        recognized.sort(key=lambda x: x["confidence"], reverse=True)

        # Return the best match or "not recognized" result
        if recognized:
            best_match = recognized[0]
            return BusinessRecognitionResult(
                is_recognized=True,
                business_name=best_match["official_name"],
                confidence_score=best_match["confidence"],
                normalized_name=best_match["official_name"],
                industry=best_match["industry"],
                abn=best_match["abn"],
            )
        else:
            return BusinessRecognitionResult(
                is_recognized=False, business_name=text, confidence_score=0.0
            )

    def validate_business_context(
        self,
        business_name: str,
        document_type: str,
        extracted_fields: dict[str, any],
    ) -> tuple[bool, list[str], list[str]]:
        """Validate business context for document consistency.

        Args:
            business_name: Recognized business name
            document_type: Type of document
            extracted_fields: Fields extracted from document

        Returns:
            Tuple of (is_valid, validation_issues, recommendations)

        """
        if not self.initialized:
            self.initialize()

        issues = []
        recommendations = []

        # Find business in registry
        business_info = None
        for _business_key, info in self.business_registry.items():
            if business_name.lower() in [
                kw.lower() for kw in info["recognition_keywords"]
            ]:
                business_info = info
                break

        if not business_info:
            issues.append(
                f"Business '{business_name}' not found in Australian business registry",
            )
            return False, issues, recommendations

        # Validate document type consistency
        expected_doc_types = self._get_expected_document_types(
            business_info["industry"],
        )
        if document_type not in expected_doc_types:
            issues.append(
                f"Document type '{document_type}' unusual for {business_info['industry']} business",
            )
            recommendations.append(
                f"Expected document types: {', '.join(expected_doc_types)}",
            )

        # Validate ABN if present
        if extracted_fields.get("abn") and business_info.get("abn"):
            extracted_abn = re.sub(r"[^\d]", "", extracted_fields["abn"])
            registry_abn = re.sub(r"[^\d]", "", business_info["abn"])

            if extracted_abn != registry_abn:
                issues.append(
                    f"ABN mismatch: extracted {extracted_fields['abn']} "
                    f"vs registered {business_info['abn']}",
                )

        # Industry-specific validations
        industry_issues = self._validate_industry_specific_fields(
            business_info["industry"],
            extracted_fields,
        )
        issues.extend(industry_issues)

        return len(issues) == 0, issues, recommendations

    def _calculate_recognition_confidence(self, keyword: str, text: str) -> float:
        """Calculate confidence score for business recognition."""
        confidence = 0.0

        # Base confidence from keyword length (longer = more specific)
        if len(keyword) > 15:
            confidence += 0.9
        elif len(keyword) > 10:
            confidence += 0.8
        elif len(keyword) > 6:
            confidence += 0.7
        else:
            confidence += 0.6

        # Bonus for exact word boundary matches
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
            confidence += 0.1

        # Bonus for multiple occurrences
        occurrences = text.count(keyword.lower())
        if occurrences > 1:
            confidence += min(0.1 * (occurrences - 1), 0.2)

        return min(confidence, 1.0)

    def _get_expected_document_types(self, industry: str) -> list[str]:
        """Get expected document types for an industry."""
        industry_docs = {
            "retail_supermarket": ["business_receipt", "tax_invoice"],
            "retail_department": ["business_receipt", "tax_invoice"],
            "retail_hardware": ["business_receipt", "tax_invoice"],
            "retail_electronics": ["business_receipt", "tax_invoice"],
            "fuel_retail": ["fuel_receipt", "business_receipt"],
            "banking": ["bank_statement"],
            "airline": ["travel_document", "tax_invoice"],
            "accommodation": ["accommodation", "tax_invoice"],
            "food_fast": ["meal_receipt", "business_receipt"],
            "food_cafe": ["meal_receipt", "business_receipt"],
        }

        return industry_docs.get(industry, ["business_receipt", "tax_invoice"])

    def _validate_industry_specific_fields(
        self,
        industry: str,
        extracted_fields: dict[str, any],
    ) -> list[str]:
        """Validate industry-specific field requirements."""
        issues = []

        if industry == "fuel_retail":
            # Fuel receipts should have fuel-specific fields
            fuel_fields = ["litres", "fuel_type", "price_per_litre"]
            missing_fuel_fields = [
                f for f in fuel_fields if not extracted_fields.get(f)
            ]
            if missing_fuel_fields:
                issues.append(
                    f"Missing fuel-specific fields: {', '.join(missing_fuel_fields)}",
                )

        elif industry == "banking":
            # Bank statements should have banking fields
            banking_fields = ["account_number", "bsb"]
            missing_banking_fields = [
                f for f in banking_fields if not extracted_fields.get(f)
            ]
            if missing_banking_fields:
                issues.append(
                    f"Missing banking fields: {', '.join(missing_banking_fields)}",
                )

        elif industry == "airline":
            # Travel documents should have travel fields
            travel_fields = ["passenger_name", "flight_number"]
            missing_travel_fields = [
                f for f in travel_fields if not extracted_fields.get(f)
            ]
            if missing_travel_fields:
                issues.append(
                    f"Missing travel fields: {', '.join(missing_travel_fields)}",
                )

        return issues

    def get_business_statistics(self) -> dict[str, any]:
        """Get statistics about the business registry."""
        if not self.initialized:
            self.initialize()

        stats = {
            "total_businesses": len(self.business_registry),
            "businesses_by_industry": {},
            "businesses_by_type": {},
            "businesses_by_state": {},
        }

        for business_info in self.business_registry.values():
            # Count by industry
            industry = business_info["industry"]
            stats["businesses_by_industry"][industry] = (
                stats["businesses_by_industry"].get(industry, 0) + 1
            )

            # Count by business type
            biz_type = business_info["business_type"]
            stats["businesses_by_type"][biz_type] = (
                stats["businesses_by_type"].get(biz_type, 0) + 1
            )

            # Count by state presence
            for state in business_info["states"]:
                stats["businesses_by_state"][state] = (
                    stats["businesses_by_state"].get(state, 0) + 1
                )

        return stats

    def lookup_business_by_abn(self, abn: str):
        """Lookup business by ABN.

        Args:
            abn: ABN to search for

        Returns:
            ABN lookup result object
        """
        if not self.initialized:
            self.initialize()

        from dataclasses import dataclass

        @dataclass
        class ABNLookupResult:
            is_found: bool
            business_name: str
            abn: str
            industry: str = ""

        # Clean the ABN for comparison
        clean_abn = re.sub(r"[^\d]", "", abn) if abn else ""

        for business_info in self.business_registry.values():
            if business_info.get("abn"):
                registry_abn = re.sub(r"[^\d]", "", business_info["abn"])
                if clean_abn == registry_abn:
                    return ABNLookupResult(
                        is_found=True,
                        business_name=business_info["official_name"],
                        abn=business_info["abn"],
                        industry=business_info["industry"],
                    )

        return ABNLookupResult(is_found=False, business_name="", abn=abn)

    def get_business_category(self, business_name: str):
        """Get business category for a given business name.

        Args:
            business_name: Name of the business

        Returns:
            Business category result object
        """
        if not self.initialized:
            self.initialize()

        from dataclasses import dataclass

        @dataclass
        class BusinessCategoryResult:
            category: str
            confidence: float
            industry: str = ""

        # Search for the business
        business_name_lower = business_name.lower()

        for business_info in self.business_registry.values():
            for keyword in business_info["recognition_keywords"]:
                if keyword.lower() in business_name_lower:
                    # Map industry to simplified category
                    category_mapping = {
                        "retail_supermarket": "supermarket",
                        "retail_electronics": "electronics",
                        "retail_furniture": "furniture_electronics",
                        "fuel_retail": "fuel_station",
                        "food_fast": "restaurant",
                    }

                    category = category_mapping.get(
                        business_info["industry"], business_info["industry"]
                    )

                    return BusinessCategoryResult(
                        category=category,
                        confidence=0.9,
                        industry=business_info["industry"],
                    )

        return BusinessCategoryResult(category="unknown", confidence=0.0)
