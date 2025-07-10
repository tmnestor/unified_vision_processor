"""
Prompt Factory

Unified prompt management system combining InternVL's 47 specialized prompts
with Llama's 13 ATO-compliant prompts for Australian tax document processing.
"""

import logging
from typing import Any, Dict, List, Optional

from ..extraction.pipeline_components import DocumentType
from .internvl_prompts import InternVLPrompts
from .llama_prompts import LlamaPrompts

logger = logging.getLogger(__name__)


class PromptFactory:
    """
    Factory for managing unified prompt selection and optimization.

    Features:
    - 47 InternVL specialized prompts for document processing
    - 13 Llama ATO-compliant prompts for Australian tax compliance
    - Dynamic prompt selection based on document type and features
    - Highlight-aware prompt variations
    - Performance optimization tracking
    """

    def __init__(self, config: Any = None):
        self.config = config or {}
        self.initialized = False

        # Prompt libraries
        self.internvl_prompts = InternVLPrompts()
        self.llama_prompts = LlamaPrompts()

        # Prompt performance tracking
        self.prompt_performance: Dict[str, Dict[str, float]] = {}

        # Feature flags
        self.enable_highlight_prompts = getattr(config, "highlight_detection", True)
        self.enable_ato_compliance = getattr(config, "ato_compliance", True)
        self.enable_prompt_optimization = getattr(config, "prompt_optimization", True)

    def initialize(self) -> None:
        """Initialize the prompt factory with all prompt libraries."""
        if self.initialized:
            return

        # Initialize prompt libraries
        self.internvl_prompts.initialize()
        self.llama_prompts.initialize()

        # Load prompt performance data if available
        self._load_prompt_performance()

        logger.info(
            f"PromptFactory initialized with {self.get_total_prompt_count()} prompts"
        )
        self.initialized = True

    def get_prompt(
        self,
        document_type: DocumentType,
        has_highlights: bool = False,
        extraction_quality: float = 0.0,
        prefer_ato_compliance: bool = True,
    ) -> str:
        """
        Get the optimal prompt for a given document type and context.

        Args:
            document_type: Type of document being processed
            has_highlights: Whether highlights were detected (InternVL feature)
            extraction_quality: Quality score from previous extraction attempts
            prefer_ato_compliance: Whether to prefer ATO-compliant prompts

        Returns:
            Optimized prompt string for the given context
        """
        if not self.initialized:
            self.initialize()

        # Determine prompt strategy
        prompt_strategy = self._determine_prompt_strategy(
            document_type, has_highlights, extraction_quality, prefer_ato_compliance
        )

        # Get base prompt based on strategy
        if prompt_strategy == "ato_compliance":
            base_prompt = self.llama_prompts.get_ato_prompt(document_type)
        elif prompt_strategy == "highlight_enhanced":
            base_prompt = self.internvl_prompts.get_highlight_prompt(document_type)
        elif prompt_strategy == "specialized":
            base_prompt = self.internvl_prompts.get_specialized_prompt(document_type)
        else:
            # Fallback to unified prompt
            base_prompt = self._get_unified_prompt(document_type)

        # Apply prompt enhancements
        enhanced_prompt = self._enhance_prompt(
            base_prompt, document_type, has_highlights, extraction_quality
        )

        # Track prompt usage for optimization
        if self.enable_prompt_optimization:
            self._track_prompt_usage(document_type, prompt_strategy)

        return enhanced_prompt

    def _determine_prompt_strategy(
        self,
        document_type: DocumentType,
        has_highlights: bool,
        extraction_quality: float,
        prefer_ato_compliance: bool,
    ) -> str:
        """Determine the best prompt strategy for the given context."""

        # If previous extraction quality was low, try different strategy
        if extraction_quality < 0.5:
            return "specialized"

        # For bank statements with highlights, use highlight-enhanced prompts
        if document_type == DocumentType.BANK_STATEMENT and has_highlights:
            return "highlight_enhanced"

        # For tax invoices and business receipts, prefer ATO compliance
        if (
            document_type in [DocumentType.TAX_INVOICE, DocumentType.BUSINESS_RECEIPT]
            and prefer_ato_compliance
        ):
            return "ato_compliance"

        # For fuel receipts and parking, use specialized prompts
        if document_type in [DocumentType.FUEL_RECEIPT, DocumentType.PARKING_TOLL]:
            return "specialized"

        # Check prompt performance history
        if self.enable_prompt_optimization:
            best_strategy = self._get_best_performing_strategy(document_type)
            if best_strategy:
                return best_strategy

        # Default to ATO compliance if enabled
        if prefer_ato_compliance and self.enable_ato_compliance:
            return "ato_compliance"

        return "specialized"

    def _get_unified_prompt(self, document_type: DocumentType) -> str:
        """Get a unified prompt combining best elements from both systems."""
        # Combine InternVL's technical precision with Llama's ATO compliance
        internvl_prompt = self.internvl_prompts.get_base_prompt(document_type)
        llama_prompt = self.llama_prompts.get_ato_prompt(document_type)

        # Create unified prompt structure
        unified_prompt = f"""You are an expert in Australian tax document processing.

{llama_prompt}

Additional Instructions:
{internvl_prompt}

Important Australian Tax Requirements:
- All dates must be in DD/MM/YYYY format
- GST rate is 10% in Australia
- ABN format is XX XXX XXX XXX (11 digits)
- All amounts in Australian dollars (AUD)

Extract all relevant information accurately and provide confidence scores."""

        return unified_prompt

    def _enhance_prompt(
        self,
        base_prompt: str,
        document_type: DocumentType,
        has_highlights: bool,
        extraction_quality: float,
    ) -> str:
        """Enhance the base prompt with context-specific information."""

        enhanced_prompt = base_prompt

        # Add highlight-specific instructions
        if has_highlights and self.enable_highlight_prompts:
            highlight_instructions = """
HIGHLIGHT DETECTION ACTIVE:
- Pay special attention to highlighted regions in the image
- Highlighted areas often contain key information like totals, dates, or important fields
- Cross-reference highlighted information with surrounding context
- Prioritize information from highlighted regions when there are conflicts
"""
            enhanced_prompt = base_prompt + highlight_instructions

        # Add quality improvement instructions for low-quality extractions
        if extraction_quality < 0.3:
            quality_instructions = """
EXTRACTION QUALITY IMPROVEMENT:
- Be extra careful with field identification
- Look for alternative field labels and formats
- Check for information in headers, footers, and margins
- Use context clues to validate extracted information
"""
            enhanced_prompt += quality_instructions

        # Add document-specific enhancements
        document_enhancements = self._get_document_specific_enhancements(document_type)
        if document_enhancements:
            enhanced_prompt += f"\n{document_enhancements}"

        return enhanced_prompt

    def _get_document_specific_enhancements(self, document_type: DocumentType) -> str:
        """Get document-type specific prompt enhancements."""

        enhancements = {
            DocumentType.FUEL_RECEIPT: """
FUEL RECEIPT SPECIFIC:
- Look for pump numbers, fuel type (unleaded/diesel/premium)
- Extract litres, price per litre, and validate calculations
- Identify Australian fuel stations (BP, Shell, Caltex, Ampol, Mobil, 7-Eleven)
""",
            DocumentType.BANK_STATEMENT: """
BANK STATEMENT SPECIFIC:
- Focus on transaction lists and work-related expenses
- Extract BSB (XXX-XXX format) and account numbers
- Identify Australian banks (ANZ, Commonwealth, Westpac, NAB)
- Categorize transactions for business expense purposes
""",
            DocumentType.TAX_INVOICE: """
TAX INVOICE SPECIFIC:
- Ensure "Tax Invoice" text is present
- Validate ABN format and GST calculations
- Extract supplier and customer details
- Verify 10% GST rate compliance
""",
        }

        return enhancements.get(document_type, "")

    def _track_prompt_usage(self, document_type: DocumentType, strategy: str) -> None:
        """Track prompt usage for performance optimization."""
        doc_type_str = document_type.value

        if doc_type_str not in self.prompt_performance:
            self.prompt_performance[doc_type_str] = {}

        if strategy not in self.prompt_performance[doc_type_str]:
            self.prompt_performance[doc_type_str][strategy] = 0.0

        # Increment usage count
        self.prompt_performance[doc_type_str][strategy] += 1

    def _get_best_performing_strategy(
        self, document_type: DocumentType
    ) -> Optional[str]:
        """Get the best performing prompt strategy for a document type."""
        doc_type_str = document_type.value

        if doc_type_str not in self.prompt_performance:
            return None

        strategies = self.prompt_performance[doc_type_str]
        if not strategies:
            return None

        # Return strategy with highest usage (simple optimization)
        return max(strategies, key=strategies.get)

    def _load_prompt_performance(self) -> None:
        """Load prompt performance data from storage."""
        # Placeholder for loading performance data
        # In a real implementation, this would load from a file or database
        logger.info("Prompt performance data loaded")

    def get_total_prompt_count(self) -> int:
        """Get total number of available prompts."""
        internvl_count = self.internvl_prompts.get_prompt_count()
        llama_count = self.llama_prompts.get_prompt_count()
        return internvl_count + llama_count

    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get comprehensive prompt system statistics."""
        return {
            "total_prompts": self.get_total_prompt_count(),
            "internvl_prompts": self.internvl_prompts.get_prompt_count(),
            "llama_prompts": self.llama_prompts.get_prompt_count(),
            "highlight_enabled": self.enable_highlight_prompts,
            "ato_compliance_enabled": self.enable_ato_compliance,
            "optimization_enabled": self.enable_prompt_optimization,
            "performance_data": self.prompt_performance,
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of available prompt strategies."""
        return ["ato_compliance", "highlight_enhanced", "specialized", "unified"]

    def optimize_prompts_for_document_type(
        self, document_type: DocumentType
    ) -> Dict[str, str]:
        """Get optimized prompts for all strategies for a document type."""
        strategies = {}

        for strategy in self.get_available_strategies():
            if strategy == "ato_compliance":
                strategies[strategy] = self.llama_prompts.get_ato_prompt(document_type)
            elif strategy == "highlight_enhanced":
                strategies[strategy] = self.internvl_prompts.get_highlight_prompt(
                    document_type
                )
            elif strategy == "specialized":
                strategies[strategy] = self.internvl_prompts.get_specialized_prompt(
                    document_type
                )
            else:  # unified
                strategies[strategy] = self._get_unified_prompt(document_type)

        return strategies
