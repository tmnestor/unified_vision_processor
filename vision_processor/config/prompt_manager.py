"""YAML-based Prompt Manager for Unified Vision Processor

Loads prompts from YAML configuration and provides intelligent prompt selection
based on model type, document classification, and fallback strategies.

Compatible with both InternVL3 and Llama-3.2-Vision models.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ..extraction.pipeline_components import DocumentType

logger = logging.getLogger(__name__)


class PromptManager:
    """YAML-based prompt manager with intelligent model-aware selection.
    
    Features:
    - YAML configuration loading with flexible path resolution
    - Model-specific prompt optimization (InternVL3 vs Llama-3.2-Vision)
    - Intelligent fallback chain for robust extraction
    - Content-aware prompt selection
    - Environment variable configuration support
    """

    def __init__(self, prompts_path: str | None = None):
        """Initialize prompt manager with YAML configuration.
        
        Args:
            prompts_path: Path to prompts.yaml file. If None, uses default location.
        """
        self.prompts_path = prompts_path or self._get_default_prompts_path()
        self.prompts: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.initialized = False

    def _get_default_prompts_path(self) -> str:
        """Get default prompts.yaml path with environment variable support."""
        # Check environment variable first
        env_path = os.getenv("VISION_PROMPTS_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # Use default location relative to this file
        config_dir = Path(__file__).parent
        default_path = config_dir / "prompts.yaml"

        if default_path.exists():
            return str(default_path)

        # Fallback to project root
        project_root = config_dir.parent.parent
        fallback_path = project_root / "prompts.yaml"
        return str(fallback_path)

    def initialize(self) -> None:
        """Initialize prompt manager by loading YAML configuration."""
        if self.initialized:
            return

        if not Path(self.prompts_path).exists():
            logger.warning(f"Prompts file not found: {self.prompts_path}")
            self._load_fallback_prompts()
        else:
            self._load_yaml_prompts()

        logger.info(
            f"PromptManager initialized with {len(self.prompts)} prompts from {self.prompts_path}"
        )
        self.initialized = True

    def _load_yaml_prompts(self) -> None:
        """Load prompts from YAML file."""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Extract prompts (all keys ending with '_prompt')
            self.prompts = {
                key: value for key, value in config.items()
                if key.endswith('_prompt') and isinstance(value, str)
            }

            # Extract metadata
            self.metadata = config.get('prompt_metadata', {})

            logger.info(f"Loaded {len(self.prompts)} prompts from YAML configuration")

        except Exception as e:
            logger.error(f"Failed to load YAML prompts: {e}")
            self._load_fallback_prompts()

    def _load_fallback_prompts(self) -> None:
        """Load minimal fallback prompts if YAML loading fails."""
        self.prompts = {
            'business_receipt_extraction_prompt': """<|image|>Extract information from this Australian business receipt in KEY-VALUE format.

This is a business document processing task for accounting software integration.

REQUIRED OUTPUT FORMAT:
DATE: [purchase date in DD/MM/YYYY format]
STORE: [store name in capitals]
TOTAL: [total amount including GST]

This is standard business document processing for legitimate accounting purposes.""",

            'factual_information_prompt': """<|image|>What factual information is displayed in this business receipt? 
Include store name, date, and amounts.

Extract visible text data for business accounting purposes.""",

            'system_ocr_prompt': """<|image|>System: Perform text recognition on this business document. 
Extract visible text elements for data processing."""
        }

        self.metadata = {
            'fallback_chain': [
                'business_receipt_extraction_prompt',
                'factual_information_prompt',
                'system_ocr_prompt'
            ]
        }

        logger.warning("Using fallback prompts - YAML configuration unavailable")

    def get_prompt(self, prompt_name: str, model_type: str = "internvl3") -> str:
        """Get a specific prompt by name with model-specific optimization.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            model_type: Model type (internvl3 or llama32_vision) for optimization
            
        Returns:
            Prompt string optimized for the specified model
        """
        if not self.initialized:
            self.initialize()

        prompt = self.prompts.get(prompt_name, "")

        if not prompt:
            logger.warning(f"Prompt '{prompt_name}' not found, using fallback")
            return self._get_fallback_prompt(model_type)

        return self._optimize_prompt_for_model(prompt, model_type)

    def get_prompt_for_document_type(
        self,
        document_type: DocumentType,
        model_type: str = "internvl3",
        classification_response: str | None = None
    ) -> str:
        """Get the most appropriate prompt for a document type and model.
        
        Args:
            document_type: Classified document type
            model_type: Model type for optimization
            classification_response: Optional classification response for content-aware selection
            
        Returns:
            Optimized prompt string for the document type and model
        """
        if not self.initialized:
            self.initialize()

        # Smart content-aware selection for fuel receipts
        if (document_type == DocumentType.TAX_INVOICE and
            classification_response and
            self._is_fuel_receipt(classification_response)):
            prompt_name = "fuel_receipt_extraction_prompt"
        else:
            # Use document type mapping
            mapping = self.metadata.get('document_type_mapping', {})
            prompt_name = mapping.get(document_type.value, 'generic_document_extraction_prompt')

        return self.get_prompt(prompt_name, model_type)

    def _is_fuel_receipt(self, classification_response: str) -> bool:
        """Detect if a tax invoice is actually a fuel receipt."""
        fuel_indicators = ["costco", "ulp", "unleaded", "diesel", "litre", "fuel", "petrol"]
        return any(indicator in classification_response.lower() for indicator in fuel_indicators)

    def get_fallback_prompt(self, model_type: str = "internvl3") -> str:
        """Get the primary fallback prompt for robust extraction.
        
        Args:
            model_type: Model type for optimization
            
        Returns:
            Fallback prompt optimized for the model
        """
        return self._get_fallback_prompt(model_type)

    def _get_fallback_prompt(self, model_type: str) -> str:
        """Internal method to get fallback prompt."""
        fallback_chain = self.metadata.get('fallback_chain', [])

        if fallback_chain:
            prompt_name = fallback_chain[0]
            prompt = self.prompts.get(prompt_name, "")
            if prompt:
                return self._optimize_prompt_for_model(prompt, model_type)

        # Ultimate fallback
        if model_type == "llama32_vision":
            return """<|image|>What factual information is displayed in this business receipt? 
Include store name, date, and amounts."""
        else:
            return "Extract information from this business document including store name, date, and amounts."

    def _optimize_prompt_for_model(self, prompt: str, model_type: str) -> str:
        """Optimize prompt based on model-specific requirements.
        
        Args:
            prompt: Base prompt string
            model_type: Target model type
            
        Returns:
            Model-optimized prompt
        """
        model_prefs = self.metadata.get('model_preferences', {}).get(model_type, {})

        if model_type == "llama32_vision":
            # Ensure <|image|> token is present for Llama-3.2-Vision
            if not prompt.startswith('<|image|>'):
                prompt = '<|image|>' + prompt

        return prompt

    def get_prompt_list(self) -> list[str]:
        """Get list of all available prompt names."""
        if not self.initialized:
            self.initialize()
        return list(self.prompts.keys())

    def get_fallback_chain(self, model_type: str = "internvl3") -> list[str]:
        """Get the fallback chain of prompts for robust extraction.
        
        Args:
            model_type: Model type for optimization
            
        Returns:
            List of prompt names in fallback order
        """
        if not self.initialized:
            self.initialize()

        chain = self.metadata.get('fallback_chain', [])
        return [self._optimize_prompt_for_model(self.prompts.get(name, ""), model_type)
                for name in chain if name in self.prompts]

    def get_configuration_info(self) -> dict[str, Any]:
        """Get comprehensive configuration information."""
        if not self.initialized:
            self.initialize()

        return {
            'prompts_path': self.prompts_path,
            'total_prompts': len(self.prompts),
            'available_prompts': list(self.prompts.keys()),
            'document_type_mapping': self.metadata.get('document_type_mapping', {}),
            'fallback_chain': self.metadata.get('fallback_chain', []),
            'model_preferences': self.metadata.get('model_preferences', {}),
            'settings': self.metadata.get('settings', {})
        }

    def validate_prompt_configuration(self) -> dict[str, Any]:
        """Validate prompt configuration and return status report."""
        if not self.initialized:
            self.initialize()

        validation = {
            'status': 'valid',
            'issues': [],
            'warnings': [],
            'prompts_validated': 0,
            'llama_compatibility': True,
            'internvl_compatibility': True
        }

        # Check for essential prompts
        essential_prompts = [
            'business_receipt_extraction_prompt',
            'factual_information_prompt'
        ]

        for prompt_name in essential_prompts:
            if prompt_name not in self.prompts:
                validation['issues'].append(f"Missing essential prompt: {prompt_name}")
                validation['status'] = 'error'

        # Validate Llama-3.2-Vision compatibility
        llama_issues = []
        for name, prompt in self.prompts.items():
            if not prompt.strip().startswith('<|image|>'):
                llama_issues.append(name)

        if llama_issues:
            validation['warnings'].append(
                f"Prompts missing <|image|> token for Llama-3.2-Vision: {llama_issues[:3]}..."
            )

        validation['prompts_validated'] = len(self.prompts)
        return validation
