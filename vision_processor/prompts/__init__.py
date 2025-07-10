"""Prompts Module

This module provides the unified prompt system combining InternVL's 47 specialized prompts
with Llama's 13 ATO-compliant prompts for a total of 60+ prompts optimized for Australian
tax document processing.
"""

from .internvl_prompts import InternVLPrompts
from .llama_prompts import LlamaPrompts
from .prompt_factory import PromptFactory
from .prompt_optimizer import PromptOptimizer
from .prompt_validator import PromptValidator

__all__ = [
    "InternVLPrompts",
    "LlamaPrompts",
    "PromptFactory",
    "PromptOptimizer",
    "PromptValidator",
]
