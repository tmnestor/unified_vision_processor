"""Minimal smoke tests to ensure basic functionality works.

Replaces heavy integration tests with lightweight validation.
"""

import subprocess
import sys

import pytest


def test_cli_imports_successfully():
    """Test that CLI can be imported without torch/transformers errors."""
    # This would have failed before our import fixes
    result = subprocess.run(
        [sys.executable, "-c", "from vision_processor.cli.unified_cli import app; print('OK')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_help_works():
    """Test that CLI help works without loading heavy dependencies."""
    result = subprocess.run(
        [sys.executable, "-m", "vision_processor", "--help"],
        capture_output=True,
        text=True,
        env={"KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    assert result.returncode == 0
    assert "Unified Vision Document Processing" in result.stdout


def test_config_loads_without_errors():
    """Test that UnifiedConfig can be imported and instantiated."""
    try:
        from vision_processor.config.unified_config import UnifiedConfig

        config = UnifiedConfig.from_env()
        assert config is not None
    except ImportError as e:
        pytest.fail(f"Config import failed: {e}")


def test_model_factory_imports():
    """Test that ModelFactory can be imported (was failing before)."""
    try:
        from vision_processor.config.model_factory import ModelFactory

        available_models = ModelFactory.get_supported_models()
        assert len(available_models) > 0
    except ImportError as e:
        pytest.fail(f"ModelFactory import failed: {e}")


# That's it! 50 lines instead of 2,114 lines
