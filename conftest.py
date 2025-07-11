"""Global pytest configuration for unified vision processor.

This file configures pytest markers and global settings for the test suite.
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "fast: marks tests as fast (can run on Mac M1)")
    config.addinivalue_line("markers", "slow: marks tests as slow (requires powerful hardware)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests (real components)")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU/development machine")
    config.addinivalue_line("markers", "unit: marks tests as unit tests (isolated components)")
    config.addinivalue_line("markers", "cli: marks tests as CLI-specific")
    config.addinivalue_line("markers", "cv: marks tests as computer vision-specific")
    config.addinivalue_line("markers", "prompt: marks tests as prompt system-specific")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Automatically add markers based on test file location."""
    for item in items:
        # Add CLI marker for CLI tests
        if "cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)

        # Add CV marker for computer vision tests
        if "computer_vision" in str(item.fspath) or "cv" in str(item.fspath):
            item.add_marker(pytest.mark.cv)

        # Add prompt marker for prompt tests
        if "prompt" in str(item.fspath):
            item.add_marker(pytest.mark.prompt)

        # Add fast marker for fast test files
        if "fast" in str(item.fspath):
            item.add_marker(pytest.mark.fast)

        # Add integration marker for integration test files
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# Global test configuration
pytest_plugins = [
    # Add any global plugins here
]
