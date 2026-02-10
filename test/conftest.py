"""Pytest fixtures delegating to helpers.py."""

import tempfile

try:
    import pytest
except ImportError:
    # When running with unittest only, pytest fixtures aren't needed
    pass
else:
    from test.helpers import make_config, make_prod_config

    @pytest.fixture
    def default_cfg():
        """Small deterministic config for unit tests."""
        return make_config()

    @pytest.fixture
    def prod_cfg():
        """Production config loaded from JSON files."""
        return make_prod_config()

    @pytest.fixture
    def tmp_dir():
        """Temporary directory cleaned up after test."""
        with tempfile.TemporaryDirectory() as d:
            yield d
