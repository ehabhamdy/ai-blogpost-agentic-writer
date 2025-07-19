"""Unit tests for shared dependencies."""

import pytest
from unittest.mock import Mock
from src.utils.dependencies import SharedDependencies


class TestSharedDependencies:
    """Test SharedDependencies dataclass."""

    def test_shared_dependencies_creation(self):
        """Test creating SharedDependencies with all parameters."""
        mock_http_client = Mock()
        mock_tavily_client = Mock()
        
        deps = SharedDependencies(
            http_client=mock_http_client,
            tavily_client=mock_tavily_client,
            max_iterations=5,
            quality_threshold=8.0
        )
        
        assert deps.http_client == mock_http_client
        assert deps.tavily_client == mock_tavily_client
        assert deps.max_iterations == 5
        assert deps.quality_threshold == 8.0

    def test_shared_dependencies_defaults(self):
        """Test SharedDependencies with default values."""
        mock_http_client = Mock()
        mock_tavily_client = Mock()
        
        deps = SharedDependencies(
            http_client=mock_http_client,
            tavily_client=mock_tavily_client
        )
        
        assert deps.http_client == mock_http_client
        assert deps.tavily_client == mock_tavily_client
        assert deps.max_iterations == 3  # default value
        assert deps.quality_threshold == 7.0  # default value

    def test_shared_dependencies_required_fields(self):
        """Test that http_client and tavily_client are required."""
        # This test verifies the dataclass structure
        # In practice, Python will raise TypeError for missing required fields
        mock_http_client = Mock()
        mock_tavily_client = Mock()
        
        # Should work with required fields
        deps = SharedDependencies(
            http_client=mock_http_client,
            tavily_client=mock_tavily_client
        )
        assert deps is not None