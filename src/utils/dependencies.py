"""Shared dependencies for the AI Blog Generation Team."""

from dataclasses import dataclass
from tavily import TavilyClient
import httpx


@dataclass
class SharedDependencies:
    """Shared dependencies across all agents."""
    http_client: httpx.AsyncClient
    tavily_client: TavilyClient
    max_iterations: int = 3
    quality_threshold: float = 7.0