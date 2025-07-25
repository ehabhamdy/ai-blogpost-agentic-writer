"""Integration tests for agent error handling and retry mechanisms."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_ai import ModelRetry

from src.agents.research_agent import ResearchAgent
from src.agents.writing_agent import WritingAgent
from src.agents.critique_agent import CritiqueAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.models.data_models import (
    ResearchOutput, 
    ResearchFinding, 
    BlogDraft, 
    CritiqueOutput,
    CritiqueFeedback,
    CritiqueSeverity
)
from src.utils.dependencies import SharedDependencies
from src.utils.exceptions import (
    ResearchError,
    WritingError,
    CritiqueError,
    OrchestrationError,
    APIError,
    TimeoutError,
    ValidationError
)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from pydantic_ai.models.test import TestModel
    return TestModel()


@pytest.fixture
def shared_dependencies():
    """Create shared dependencies for testing."""
    return SharedDependencies(
        http_client=MagicMock(),
        tavily_client=MagicMock(),
        max_iterations=3,
        quality_threshold=7.0
    )


@pytest.fixture
def sample_research_output():
    """Create sample research output for testing."""
    return ResearchOutput(
        topic="Test Topic",
        findings=[
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=0.8,
                category="study"
            )
        ],
        summary="Test research summary",
        confidence_level=0.8
    )


@pytest.fixture
def sample_blog_draft():
    """Create sample blog draft for testing."""
    return BlogDraft(
        title="Test Blog Post",
        introduction="This is a test introduction.",
        body_sections=["This is the first section.", "This is the second section."],
        conclusion="This is the test conclusion.",
        word_count=100
    )


@pytest.fixture
def sample_critique_output():
    """Create sample critique output for testing."""
    return CritiqueOutput(
        overall_quality=8.0,
        feedback_items=[
            CritiqueFeedback(
                section="introduction",
                issue="Could be more engaging",
                suggestion="Add a compelling hook",
                severity=CritiqueSeverity.MINOR
            )
        ],
        approval_status="approved",
        summary_feedback="Good overall quality with minor improvements needed."
    )


class TestResearchAgentErrorHandling:
    """Test error handling in ResearchAgent."""
    
    @pytest.mark.asyncio
    async def test_research_topic_empty_topic(self, mock_model, shared_dependencies):
        """Test research_topic with empty topic."""
        agent = ResearchAgent(mock_model)
        
        with pytest.raises(ResearchError, match="Topic cannot be empty"):
            await agent.research_topic("", shared_dependencies)
        
        with pytest.raises(ResearchError, match="Topic cannot be empty"):
            await agent.research_topic("   ", shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_research_topic_api_error(self, mock_model, shared_dependencies):
        """Test research_topic with API error."""
        agent = ResearchAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.research_topic("test topic", shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_research_topic_timeout_error(self, mock_model, shared_dependencies):
        """Test research_topic with timeout error."""
        agent = ResearchAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("timeout occurred"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.research_topic("test topic", shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_research_topic_general_error(self, mock_model, shared_dependencies):
        """Test research_topic with general error."""
        agent = ResearchAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=ValueError("general error"))
        
        with pytest.raises(ResearchError, match="Research failed for topic"):
            await agent.research_topic("test topic", shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_search_web_graceful_degradation(self, mock_model, shared_dependencies):
        """Test _search_web with graceful degradation."""
        agent = ResearchAgent(mock_model)
        
        # Mock Tavily client to raise an error
        shared_dependencies.tavily_client.search.side_effect = Exception("API error")
        
        # Should return empty results instead of raising
        results = await agent._search_web(shared_dependencies, "test topic")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_web_api_error(self, mock_model, shared_dependencies):
        """Test _search_web with API-specific errors."""
        agent = ResearchAgent(mock_model)
        
        # Test rate limit error
        shared_dependencies.tavily_client.search.side_effect = Exception("rate limit exceeded")
        
        with pytest.raises(ModelRetry, match="rate limit"):
            await agent._search_web(shared_dependencies, "test topic")
        
        # Test timeout error
        shared_dependencies.tavily_client.search.side_effect = Exception("timeout occurred")
        
        with pytest.raises(ModelRetry, match="timeout"):
            await agent._search_web(shared_dependencies, "test topic")


class TestWritingAgentErrorHandling:
    """Test error handling in WritingAgent."""
    
    @pytest.mark.asyncio
    async def test_create_blog_draft_empty_topic(self, mock_model, sample_research_output, shared_dependencies):
        """Test create_blog_draft with empty topic."""
        agent = WritingAgent(mock_model)
        
        with pytest.raises(ValidationError, match="Topic cannot be empty"):
            await agent.create_blog_draft("", sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_create_blog_draft_api_error(self, mock_model, sample_research_output, shared_dependencies):
        """Test create_blog_draft with API error."""
        agent = WritingAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.create_blog_draft("test topic", sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_create_blog_draft_timeout_error(self, mock_model, sample_research_output, shared_dependencies):
        """Test create_blog_draft with timeout error."""
        agent = WritingAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("timeout occurred"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.create_blog_draft("test topic", sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_create_blog_draft_validation_error(self, mock_model, sample_research_output, shared_dependencies):
        """Test create_blog_draft with invalid output."""
        agent = WritingAgent(mock_model)
        
        # Mock agent to return incomplete draft
        incomplete_draft = BlogDraft(
            title="",  # Missing title
            introduction="",  # Missing introduction
            body_sections=[],  # Missing body sections
            conclusion="Test conclusion",
            word_count=10
        )
        
        mock_result = MagicMock()
        mock_result.output = incomplete_draft
        agent.agent.run = AsyncMock(return_value=mock_result)
        
        with pytest.raises(ValidationError, match="missing required sections"):
            await agent.create_blog_draft("test topic", sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_revise_blog_draft_validation_errors(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test revise_blog_draft with validation errors."""
        agent = WritingAgent(mock_model)
        
        # Test with None draft
        with pytest.raises(ValidationError, match="Original draft cannot be None"):
            await agent.revise_blog_draft(None, "feedback", sample_research_output, shared_dependencies)
        
        # Test with empty feedback
        with pytest.raises(ValidationError, match="Feedback cannot be empty"):
            await agent.revise_blog_draft(sample_blog_draft, "", sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_revise_blog_draft_general_error(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test revise_blog_draft with general error."""
        agent = WritingAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=ValueError("general error"))
        
        with pytest.raises(WritingError, match="Failed to revise blog draft"):
            await agent.revise_blog_draft(sample_blog_draft, "feedback", sample_research_output, shared_dependencies)


class TestCritiqueAgentErrorHandling:
    """Test error handling in CritiqueAgent."""
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_validation_errors(self, mock_model, sample_research_output, shared_dependencies):
        """Test critique_blog_draft with validation errors."""
        agent = CritiqueAgent(mock_model)
        
        # Test with None draft
        with pytest.raises(ValidationError, match="Blog draft cannot be None"):
            await agent.critique_blog_draft(None, sample_research_output, shared_dependencies)
        
        # Test with incomplete draft
        incomplete_draft = BlogDraft(
            title="",  # Missing title
            introduction="Test intro",
            body_sections=[],  # Missing body sections
            conclusion="Test conclusion",
            word_count=50
        )
        
        with pytest.raises(ValidationError, match="missing required sections"):
            await agent.critique_blog_draft(incomplete_draft, sample_research_output, shared_dependencies)
        
        # Test with None research data
        complete_draft = BlogDraft(
            title="Test Title",
            introduction="Test intro",
            body_sections=["Test body"],
            conclusion="Test conclusion",
            word_count=50
        )
        
        with pytest.raises(ValidationError, match="Research data cannot be None"):
            await agent.critique_blog_draft(complete_draft, None, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_api_error(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test critique_blog_draft with API error."""
        agent = CritiqueAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.critique_blog_draft(sample_blog_draft, sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_timeout_error(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test critique_blog_draft with timeout error."""
        agent = CritiqueAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=Exception("timeout occurred"))
        
        with pytest.raises(ModelRetry, match="API issues"):
            await agent.critique_blog_draft(sample_blog_draft, sample_research_output, shared_dependencies)
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_general_error(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test critique_blog_draft with general error."""
        agent = CritiqueAgent(mock_model)
        agent.agent.run = AsyncMock(side_effect=ValueError("general error"))
        
        with pytest.raises(CritiqueError, match="Failed to critique blog draft"):
            await agent.critique_blog_draft(sample_blog_draft, sample_research_output, shared_dependencies)


class TestOrchestratorAgentErrorHandling:
    """Test error handling in OrchestratorAgent."""
    
    @pytest.mark.asyncio
    async def test_generate_blog_post_validation_errors(self, mock_model):
        """Test generate_blog_post with validation errors."""
        orchestrator = OrchestratorAgent(mock_model)
        
        mock_research_agent = MagicMock()
        mock_writing_agent = MagicMock()
        mock_critique_agent = MagicMock()
        mock_deps = MagicMock()
        
        # Test with empty topic
        with pytest.raises(ValidationError, match="Topic cannot be empty"):
            await orchestrator.generate_blog_post(
                "", mock_research_agent, mock_writing_agent, mock_critique_agent, mock_deps
            )
        
        # Test with None agents
        with pytest.raises(ValidationError, match="All agents must be provided"):
            await orchestrator.generate_blog_post(
                "test topic", None, mock_writing_agent, mock_critique_agent, mock_deps
            )
    
    @pytest.mark.asyncio
    async def test_generate_blog_post_api_error(self, mock_model):
        """Test generate_blog_post with API error."""
        orchestrator = OrchestratorAgent(mock_model)
        orchestrator.agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
        
        mock_research_agent = MagicMock()
        mock_writing_agent = MagicMock()
        mock_critique_agent = MagicMock()
        mock_deps = MagicMock()
        
        with pytest.raises(ModelRetry, match="API issues"):
            await orchestrator.generate_blog_post(
                "test topic", mock_research_agent, mock_writing_agent, mock_critique_agent, mock_deps
            )
    
    @pytest.mark.asyncio
    async def test_generate_blog_post_timeout_error(self, mock_model):
        """Test generate_blog_post with timeout error."""
        orchestrator = OrchestratorAgent(mock_model)
        orchestrator.agent.run = AsyncMock(side_effect=Exception("timeout occurred"))
        
        mock_research_agent = MagicMock()
        mock_writing_agent = MagicMock()
        mock_critique_agent = MagicMock()
        mock_deps = MagicMock()
        
        with pytest.raises(ModelRetry, match="API issues"):
            await orchestrator.generate_blog_post(
                "test topic", mock_research_agent, mock_writing_agent, mock_critique_agent, mock_deps
            )
    
    @pytest.mark.asyncio
    async def test_generate_blog_post_agent_error_propagation(self, mock_model):
        """Test that agent-specific errors are properly wrapped."""
        orchestrator = OrchestratorAgent(mock_model)
        
        # Mock orchestrator to raise a research error
        research_error = ResearchError("Research failed", topic="test topic")
        orchestrator.agent.run = AsyncMock(side_effect=research_error)
        
        mock_research_agent = MagicMock()
        mock_writing_agent = MagicMock()
        mock_critique_agent = MagicMock()
        mock_deps = MagicMock()
        
        with pytest.raises(OrchestrationError, match="research phase"):
            await orchestrator.generate_blog_post(
                "test topic", mock_research_agent, mock_writing_agent, mock_critique_agent, mock_deps
            )
    
    @pytest.mark.asyncio
    async def test_delegate_research_graceful_degradation(self, mock_model, shared_dependencies):
        """Test delegate_research with graceful degradation."""
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create mock context
        mock_context = MagicMock()
        mock_context.deps.usage_tracking = {
            'research_calls': 0,
            'api_calls': 0,
            'total_tokens': 0
        }
        mock_context.deps.shared_deps = shared_dependencies
        
        # Mock research agent to raise a general error
        mock_research_agent = MagicMock()
        mock_research_agent.research_topic = AsyncMock(side_effect=ValueError("general error"))
        mock_context.deps.research_agent = mock_research_agent
        
        # Should return minimal research result instead of raising
        result = await orchestrator.delegate_research(mock_context, "test topic")
        
        assert result.topic == "test topic"
        assert result.findings == []
        assert "technical issues" in result.summary
        assert result.confidence_level == 0.1
    
    @pytest.mark.asyncio
    async def test_delegate_writing_graceful_degradation(self, mock_model, sample_research_output, shared_dependencies):
        """Test delegate_writing with graceful degradation."""
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create mock context
        mock_context = MagicMock()
        mock_context.deps.usage_tracking = {
            'writing_calls': 0,
            'api_calls': 0,
            'total_tokens': 0
        }
        mock_context.deps.shared_deps = shared_dependencies
        
        # Mock writing agent to raise a general error
        mock_writing_agent = MagicMock()
        mock_writing_agent.create_blog_draft = AsyncMock(side_effect=ValueError("general error"))
        mock_context.deps.writing_agent = mock_writing_agent
        
        # Should return minimal draft instead of raising
        result = await orchestrator.delegate_writing(mock_context, sample_research_output)
        
        assert sample_research_output.topic in result.title
        assert result.introduction
        assert result.body_sections
        assert result.conclusion
        assert result.word_count == 50
    
    @pytest.mark.asyncio
    async def test_delegate_critique_graceful_degradation(self, mock_model, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test delegate_critique with graceful degradation."""
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create mock context
        mock_context = MagicMock()
        mock_context.deps.usage_tracking = {
            'critique_calls': 0,
            'api_calls': 0,
            'total_tokens': 0
        }
        mock_context.deps.shared_deps = shared_dependencies
        
        # Mock critique agent to raise a general error
        mock_critique_agent = MagicMock()
        mock_critique_agent.critique_blog_draft = AsyncMock(side_effect=ValueError("general error"))
        mock_context.deps.critique_agent = mock_critique_agent
        
        # Should return minimal critique instead of raising
        result = await orchestrator.delegate_critique(mock_context, sample_blog_draft, sample_research_output)
        
        assert result.overall_quality == 6.0
        assert result.feedback_items == []
        assert result.approval_status == "approved"
        assert "technical issues" in result.summary_feedback


class TestErrorChainPropagation:
    """Test error propagation through the agent chain."""
    
    @pytest.mark.asyncio
    async def test_research_error_propagation(self, mock_model):
        """Test that research errors propagate correctly through the chain."""
        # This would test the full chain from research -> writing -> critique -> orchestrator
        # In a real scenario, we'd want to ensure that errors maintain context
        pass
    
    @pytest.mark.asyncio
    async def test_intermediate_result_preservation(self, mock_model):
        """Test that intermediate results are preserved during failures."""
        # This would test that if critique fails, we still have the research and draft
        pass
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion_handling(self, mock_model):
        """Test behavior when all retries are exhausted."""
        # This would test the final error handling when retries don't help
        pass


if __name__ == "__main__":
    pytest.main([__file__])