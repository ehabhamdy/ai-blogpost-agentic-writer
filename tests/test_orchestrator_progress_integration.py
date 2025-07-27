"""Integration tests for progress tracking in the orchestrator agent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from src.models.data_models import (
    ProcessingStage,
    ProgressStatus,
    ResearchOutput,
    ResearchFinding,
    BlogDraft,
    CritiqueOutput,
    CritiqueFeedback,
    CritiqueSeverity
)
from src.utils.dependencies import SharedDependencies
from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.progress_tracker import ProgressTracker


class TestOrchestratorProgressIntegration:
    """Integration tests for progress tracking in the orchestrator."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return 'test'  # Use pydantic_ai's test model
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        research_agent = Mock()
        writing_agent = Mock()
        critique_agent = Mock()
        
        # Mock research agent response
        research_agent.research_topic = AsyncMock(return_value=ResearchOutput(
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
        ))
        
        # Mock writing agent response
        writing_agent.create_blog_draft = AsyncMock(return_value=BlogDraft(
            title="Test Blog Post",
            introduction="Test introduction",
            body_sections=["Test body section"],
            conclusion="Test conclusion",
            word_count=100
        ))
        
        writing_agent.revise_blog_draft = AsyncMock(return_value=BlogDraft(
            title="Revised Test Blog Post",
            introduction="Revised test introduction",
            body_sections=["Revised test body section"],
            conclusion="Revised test conclusion",
            word_count=120
        ))
        
        # Mock critique agent response
        critique_agent.critique_blog_draft = AsyncMock(return_value=CritiqueOutput(
            overall_quality=8.5,
            feedback_items=[],
            approval_status="approved",
            summary_feedback="Good quality draft"
        ))
        
        return research_agent, writing_agent, critique_agent
    
    @pytest.fixture
    def shared_deps(self):
        """Create shared dependencies with progress callback."""
        progress_updates = []
        
        def progress_callback(status: ProgressStatus):
            progress_updates.append(status)
        
        deps = SharedDependencies(
            http_client=Mock(),
            tavily_client=Mock(),
            max_iterations=3,
            quality_threshold=7.0,
            progress_callback=progress_callback
        )
        
        # Store progress updates for testing
        deps._progress_updates = progress_updates
        
        return deps
    
    @pytest.mark.asyncio
    async def test_orchestrator_progress_tracking_success(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test that progress tracking works correctly during successful workflow."""
        research_agent, writing_agent, critique_agent = mock_agents
        
        # Create orchestrator with mocked agent.run
        orchestrator = OrchestratorAgent(mock_model)
        
        # Mock the agent.run method to simulate successful workflow
        mock_result = Mock()
        mock_result.output = Mock()
        mock_result.output.quality_score = 8.5
        mock_result.output.revision_count = 0
        
        with patch.object(orchestrator.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            
            # Run the workflow
            result = await orchestrator.generate_blog_post(
                topic="Test Topic",
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                deps=shared_deps
            )
            
            # Verify progress updates were received
            progress_updates = shared_deps._progress_updates
            assert len(progress_updates) > 0
            
            # Check that we got a completion update
            final_update = progress_updates[-1]
            assert final_update.stage == ProcessingStage.COMPLETED
            
            # Verify workflow progress is included in result
            assert hasattr(result, 'workflow_progress')
            assert result.workflow_progress.topic == "Test Topic"
            assert result.workflow_progress.current_status.stage == ProcessingStage.COMPLETED
            assert result.workflow_progress.final_quality_score == 8.5
    
    @pytest.mark.asyncio
    async def test_orchestrator_progress_tracking_with_error(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test that progress tracking works correctly when workflow fails."""
        research_agent, writing_agent, critique_agent = mock_agents
        
        # Make research agent fail
        research_agent.research_topic.side_effect = Exception("Network error")
        
        orchestrator = OrchestratorAgent(mock_model)
        
        # Mock the agent.run method to raise an exception
        with patch.object(orchestrator.agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Workflow failed")
            
            # Run the workflow and expect it to fail
            with pytest.raises(Exception):
                await orchestrator.generate_blog_post(
                    topic="Test Topic",
                    research_agent=research_agent,
                    writing_agent=writing_agent,
                    critique_agent=critique_agent,
                    deps=shared_deps
                )
            
            # Verify progress updates were received
            progress_updates = shared_deps._progress_updates
            assert len(progress_updates) > 0
            
            # Check that we got a failure update
            final_update = progress_updates[-1]
            assert final_update.stage == ProcessingStage.FAILED
    
    @pytest.mark.asyncio
    async def test_delegate_research_progress_tracking(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test progress tracking in delegate_research method."""
        research_agent, writing_agent, critique_agent = mock_agents
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create orchestration context
        from src.agents.orchestrator_agent import OrchestrationContext
        from src.utils.progress_tracker import ProgressTracker
        
        progress_tracker = ProgressTracker("Test Topic", shared_deps.progress_callback)
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=research_agent,
            writing_agent=writing_agent,
            critique_agent=critique_agent,
            start_time=0.0,
            usage_tracking={
                'research_calls': 0,
                'writing_calls': 0,
                'critique_calls': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'revision_cycles': 0
            },
            shared_deps=shared_deps,
            progress_tracker=progress_tracker
        )
        
        # Create mock run context
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Call delegate_research
        result = await orchestrator.delegate_research(mock_ctx, "Test Topic")
        
        # Verify research was called
        research_agent.research_topic.assert_called_once()
        
        # Verify result
        assert result.topic == "Test Topic"
        assert len(result.findings) == 1
        
        # Verify progress tracking
        progress_updates = shared_deps._progress_updates
        research_updates = [u for u in progress_updates if u.stage == ProcessingStage.RESEARCHING]
        assert len(research_updates) > 0
        assert research_updates[0].agent_name == "ResearchAgent"
    
    @pytest.mark.asyncio
    async def test_delegate_writing_progress_tracking(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test progress tracking in delegate_writing method."""
        research_agent, writing_agent, critique_agent = mock_agents
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create orchestration context
        from src.agents.orchestrator_agent import OrchestrationContext
        from src.utils.progress_tracker import ProgressTracker
        
        progress_tracker = ProgressTracker("Test Topic", shared_deps.progress_callback)
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=research_agent,
            writing_agent=writing_agent,
            critique_agent=critique_agent,
            start_time=0.0,
            usage_tracking={
                'research_calls': 0,
                'writing_calls': 0,
                'critique_calls': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'revision_cycles': 0
            },
            shared_deps=shared_deps,
            progress_tracker=progress_tracker
        )
        
        # Create mock run context
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Create research data
        research_data = ResearchOutput(
            topic="Test Topic",
            findings=[],
            summary="Test summary",
            confidence_level=0.8
        )
        
        # Call delegate_writing (initial draft)
        result = await orchestrator.delegate_writing(mock_ctx, research_data)
        
        # Verify writing was called
        writing_agent.create_blog_draft.assert_called_once()
        
        # Verify result
        assert result.title == "Test Blog Post"
        
        # Verify progress tracking
        progress_updates = shared_deps._progress_updates
        writing_updates = [u for u in progress_updates if u.stage == ProcessingStage.WRITING]
        assert len(writing_updates) > 0
        assert writing_updates[0].agent_name == "WritingAgent"
    
    @pytest.mark.asyncio
    async def test_delegate_critique_progress_tracking(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test progress tracking in delegate_critique method."""
        research_agent, writing_agent, critique_agent = mock_agents
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create orchestration context
        from src.agents.orchestrator_agent import OrchestrationContext
        from src.utils.progress_tracker import ProgressTracker
        
        progress_tracker = ProgressTracker("Test Topic", shared_deps.progress_callback)
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=research_agent,
            writing_agent=writing_agent,
            critique_agent=critique_agent,
            start_time=0.0,
            usage_tracking={
                'research_calls': 0,
                'writing_calls': 0,
                'critique_calls': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'revision_cycles': 0
            },
            shared_deps=shared_deps,
            progress_tracker=progress_tracker
        )
        
        # Create mock run context
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Create test data
        blog_draft = BlogDraft(
            title="Test Blog",
            introduction="Test intro",
            body_sections=["Test body"],
            conclusion="Test conclusion",
            word_count=100
        )
        
        research_data = ResearchOutput(
            topic="Test Topic",
            findings=[],
            summary="Test summary",
            confidence_level=0.8
        )
        
        # Call delegate_critique
        result = await orchestrator.delegate_critique(mock_ctx, blog_draft, research_data)
        
        # Verify critique was called
        critique_agent.critique_blog_draft.assert_called_once()
        
        # Verify result
        assert result.overall_quality == 8.5
        assert result.approval_status == "approved"
        
        # Verify progress tracking
        progress_updates = shared_deps._progress_updates
        critique_updates = [u for u in progress_updates if u.stage == ProcessingStage.CRITIQUING]
        assert len(critique_updates) > 0
        assert critique_updates[0].agent_name == "CritiqueAgent"
    
    @pytest.mark.asyncio
    async def test_revision_progress_tracking(
        self,
        mock_model,
        mock_agents,
        shared_deps
    ):
        """Test progress tracking during revision cycles."""
        research_agent, writing_agent, critique_agent = mock_agents
        orchestrator = OrchestratorAgent(mock_model)
        
        # Create orchestration context
        from src.agents.orchestrator_agent import OrchestrationContext
        from src.utils.progress_tracker import ProgressTracker
        
        progress_tracker = ProgressTracker("Test Topic", shared_deps.progress_callback)
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=research_agent,
            writing_agent=writing_agent,
            critique_agent=critique_agent,
            start_time=0.0,
            usage_tracking={
                'research_calls': 0,
                'writing_calls': 0,
                'critique_calls': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'revision_cycles': 0
            },
            shared_deps=shared_deps,
            progress_tracker=progress_tracker
        )
        
        # Create mock run context
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Create test data
        research_data = ResearchOutput(
            topic="Test Topic",
            findings=[],
            summary="Test summary",
            confidence_level=0.8
        )
        
        original_draft = BlogDraft(
            title="Original Blog",
            introduction="Original intro",
            body_sections=["Original body"],
            conclusion="Original conclusion",
            word_count=100
        )
        
        feedback = "Please improve the introduction"
        
        # Call delegate_writing with revision
        result = await orchestrator.delegate_writing(
            mock_ctx, research_data, feedback, original_draft
        )
        
        # Verify revision was called
        writing_agent.revise_blog_draft.assert_called_once()
        
        # Verify result
        assert result.title == "Revised Test Blog Post"
        
        # Verify progress tracking shows revision stage
        progress_updates = shared_deps._progress_updates
        revision_updates = [u for u in progress_updates if u.stage == ProcessingStage.REVISING]
        assert len(revision_updates) > 0
        assert revision_updates[0].agent_name == "WritingAgent"
        
        # Verify revision count was incremented
        assert context.usage_tracking['revision_cycles'] == 1


if __name__ == "__main__":
    pytest.main([__file__])