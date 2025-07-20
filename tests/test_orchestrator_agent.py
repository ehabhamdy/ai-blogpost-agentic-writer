"""Tests for the Orchestrator Agent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.agents.orchestrator_agent import OrchestratorAgent, OrchestrationContext
from src.models.data_models import (
    BlogGenerationResult,
    BlogDraft,
    ResearchOutput,
    ResearchFinding,
    CritiqueOutput,
    CritiqueFeedback,
    CritiqueSeverity
)


class TestOrchestratorAgent:
    """Test cases for the Orchestrator Agent."""
    
    @pytest.fixture
    def orchestrator_agent(self):
        """Create an Orchestrator Agent instance for testing."""
        from pydantic_ai.models.test import TestModel
        model = TestModel()
        return OrchestratorAgent(model)
    

    
    @pytest.fixture
    def sample_research_output(self):
        """Create sample research output for testing."""
        return ResearchOutput(
            topic="Benefits of Intermittent Fasting",
            findings=[
                ResearchFinding(
                    fact="Intermittent fasting can help with weight loss",
                    source_url="https://example.com/study1",
                    relevance_score=0.9,
                    category="benefit"
                ),
                ResearchFinding(
                    fact="Studies show 16:8 method is most popular",
                    source_url="https://example.com/study2",
                    relevance_score=0.8,
                    category="statistic"
                )
            ],
            summary="Research shows intermittent fasting has multiple health benefits",
            confidence_level=0.8
        )
    
    @pytest.fixture
    def sample_blog_draft(self):
        """Create sample blog draft for testing."""
        return BlogDraft(
            title="The Complete Guide to Intermittent Fasting",
            introduction="Intermittent fasting has gained popularity as a health practice.",
            body_sections=[
                "What is intermittent fasting and how does it work?",
                "The science-backed benefits of intermittent fasting",
                "Different methods and how to choose the right one"
            ],
            conclusion="Intermittent fasting can be a valuable tool for health improvement.",
            word_count=850
        )
    
    @pytest.fixture
    def sample_critique_output_approved(self):
        """Create sample critique output that approves the draft."""
        return CritiqueOutput(
            overall_quality=8.5,
            feedback_items=[
                CritiqueFeedback(
                    section="introduction",
                    issue="Could be more engaging",
                    suggestion="Add a compelling hook",
                    severity=CritiqueSeverity.MINOR
                )
            ],
            approval_status="approved",
            summary_feedback="Well-written article with minor improvements possible"
        )
    
    @pytest.fixture
    def sample_critique_output_needs_revision(self):
        """Create sample critique output that needs revision."""
        return CritiqueOutput(
            overall_quality=5.5,
            feedback_items=[
                CritiqueFeedback(
                    section="body",
                    issue="Lacks supporting evidence",
                    suggestion="Add more research citations",
                    severity=CritiqueSeverity.MAJOR
                ),
                CritiqueFeedback(
                    section="conclusion",
                    issue="Too abrupt",
                    suggestion="Provide better summary",
                    severity=CritiqueSeverity.MODERATE
                )
            ],
            approval_status="needs_revision",
            summary_feedback="Article needs significant improvements in evidence and structure"
        )
    
    def test_orchestrator_agent_initialization(self, orchestrator_agent):
        """Test that Orchestrator Agent initializes correctly."""
        assert orchestrator_agent.agent is not None
        # Check that the agent has the expected tools
        assert hasattr(orchestrator_agent, 'delegate_research')
        assert hasattr(orchestrator_agent, 'delegate_writing')
        assert hasattr(orchestrator_agent, 'delegate_critique')
        assert hasattr(orchestrator_agent, 'make_revision_decision')
    

    
    @pytest.mark.asyncio
    async def test_make_revision_decision_approved(
        self, 
        orchestrator_agent, 
        sample_critique_output_approved
    ):
        """Test revision decision when draft is approved."""
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=Mock(),
            writing_agent=Mock(),
            critique_agent=Mock(),
            start_time=0.0,
            usage_tracking={}
        )
        
        # Create a mock RunContext
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Execute
        decision = await orchestrator_agent.make_revision_decision(
            mock_ctx,
            sample_critique_output_approved,
            current_iteration=1,
            max_iterations=3,
            quality_threshold=7.0
        )
        
        # Verify
        assert decision['should_revise'] is False
        assert decision['approval_status'] == "approved"
        assert "approved" in decision['reasoning'].lower()
    
    @pytest.mark.asyncio
    async def test_make_revision_decision_needs_revision(
        self, 
        orchestrator_agent, 
        sample_critique_output_needs_revision
    ):
        """Test revision decision when draft needs revision."""
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=Mock(),
            writing_agent=Mock(),
            critique_agent=Mock(),
            start_time=0.0,
            usage_tracking={}
        )
        
        # Create a mock RunContext
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Execute
        decision = await orchestrator_agent.make_revision_decision(
            mock_ctx,
            sample_critique_output_needs_revision,
            current_iteration=1,
            max_iterations=3,
            quality_threshold=7.0
        )
        
        # Verify
        assert decision['should_revise'] is True
        assert decision['quality_score'] == 5.5
        assert "major issues" in decision['reasoning'].lower()
    
    @pytest.mark.asyncio
    async def test_make_revision_decision_max_iterations(
        self, 
        orchestrator_agent, 
        sample_critique_output_needs_revision
    ):
        """Test revision decision when max iterations reached."""
        context = OrchestrationContext(
            topic="Test Topic",
            research_agent=Mock(),
            writing_agent=Mock(),
            critique_agent=Mock(),
            start_time=0.0,
            usage_tracking={}
        )
        
        # Create a mock RunContext
        mock_ctx = Mock()
        mock_ctx.deps = context
        
        # Execute
        decision = await orchestrator_agent.make_revision_decision(
            mock_ctx,
            sample_critique_output_needs_revision,
            current_iteration=3,
            max_iterations=3,
            quality_threshold=7.0
        )
        
        # Verify
        assert decision['should_revise'] is False
        assert "maximum iterations" in decision['reasoning'].lower()
    
    def test_format_feedback_for_revision(
        self, 
        orchestrator_agent, 
        sample_critique_output_needs_revision
    ):
        """Test formatting of critique feedback for revision."""
        feedback = orchestrator_agent._format_feedback_for_revision(
            sample_critique_output_needs_revision
        )
        
        assert "CRITICAL ISSUES" in feedback
        assert "IMPORTANT IMPROVEMENTS" in feedback
        assert "Lacks supporting evidence" in feedback
        assert "Too abrupt" in feedback
    
    def test_calculate_final_metrics(self, orchestrator_agent):
        """Test calculation of final processing metrics."""
        usage_tracking = {
            'research_calls': 1,
            'writing_calls': 2,
            'critique_calls': 2,
            'total_tokens': 1000,
            'api_calls': 5
        }
        
        metrics = orchestrator_agent._calculate_final_metrics(
            start_time=0.0,
            usage_tracking=usage_tracking,
            final_quality=8.5
        )
        
        assert 'total_processing_time' in metrics
        assert metrics['quality_score'] == 8.5
        assert metrics['usage_stats'] == usage_tracking
        assert 'efficiency_score' in metrics
    
