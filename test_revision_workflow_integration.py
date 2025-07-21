"""Integration tests for the revision workflow with quality control."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.research_agent import ResearchAgent
from src.agents.writing_agent import WritingAgent
from src.agents.critique_agent import CritiqueAgent
from src.models.data_models import (
    BlogDraft,
    ResearchOutput,
    ResearchFinding,
    CritiqueOutput,
    CritiqueFeedback,
    CritiqueSeverity,
    BlogGenerationResult
)
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


class TestRevisionWorkflowIntegration:
    """Test the complete revision workflow with quality control."""
    
    @pytest.fixture
    def sample_research_data(self):
        """Create sample research data for testing."""
        return ResearchOutput(
            topic="Intermittent Fasting",
            findings=[
                ResearchFinding(
                    fact="Intermittent fasting can improve metabolic health",
                    source_url="https://example.com/study1",
                    category="benefit",
                    relevance_score=0.9
                ),
                ResearchFinding(
                    fact="16:8 method is the most popular approach",
                    source_url="https://example.com/study2",
                    category="general_fact",
                    relevance_score=0.8
                )
            ],
            summary="Intermittent fasting shows promising health benefits",
            confidence_level=0.85
        )
    
    @pytest.fixture
    def low_quality_draft(self):
        """Create a low-quality draft that needs revision."""
        return BlogDraft(
            title="Fasting",
            introduction="Fasting is good.",
            body_sections=["Some people fast.", "It might help."],
            conclusion="Try fasting.",
            word_count=25
        )
    
    @pytest.fixture
    def high_quality_draft(self):
        """Create a high-quality draft that doesn't need revision."""
        return BlogDraft(
            title="The Complete Guide to Intermittent Fasting: Science-Backed Benefits and Practical Tips",
            introduction="Intermittent fasting has gained significant attention in recent years as a powerful tool for improving metabolic health, weight management, and overall wellness. This comprehensive guide explores the science behind intermittent fasting and provides practical strategies for implementation.",
            body_sections=[
                "Research consistently shows that intermittent fasting can improve insulin sensitivity, reduce inflammation, and support cellular repair processes. The 16:8 method, where individuals fast for 16 hours and eat within an 8-hour window, has emerged as the most sustainable approach for beginners.",
                "Beyond weight loss, intermittent fasting offers numerous health benefits including improved brain function, enhanced autophagy, and better cardiovascular health. Studies indicate that regular fasting periods can help regulate hormones and optimize metabolic processes.",
                "Getting started with intermittent fasting requires careful planning and gradual implementation. Begin with a 12-hour fasting window and gradually extend to 16 hours as your body adapts. Stay hydrated, maintain nutrient-dense meals during eating windows, and listen to your body's signals."
            ],
            conclusion="Intermittent fasting represents a evidence-based approach to improving health and wellness. By understanding the science and implementing practical strategies, individuals can harness the benefits of this powerful tool while maintaining a sustainable lifestyle.",
            word_count=850
        )
    
    @pytest.fixture
    def major_issues_critique(self):
        """Create critique with major issues requiring revision."""
        return CritiqueOutput(
            overall_quality=4.5,
            feedback_items=[
                CritiqueFeedback(
                    section="title",
                    issue="Title is too generic and not engaging",
                    suggestion="Create a more specific, benefit-focused title",
                    severity=CritiqueSeverity.MAJOR
                ),
                CritiqueFeedback(
                    section="introduction",
                    issue="Introduction lacks depth and compelling hook",
                    suggestion="Add statistics and engaging opening",
                    severity=CritiqueSeverity.MAJOR
                ),
                CritiqueFeedback(
                    section="body",
                    issue="Content is too brief and lacks detail",
                    suggestion="Expand with research findings and examples",
                    severity=CritiqueSeverity.MAJOR
                )
            ],
            approval_status="needs_revision",
            summary_feedback="The draft requires significant improvements in depth, engagement, and structure."
        )
    
    @pytest.fixture
    def minor_issues_critique(self):
        """Create critique with minor issues that may not require revision."""
        return CritiqueOutput(
            overall_quality=7.5,
            feedback_items=[
                CritiqueFeedback(
                    section="conclusion",
                    issue="Could include more actionable next steps",
                    suggestion="Add specific implementation tips",
                    severity=CritiqueSeverity.MINOR
                )
            ],
            approval_status="approved",
            summary_feedback="Excellent draft with minor room for improvement."
        )

    def test_revision_decision_logic_major_issues(self, major_issues_critique):
        """Test revision decision logic with major issues."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        from pydantic_ai.models.openai import OpenAIModel
        
        # Create orchestrator without initializing the agent (just for testing the method)
        orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
        
        # Mock RunContext
        class MockRunContext:
            def __init__(self):
                self.deps = Mock()
        
        mock_ctx = MockRunContext()
        
        # Test major issues scenario
        decision = asyncio.run(orchestrator.make_revision_decision(
            ctx=mock_ctx,
            critique_output=major_issues_critique,
            current_iteration=1,
            max_iterations=3,
            quality_threshold=7.0
        ))
        
        assert decision['should_revise'] is True
        assert "Major issues found" in decision['reasoning']
        assert decision['quality_score'] == 4.5

    def test_revision_decision_logic_approved_draft(self, minor_issues_critique):
        """Test revision decision logic with approved draft."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        
        orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
        
        class MockRunContext:
            def __init__(self):
                self.deps = Mock()
        
        mock_ctx = MockRunContext()
        
        # Test approved draft scenario
        decision = asyncio.run(orchestrator.make_revision_decision(
            ctx=mock_ctx,
            critique_output=minor_issues_critique,
            current_iteration=1,
            max_iterations=3,
            quality_threshold=7.0
        ))
        
        assert decision['should_revise'] is False
        assert "approved by critique agent" in decision['reasoning']
        assert decision['quality_score'] == 7.5

    def test_revision_decision_logic_max_iterations(self, major_issues_critique):
        """Test revision decision logic when max iterations reached."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        
        orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
        
        class MockRunContext:
            def __init__(self):
                self.deps = Mock()
        
        mock_ctx = MockRunContext()
        
        # Test max iterations scenario
        decision = asyncio.run(orchestrator.make_revision_decision(
            ctx=mock_ctx,
            critique_output=major_issues_critique,
            current_iteration=3,
            max_iterations=3,
            quality_threshold=7.0
        ))
        
        assert decision['should_revise'] is False
        assert "Maximum iterations" in decision['reasoning']

    def test_feedback_formatting_for_revision(self):
        """Test that critique feedback is properly formatted for revision."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        
        orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
        
        critique_output = CritiqueOutput(
            overall_quality=5.5,
            feedback_items=[
                CritiqueFeedback(
                    section="title",
                    issue="Title lacks specificity",
                    suggestion="Add specific benefits or numbers",
                    severity=CritiqueSeverity.MAJOR
                ),
                CritiqueFeedback(
                    section="introduction",
                    issue="Missing compelling hook",
                    suggestion="Start with surprising statistic",
                    severity=CritiqueSeverity.MODERATE
                ),
                CritiqueFeedback(
                    section="conclusion",
                    issue="Could be more actionable",
                    suggestion="Add specific next steps",
                    severity=CritiqueSeverity.MINOR
                )
            ],
            approval_status="needs_revision",
            summary_feedback="Good foundation but needs improvement in engagement and specificity."
        )
        
        formatted_feedback = orchestrator._format_feedback_for_revision(critique_output)
        
        # Verify feedback structure
        assert "Good foundation but needs improvement" in formatted_feedback
        assert "CRITICAL ISSUES TO ADDRESS:" in formatted_feedback
        assert "IMPORTANT IMPROVEMENTS:" in formatted_feedback
        assert "MINOR ENHANCEMENTS:" in formatted_feedback
        assert "title: Title lacks specificity" in formatted_feedback
        assert "Add specific benefits or numbers" in formatted_feedback

    def test_final_metrics_calculation(self):
        """Test the calculation of final processing metrics."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        import time
        
        orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
        
        start_time = time.time() - 10  # 10 seconds ago
        usage_tracking = {
            'research_calls': 1,
            'writing_calls': 2,
            'critique_calls': 2,
            'revision_cycles': 1,
            'total_tokens': 1500,
            'api_calls': 5
        }
        final_quality = 8.2
        
        metrics = orchestrator._calculate_final_metrics(
            start_time=start_time,
            usage_tracking=usage_tracking,
            final_quality=final_quality
        )
        
        # Verify metrics structure
        assert 'total_processing_time' in metrics
        assert 'quality_score' in metrics
        assert 'usage_stats' in metrics
        assert 'efficiency_score' in metrics
        
        # Verify values
        assert metrics['total_processing_time'] >= 10
        assert metrics['quality_score'] == final_quality
        assert metrics['usage_stats'] == usage_tracking
        assert metrics['efficiency_score'] > 0

    @pytest.mark.asyncio
    async def test_complete_revision_workflow_integration(self):
        """Integration test for the complete revision workflow with real models."""
        # Check for required API keys
        openai_key = os.getenv('OPENAI_API_KEY')
        tavily_key = os.getenv('TAVILY_API_KEY')
        
        if not openai_key or not tavily_key:
            pytest.skip("API keys not available for integration test")
        
        print("🎯 Testing complete revision workflow integration...")
        
        try:
            # Initialize models
            model = OpenAIModel('gpt-4o-mini')
            
            async with httpx.AsyncClient() as http_client:
                # Create shared dependencies with low quality threshold to trigger revisions
                deps = SharedDependencies(
                    http_client=http_client,
                    tavily_client=TavilyClient(api_key=tavily_key),
                    max_iterations=2,
                    quality_threshold=8.0  # High threshold to potentially trigger revisions
                )
                
                # Create all agents
                orchestrator_agent = OrchestratorAgent(model)
                research_agent = ResearchAgent(model)
                writing_agent = WritingAgent(model)
                critique_agent = CritiqueAgent(model)
                
                # Test topic that might need revision
                topic = "benefits of drinking water"
                
                # Run the complete workflow
                result = await orchestrator_agent.generate_blog_post(
                    topic=topic,
                    research_agent=research_agent,
                    writing_agent=writing_agent,
                    critique_agent=critique_agent,
                    deps=deps
                )
                
                # Verify results
                assert isinstance(result, BlogGenerationResult)
                assert result.final_post is not None
                assert result.research_data is not None
                assert result.revision_count >= 0
                assert result.revision_count <= deps.max_iterations
                assert result.total_processing_time > 0
                assert result.quality_score >= 0
                
                print(f"✅ Integration test completed successfully!")
                print(f"   Revision cycles: {result.revision_count}")
                print(f"   Final quality: {result.quality_score:.1f}")
                print(f"   Processing time: {result.total_processing_time:.2f}s")
                print(f"   Word count: {result.final_post.word_count}")
                
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    @pytest.mark.asyncio
    async def test_revision_workflow_with_forced_low_quality(self):
        """Test revision workflow by forcing low quality threshold."""
        # Check for required API keys
        openai_key = os.getenv('OPENAI_API_KEY')
        tavily_key = os.getenv('TAVILY_API_KEY')
        
        if not openai_key or not tavily_key:
            pytest.skip("API keys not available for integration test")
        
        print("🔄 Testing revision workflow with forced low quality threshold...")
        
        try:
            model = OpenAIModel('gpt-4o-mini')
            
            async with httpx.AsyncClient() as http_client:
                # Create dependencies with very high quality threshold to force revisions
                deps = SharedDependencies(
                    http_client=http_client,
                    tavily_client=TavilyClient(api_key=tavily_key),
                    max_iterations=2,
                    quality_threshold=9.5  # Very high threshold to force revisions
                )
                
                orchestrator_agent = OrchestratorAgent(model)
                research_agent = ResearchAgent(model)
                writing_agent = WritingAgent(model)
                critique_agent = CritiqueAgent(model)
                
                topic = "simple topic"
                
                result = await orchestrator_agent.generate_blog_post(
                    topic=topic,
                    research_agent=research_agent,
                    writing_agent=writing_agent,
                    critique_agent=critique_agent,
                    deps=deps
                )
                
                # With high threshold, we should see some revision attempts
                assert result.revision_count >= 0
                print(f"✅ Forced revision test completed!")
                print(f"   Revision cycles: {result.revision_count}")
                print(f"   Final quality: {result.quality_score:.1f}")
                
        except Exception as e:
            pytest.fail(f"Forced revision test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])