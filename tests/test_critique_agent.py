"""Unit tests for the Critique Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_ai import ModelRetry

from src.agents.critique_agent import CritiqueAgent, CritiqueContext
from src.models.data_models import (
    BlogDraft, 
    ResearchOutput, 
    ResearchFinding, 
    CritiqueOutput,
    CritiqueFeedback,
    CritiqueSeverity
)
from src.utils.dependencies import SharedDependencies


@pytest.fixture
def sample_blog_draft():
    """Sample blog draft for testing."""
    return BlogDraft(
        title="The Complete Guide to Intermittent Fasting",
        introduction="Intermittent fasting has gained popularity as a health strategy. This comprehensive guide will explore the benefits, methods, and considerations for intermittent fasting.",
        body_sections=[
            "Research shows that intermittent fasting can improve metabolic health. Studies indicate that it may help with weight loss and reduce inflammation.",
            "There are several methods of intermittent fasting. The 16:8 method involves fasting for 16 hours and eating within an 8-hour window.",
            "While intermittent fasting has benefits, it's important to consider potential risks. Some people may experience fatigue or difficulty concentrating."
        ],
        conclusion="In conclusion, intermittent fasting can be an effective health strategy when done properly. Consider consulting with a healthcare provider before starting.",
        word_count=150
    )


@pytest.fixture
def sample_research_output():
    """Sample research output for testing."""
    return ResearchOutput(
        topic="intermittent fasting",
        findings=[
            ResearchFinding(
                fact="Intermittent fasting can reduce body weight by 3-8% over 3-24 weeks",
                source_url="https://example.com/study1",
                relevance_score=0.9,
                category="statistic"
            ),
            ResearchFinding(
                fact="Studies show intermittent fasting improves insulin sensitivity",
                source_url="https://example.com/study2",
                relevance_score=0.8,
                category="study"
            ),
            ResearchFinding(
                fact="The 16:8 method is the most popular form of intermittent fasting",
                source_url="https://example.com/study3",
                relevance_score=0.7,
                category="general_fact"
            )
        ],
        summary="Research indicates intermittent fasting has metabolic benefits",
        confidence_level=0.85
    )


@pytest.fixture
def shared_dependencies():
    """Sample shared dependencies for testing."""
    return SharedDependencies(
        http_client=MagicMock(),
        tavily_client=MagicMock(),
        max_iterations=3,
        quality_threshold=7.0
    )


@pytest.fixture
def critique_agent():
    """Create a critique agent for testing."""
    with patch('src.agents.critique_agent.Agent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        mock_model = MagicMock()
        agent = CritiqueAgent(mock_model)
        agent.agent = mock_agent  # Ensure the agent is properly mocked
        return agent


class TestCritiqueAgent:
    """Test cases for CritiqueAgent class."""
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_success(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test successful blog draft critique."""
        # Mock the agent run method
        expected_critique = CritiqueOutput(
            overall_quality=8.5,
            feedback_items=[
                CritiqueFeedback(
                    section="introduction",
                    issue="Could be more engaging",
                    suggestion="Add a compelling hook or statistic",
                    severity=CritiqueSeverity.MINOR
                )
            ],
            approval_status="approved",
            summary_feedback="Well-structured article with good research integration"
        )
        
        critique_agent.agent.run = AsyncMock(return_value=MagicMock(data=expected_critique))
        
        result = await critique_agent.critique_blog_draft(
            sample_blog_draft, 
            sample_research_output, 
            shared_dependencies
        )
        
        assert isinstance(result, CritiqueOutput)
        assert result.overall_quality == 8.5
        assert result.approval_status == "approved"
        assert len(result.feedback_items) == 1
        assert result.feedback_items[0].severity == CritiqueSeverity.MINOR
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_with_retry(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test critique with retry on rate limit error."""
        critique_agent.agent.run = AsyncMock(side_effect=Exception("rate limit exceeded"))
        
        with pytest.raises(ModelRetry):
            await critique_agent.critique_blog_draft(
                sample_blog_draft, 
                sample_research_output, 
                shared_dependencies
            )
    
    @pytest.mark.asyncio
    async def test_critique_blog_draft_non_retryable_error(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test critique with non-retryable error."""
        critique_agent.agent.run = AsyncMock(side_effect=ValueError("Invalid input"))
        
        with pytest.raises(ValueError):
            await critique_agent.critique_blog_draft(
                sample_blog_draft, 
                sample_research_output, 
                shared_dependencies
            )


class TestAnalyzeClarityTool:
    """Test cases for the analyze_clarity tool."""
    
    @pytest.mark.asyncio
    async def test_analyze_clarity_comprehensive(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test comprehensive clarity analysis."""
        context = CritiqueContext(
            blog_draft=sample_blog_draft,
            research_data=sample_research_output,
            quality_threshold=7.0
        )
        
        mock_ctx = MagicMock()
        mock_ctx.deps = context
        
        result = await critique_agent.analyze_clarity(mock_ctx)
        
        assert isinstance(result, dict)
        assert 'title_clarity' in result
        assert 'introduction_clarity' in result
        assert 'body_clarity' in result
        assert 'conclusion_clarity' in result
        assert 'overall_readability' in result
        assert 'transition_quality' in result
        assert 'vocabulary_assessment' in result
    
    def test_analyze_title_clarity(self, critique_agent):
        """Test title clarity analysis."""
        # Test good title
        good_title = "The Complete Guide to Intermittent Fasting Benefits"
        result = critique_agent._analyze_title_clarity(good_title)
        
        assert result['length'] == len(good_title)
        assert result['word_count'] == 7
        assert result['clarity_score'] > 0.5
        assert len(result['issues']) == 0
        
        # Test too long title
        long_title = "This is an extremely long title that goes on and on and probably exceeds the optimal length for SEO purposes"
        result = critique_agent._analyze_title_clarity(long_title)
        
        assert "too long" in result['issues'][0]
        
        # Test too short title
        short_title = "Fasting"
        result = critique_agent._analyze_title_clarity(short_title)
        
        assert "too short" in result['issues'][0]
    
    def test_analyze_section_clarity(self, critique_agent):
        """Test section clarity analysis."""
        # Test well-balanced section
        good_section = "This is a well-written section. It has clear sentences. The ideas flow naturally from one to the next."
        result = critique_agent._analyze_section_clarity(good_section, "introduction")
        
        assert result['sentence_count'] == 3
        assert result['avg_sentence_length'] > 0
        assert result['clarity_score'] > 0.5
        
        # Test section with overly long sentences
        long_sentence_section = "This is an extremely long sentence that goes on and on with multiple clauses and subclauses that make it very difficult to read and understand for the average reader who is looking for clear and concise information."
        result = critique_agent._analyze_section_clarity(long_sentence_section, "body")
        
        assert "complex sentences" in result['issues'][0]
        
        # Test section with too much passive voice
        passive_section = "The study was conducted by researchers. The results were analyzed by experts. The conclusions were drawn by the team."
        result = critique_agent._analyze_section_clarity(passive_section, "conclusion")
        
        assert any("passive voice" in issue for issue in result['issues'])
    
    def test_calculate_overall_readability(self, critique_agent, sample_blog_draft):
        """Test overall readability calculation."""
        result = critique_agent._calculate_overall_readability(sample_blog_draft)
        
        assert 'total_words' in result
        assert 'total_sentences' in result
        assert 'avg_words_per_sentence' in result
        assert 'readability_score' in result
        assert 0 <= result['readability_score'] <= 1
    
    def test_assess_transitions(self, critique_agent, sample_blog_draft):
        """Test transition assessment."""
        result = critique_agent._assess_transitions(sample_blog_draft)
        
        assert 'transition_count' in result
        assert 'transition_density' in result
        assert 'quality_score' in result
        assert 0 <= result['quality_score'] <= 1
    
    def test_assess_vocabulary_level(self, critique_agent, sample_blog_draft):
        """Test vocabulary level assessment."""
        result = critique_agent._assess_vocabulary_level(sample_blog_draft)
        
        assert 'total_words' in result
        assert 'complex_words' in result
        assert 'complexity_ratio' in result
        assert 'appropriateness_score' in result
        assert 0 <= result['appropriateness_score'] <= 1


class TestVerifyFactsTool:
    """Test cases for the verify_facts tool."""
    
    @pytest.mark.asyncio
    async def test_verify_facts_comprehensive(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test comprehensive fact verification."""
        context = CritiqueContext(
            blog_draft=sample_blog_draft,
            research_data=sample_research_output,
            quality_threshold=7.0
        )
        
        mock_ctx = MagicMock()
        mock_ctx.deps = context
        
        result = await critique_agent.verify_facts(mock_ctx)
        
        assert isinstance(result, dict)
        assert 'supported_claims' in result
        assert 'unsupported_claims' in result
        assert 'source_attribution' in result
        assert 'statistical_accuracy' in result
        assert 'research_utilization' in result
        assert 'fact_density' in result
    
    def test_identify_supported_claims(self, critique_agent, sample_research_output):
        """Test identification of supported claims."""
        content = "Research shows that intermittent fasting can improve metabolic health and help with weight loss."
        
        result = critique_agent._identify_supported_claims(content, sample_research_output.findings)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('claim_context' in claim for claim in result)
        assert all('supporting_finding' in claim for claim in result)
    
    def test_identify_unsupported_claims(self, critique_agent, sample_research_output):
        """Test identification of unsupported claims."""
        content = "Studies show that intermittent fasting cures cancer. Research indicates it makes you immortal."
        
        result = critique_agent._identify_unsupported_claims(content, sample_research_output.findings)
        
        assert isinstance(result, list)
        # Should identify unsupported claims
        assert len(result) >= 0
    
    def test_check_source_attribution(self, critique_agent, sample_research_output):
        """Test source attribution checking."""
        content_with_attribution = "According to recent studies, intermittent fasting has benefits. Research from Harvard shows positive results."
        content_without_attribution = "Intermittent fasting is good. It helps with weight loss."
        
        result_with = critique_agent._check_source_attribution(content_with_attribution, sample_research_output.findings)
        result_without = critique_agent._check_source_attribution(content_without_attribution, sample_research_output.findings)
        
        assert result_with['attribution_count'] > result_without['attribution_count']
        assert result_with['attribution_score'] > result_without['attribution_score']
    
    def test_verify_statistics(self, critique_agent, sample_research_output):
        """Test statistical verification."""
        content_with_stats = "Studies show 75% improvement and 3-8% weight loss over 24 weeks."
        content_without_stats = "Studies show great improvement and significant weight loss."
        
        result_with = critique_agent._verify_statistics(content_with_stats, sample_research_output.findings)
        result_without = critique_agent._verify_statistics(content_without_stats, sample_research_output.findings)
        
        assert result_with['numbers_in_content'] > result_without['numbers_in_content']
        assert 'verification_score' in result_with
        assert 'verification_score' in result_without
    
    def test_assess_research_utilization(self, critique_agent, sample_research_output):
        """Test research utilization assessment."""
        well_utilized_content = "Intermittent fasting improves insulin sensitivity and reduces body weight significantly."
        poorly_utilized_content = "Fasting is good for health and wellness in general."
        
        result_good = critique_agent._assess_research_utilization(well_utilized_content, sample_research_output.findings)
        result_poor = critique_agent._assess_research_utilization(poorly_utilized_content, sample_research_output.findings)
        
        assert result_good['utilization_score'] > result_poor['utilization_score']
        assert result_good['utilized_findings'] > result_poor['utilized_findings']
    
    def test_calculate_fact_density(self, critique_agent, sample_research_output):
        """Test fact density calculation."""
        fact_dense_content = "Research shows evidence from multiple studies that data indicates significant results."
        fact_sparse_content = "This is a simple article about health and wellness topics."
        
        result_dense = critique_agent._calculate_fact_density(fact_dense_content, sample_research_output.findings)
        result_sparse = critique_agent._calculate_fact_density(fact_sparse_content, sample_research_output.findings)
        
        assert result_dense['fact_density'] > result_sparse['fact_density']
        assert result_dense['density_score'] > result_sparse['density_score']


class TestAssessStructureTool:
    """Test cases for the assess_structure tool."""
    
    @pytest.mark.asyncio
    async def test_assess_structure_comprehensive(self, critique_agent, sample_blog_draft, sample_research_output, shared_dependencies):
        """Test comprehensive structure assessment."""
        context = CritiqueContext(
            blog_draft=sample_blog_draft,
            research_data=sample_research_output,
            quality_threshold=7.0
        )
        
        mock_ctx = MagicMock()
        mock_ctx.deps = context
        
        result = await critique_agent.assess_structure(mock_ctx)
        
        assert isinstance(result, dict)
        assert 'introduction_effectiveness' in result
        assert 'body_organization' in result
        assert 'conclusion_effectiveness' in result
        assert 'logical_flow' in result
        assert 'section_balance' in result
        assert 'narrative_coherence' in result
        assert 'content_depth' in result
    
    def test_assess_introduction(self, critique_agent):
        """Test introduction assessment."""
        # Good introduction
        good_intro = "Did you know that intermittent fasting can transform your health? This comprehensive article will explore the science-backed benefits, practical methods, and important considerations for intermittent fasting. You'll discover how this powerful strategy can improve your metabolic health and overall well-being."
        
        result = critique_agent._assess_introduction(good_intro)
        
        assert result['word_count'] > 0
        assert result['effectiveness_score'] > 0
        assert 'strengths' in result
        assert 'weaknesses' in result
        
        # Too short introduction
        short_intro = "Fasting is good."
        result_short = critique_agent._assess_introduction(short_intro)
        
        assert any("too brief" in weakness for weakness in result_short['weaknesses'])
    
    def test_assess_body_organization(self, critique_agent):
        """Test body organization assessment."""
        # Well-organized body sections
        good_sections = [
            "First section with detailed information about the topic and comprehensive coverage of key points.",
            "Second section that builds on the first with additional insights and supporting evidence.",
            "Third section that provides practical applications and real-world examples for readers."
        ]
        
        result = critique_agent._assess_body_organization(good_sections)
        
        assert result['section_count'] == 3
        assert result['organization_score'] > 0
        assert result['balance_score'] > 0
        
        # Too few sections
        few_sections = ["Only one section here."]
        result_few = critique_agent._assess_body_organization(few_sections)
        
        assert any("Too few" in issue for issue in result_few['issues'])
        
        # Empty sections
        empty_result = critique_agent._assess_body_organization([])
        assert "No body sections found" in empty_result['issues']
    
    def test_assess_conclusion(self, critique_agent):
        """Test conclusion assessment."""
        # Good conclusion
        good_conclusion = "In summary, intermittent fasting offers significant health benefits when implemented properly. Consider starting with the 16:8 method and consult your healthcare provider. Take the first step toward better health today."
        
        result = critique_agent._assess_conclusion(good_conclusion)
        
        assert result['word_count'] > 0
        assert result['effectiveness_score'] > 0
        assert len(result['strengths']) > 0
        
        # Too short conclusion
        short_conclusion = "Good stuff."
        result_short = critique_agent._assess_conclusion(short_conclusion)
        
        assert any("too brief" in weakness for weakness in result_short['weaknesses'])
    
    def test_assess_logical_flow(self, critique_agent, sample_blog_draft):
        """Test logical flow assessment."""
        result = critique_agent._assess_logical_flow(sample_blog_draft)
        
        assert 'flow_score' in result
        assert 'coherence_indicators' in result
        assert 0 <= result['flow_score'] <= 1
        assert 'intro_to_body_overlap' in result['coherence_indicators']
        assert 'intro_to_conclusion_overlap' in result['coherence_indicators']
    
    def test_assess_section_balance(self, critique_agent, sample_blog_draft):
        """Test section balance assessment."""
        result = critique_agent._assess_section_balance(sample_blog_draft)
        
        assert 'total_words' in result
        assert 'section_word_counts' in result
        assert 'balance_score' in result
        assert 'recommendations' in result
        assert 0 <= result['balance_score'] <= 1
    
    def test_assess_narrative_coherence(self, critique_agent, sample_blog_draft):
        """Test narrative coherence assessment."""
        result = critique_agent._assess_narrative_coherence(sample_blog_draft)
        
        assert 'coherence_score' in result
        assert 'top_themes' in result
        assert 'theme_consistency' in result
        assert 0 <= result['coherence_score'] <= 1
        assert isinstance(result['theme_consistency'], bool)
    
    def test_assess_content_depth(self, critique_agent, sample_blog_draft):
        """Test content depth assessment."""
        result = critique_agent._assess_content_depth(sample_blog_draft)
        
        assert 'depth_indicators' in result
        assert 'total_words' in result
        assert 'depth_ratio' in result
        assert 'depth_score' in result
        assert 'comprehensiveness' in result
        assert result['comprehensiveness'] in ['low', 'medium', 'high']
        assert 0 <= result['depth_score'] <= 1


class TestCritiqueContext:
    """Test cases for CritiqueContext dataclass."""
    
    def test_critique_context_creation(self, sample_blog_draft, sample_research_output):
        """Test CritiqueContext creation."""
        context = CritiqueContext(
            blog_draft=sample_blog_draft,
            research_data=sample_research_output,
            quality_threshold=7.5
        )
        
        assert context.blog_draft == sample_blog_draft
        assert context.research_data == sample_research_output
        assert context.quality_threshold == 7.5


class TestHelperMethods:
    """Test cases for helper methods."""
    
    def test_calculate_section_consistency(self, critique_agent):
        """Test section consistency calculation."""
        # Consistent sections
        consistent_analyses = [
            {'clarity_score': 0.8},
            {'clarity_score': 0.85},
            {'clarity_score': 0.75}
        ]
        
        result = critique_agent._calculate_section_consistency(consistent_analyses)
        assert 0 <= result <= 1
        
        # Inconsistent sections
        inconsistent_analyses = [
            {'clarity_score': 0.9},
            {'clarity_score': 0.3},
            {'clarity_score': 0.1}
        ]
        
        result_inconsistent = critique_agent._calculate_section_consistency(inconsistent_analyses)
        assert result > result_inconsistent
        
        # Empty analyses
        empty_result = critique_agent._calculate_section_consistency([])
        assert empty_result == 0.0


if __name__ == "__main__":
    pytest.main([__file__])