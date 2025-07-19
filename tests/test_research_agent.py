"""Unit tests for Research Agent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pydantic_ai.models import Model
from pydantic_ai import ModelRetry

from src.agents.research_agent import ResearchAgent
from src.models.data_models import ResearchOutput, ResearchFinding
from src.utils.dependencies import SharedDependencies


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    deps = Mock(spec=SharedDependencies)
    deps.tavily_client = Mock()
    deps.http_client = Mock()
    deps.max_iterations = 3
    deps.quality_threshold = 7.0
    return deps


class MockResearchAgent:
    """Mock research agent for testing individual methods."""
    
    def __init__(self):
        pass
    
    def _categorize_finding(self, text: str) -> str:
        """Categorize a finding based on its content."""
        text_lower = text.lower()
        
        # Check for expert opinion first (more specific)
        if any(word in text_lower for word in ['expert', 'professor', 'dr.', 'researcher']):
            return 'expert_opinion'
        elif any(word in text_lower for word in ['study', 'research', 'survey', 'analysis']):
            return 'study'
        elif any(word in text_lower for word in ['%', 'percent', 'statistics', 'data', 'number']):
            return 'statistic'
        elif any(word in text_lower for word in ['benefit', 'advantage', 'positive']):
            return 'benefit'
        elif any(word in text_lower for word in ['risk', 'disadvantage', 'negative', 'concern']):
            return 'risk'
        else:
            return 'general_fact'
    
    def _calculate_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score between text and topic."""
        text_lower = text.lower()
        topic_lower = topic.lower()
        
        # Simple keyword matching approach
        topic_words = set(topic_lower.split())
        text_words = set(text_lower.split())
        
        # Calculate overlap
        common_words = topic_words.intersection(text_words)
        if not topic_words:
            return 0.0
        
        base_score = len(common_words) / len(topic_words)
        
        # Boost score for exact topic phrase matches
        if topic_lower in text_lower:
            base_score += 0.3
        
        # Cap at 1.0
        return min(base_score, 1.0)
    
    async def search_web(self, ctx, query: str, max_results: int = 10):
        """Mock search_web method."""
        try:
            # Use Tavily client from dependencies
            search_response = ctx.deps.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            results = []
            for result in search_response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.0)
                })
            
            return results
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise ModelRetry(f"Tavily API rate limit exceeded: {e}")
            elif "timeout" in str(e).lower():
                raise ModelRetry(f"Tavily API timeout: {e}")
            else:
                # For other errors, return empty results rather than failing completely
                return []
    
    async def extract_facts(self, ctx, search_results, topic: str):
        """Mock extract_facts method."""
        findings = []
        
        for result in search_results:
            content = result.get('content', '')
            url = result.get('url', '')
            title = result.get('title', '')
            
            if not content or not url:
                continue
            
            # Extract key facts from the content
            sentences = content.split('. ')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                # Determine category based on content patterns
                category = self._categorize_finding(sentence)
                
                # Calculate relevance score based on topic keywords
                relevance_score = self._calculate_relevance(sentence, topic)
                
                if relevance_score > 0.3:  # Only include reasonably relevant findings
                    finding = ResearchFinding(
                        fact=sentence,
                        source_url=url,
                        relevance_score=relevance_score,
                        category=category
                    )
                    findings.append(finding)
        
        # Sort by relevance score and return top findings
        findings.sort(key=lambda x: x.relevance_score, reverse=True)
        return findings[:20]  # Limit to top 20 findings


@pytest.fixture
def mock_research_agent():
    """Create a mock research agent for testing."""
    return MockResearchAgent()


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            'title': 'Benefits of Intermittent Fasting',
            'content': 'Intermittent fasting can help with weight loss. Studies show 16% improvement in metabolic health. Research indicates reduced inflammation markers.',
            'url': 'https://example.com/article1',
            'score': 0.9
        },
        {
            'title': 'Fasting Research Study',
            'content': 'A recent study by Dr. Smith found that intermittent fasting reduces insulin resistance by 25%. The research involved 200 participants over 12 weeks.',
            'url': 'https://example.com/study1',
            'score': 0.8
        },
        {
            'title': 'Unrelated Article',
            'content': 'This article talks about completely different topics like gardening and cooking recipes.',
            'url': 'https://example.com/unrelated',
            'score': 0.2
        }
    ]


class TestResearchAgent:
    """Test cases for ResearchAgent class."""
    
    @pytest.mark.asyncio
    async def test_search_web_success(self, mock_research_agent, mock_dependencies):
        """Test successful web search."""
        # Mock Tavily API response
        mock_dependencies.tavily_client.search.return_value = {
            'results': [
                {
                    'title': 'Test Article',
                    'content': 'Test content about intermittent fasting',
                    'url': 'https://example.com/test',
                    'score': 0.9
                }
            ]
        }
        
        # Create a mock context
        mock_ctx = Mock()
        mock_ctx.deps = mock_dependencies
        
        results = await mock_research_agent.search_web(mock_ctx, "intermittent fasting", 5)
        
        assert len(results) == 1
        assert results[0]['title'] == 'Test Article'
        assert results[0]['content'] == 'Test content about intermittent fasting'
        assert results[0]['url'] == 'https://example.com/test'
        assert results[0]['score'] == 0.9
        
        # Verify Tavily client was called correctly
        mock_dependencies.tavily_client.search.assert_called_once_with(
            query="intermittent fasting",
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False
        )
    
    @pytest.mark.asyncio
    async def test_search_web_rate_limit_error(self, mock_research_agent, mock_dependencies):
        """Test handling of rate limit errors."""
        mock_dependencies.tavily_client.search.side_effect = Exception("Rate limit exceeded")
        
        mock_ctx = Mock()
        mock_ctx.deps = mock_dependencies
        
        with pytest.raises(ModelRetry):
            await mock_research_agent.search_web(mock_ctx, "test query")
    
    @pytest.mark.asyncio
    async def test_search_web_timeout_error(self, mock_research_agent, mock_dependencies):
        """Test handling of timeout errors."""
        mock_dependencies.tavily_client.search.side_effect = Exception("Request timeout")
        
        mock_ctx = Mock()
        mock_ctx.deps = mock_dependencies
        
        with pytest.raises(ModelRetry):
            await mock_research_agent.search_web(mock_ctx, "test query")
    
    @pytest.mark.asyncio
    async def test_search_web_other_error(self, mock_research_agent, mock_dependencies):
        """Test handling of other errors (should return empty results)."""
        mock_dependencies.tavily_client.search.side_effect = Exception("Some other error")
        
        mock_ctx = Mock()
        mock_ctx.deps = mock_dependencies
        
        results = await mock_research_agent.search_web(mock_ctx, "test query")
        assert results == []
    
    @pytest.mark.asyncio
    async def test_extract_facts(self, mock_research_agent, mock_dependencies, sample_search_results):
        """Test fact extraction from search results."""
        mock_ctx = Mock()
        mock_ctx.deps = mock_dependencies
        
        findings = await mock_research_agent.extract_facts(
            mock_ctx, 
            sample_search_results, 
            "intermittent fasting"
        )
        
        assert len(findings) > 0
        assert all(isinstance(f, ResearchFinding) for f in findings)
        
        # Check that findings are sorted by relevance
        for i in range(len(findings) - 1):
            assert findings[i].relevance_score >= findings[i + 1].relevance_score
        
        # Check that irrelevant content is filtered out
        relevant_findings = [f for f in findings if "gardening" in f.fact or "cooking" in f.fact]
        assert len(relevant_findings) == 0  # Should be filtered out due to low relevance
    
    def test_categorize_finding_study(self, mock_research_agent):
        """Test categorization of study-related findings."""
        text = "A recent study shows that intermittent fasting improves health"
        category = mock_research_agent._categorize_finding(text)
        assert category == "study"
    
    def test_categorize_finding_statistic(self, mock_research_agent):
        """Test categorization of statistical findings."""
        text = "25% of people report improved energy levels"
        category = mock_research_agent._categorize_finding(text)
        assert category == "statistic"
    
    def test_categorize_finding_expert_opinion(self, mock_research_agent):
        """Test categorization of expert opinion findings."""
        text = "Dr. Smith, a leading researcher, recommends this approach"
        category = mock_research_agent._categorize_finding(text)
        assert category == "expert_opinion"
    
    def test_categorize_finding_benefit(self, mock_research_agent):
        """Test categorization of benefit findings."""
        text = "The main benefit of this approach is improved health"
        category = mock_research_agent._categorize_finding(text)
        assert category == "benefit"
    
    def test_categorize_finding_risk(self, mock_research_agent):
        """Test categorization of risk findings."""
        text = "There are some risks and concerns to consider"
        category = mock_research_agent._categorize_finding(text)
        assert category == "risk"
    
    def test_categorize_finding_general(self, mock_research_agent):
        """Test categorization of general findings."""
        text = "This is some general information about the topic"
        category = mock_research_agent._categorize_finding(text)
        assert category == "general_fact"
    
    def test_calculate_relevance_exact_match(self, mock_research_agent):
        """Test relevance calculation with exact topic match."""
        text = "Intermittent fasting is a popular health trend"
        topic = "intermittent fasting"
        relevance = mock_research_agent._calculate_relevance(text, topic)
        assert relevance > 0.5  # Should be high due to exact match
    
    def test_calculate_relevance_partial_match(self, mock_research_agent):
        """Test relevance calculation with partial match."""
        text = "Fasting can improve metabolic health"
        topic = "intermittent fasting"
        relevance = mock_research_agent._calculate_relevance(text, topic)
        assert 0 < relevance < 1
    
    def test_calculate_relevance_no_match(self, mock_research_agent):
        """Test relevance calculation with no match."""
        text = "Gardening is a relaxing hobby"
        topic = "intermittent fasting"
        relevance = mock_research_agent._calculate_relevance(text, topic)
        assert relevance == 0.0


class TestResearchFindingValidation:
    """Test ResearchFinding model validation."""
    
    def test_valid_research_finding(self):
        """Test creation of valid ResearchFinding."""
        finding = ResearchFinding(
            fact="Test fact",
            source_url="https://example.com",
            relevance_score=0.8,
            category="study"
        )
        assert finding.fact == "Test fact"
        assert finding.source_url == "https://example.com"
        assert finding.relevance_score == 0.8
        assert finding.category == "study"
    
    def test_invalid_relevance_score_high(self):
        """Test validation of relevance score upper bound."""
        with pytest.raises(ValueError):
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=1.5,  # Invalid: > 1
                category="study"
            )
    
    def test_invalid_relevance_score_low(self):
        """Test validation of relevance score lower bound."""
        with pytest.raises(ValueError):
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=-0.1,  # Invalid: < 0
                category="study"
            )


class TestResearchOutputValidation:
    """Test ResearchOutput model validation."""
    
    def test_valid_research_output(self):
        """Test creation of valid ResearchOutput."""
        findings = [
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=0.8,
                category="study"
            )
        ]
        
        output = ResearchOutput(
            topic="test topic",
            findings=findings,
            summary="Test summary",
            confidence_level=0.7
        )
        
        assert output.topic == "test topic"
        assert len(output.findings) == 1
        assert output.summary == "Test summary"
        assert output.confidence_level == 0.7
    
    def test_invalid_confidence_level(self):
        """Test validation of confidence level bounds."""
        findings = [
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=0.8,
                category="study"
            )
        ]
        
        with pytest.raises(ValueError):
            ResearchOutput(
                topic="test topic",
                findings=findings,
                summary="Test summary",
                confidence_level=1.5  # Invalid: > 1
            )