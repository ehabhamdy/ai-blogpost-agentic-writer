"""Research Agent for gathering comprehensive research on topics."""

import asyncio
from typing import List, Dict, Any
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from ..models.data_models import ResearchOutput, ResearchFinding
from ..utils.dependencies import SharedDependencies


class ResearchAgent:
    """Agent responsible for researching topics using web search."""
    
    def __init__(self, model: Model):
        """Initialize the Research Agent with a model."""
        self.agent = Agent(
            model=model,
            result_type=ResearchOutput,
            system_prompt="""You are a research specialist tasked with gathering comprehensive, 
            factual information on given topics. Your goal is to find relevant facts, statistics, 
            studies, and expert opinions that will inform high-quality blog content.
            
            Focus on:
            - Credible sources and recent information
            - Diverse perspectives and comprehensive coverage
            - Factual accuracy and source attribution
            - Relevance to the specific topic
            
            Use the comprehensive_research tool to gather all information about the topic.
            This tool will handle web searching, fact extraction, categorization, and analysis automatically."""
        )
        
        # Register the comprehensive research tool
        self.agent.tool(self.comprehensive_research)
    
    async def research_topic(self, topic: str, deps: SharedDependencies) -> ResearchOutput:
        """Research a topic and return structured findings."""
        try:
            result = await self.agent.run(
                f"Research the topic: {topic}. Gather comprehensive information including facts, statistics, studies, and expert opinions.",
                deps=deps
            )
            return result.data
        except Exception as e:
            # Retry with exponential backoff for recoverable errors
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Research failed due to API issues: {e}")
            raise e
    
    async def comprehensive_research(
        self, 
        ctx: RunContext[SharedDependencies], 
        topic: str
    ) -> ResearchOutput:
        """Comprehensive research tool that handles the entire research workflow.
        
        Args:
            topic: The research topic to investigate
            
        Returns:
            Complete ResearchOutput with findings, summary, and confidence level
        """
        try:
            # Step 1: Search for information
            search_results = await self._search_web(ctx.deps, topic)
            
            # Step 2: Extract facts from search results
            findings = await self._extract_facts(ctx.deps, search_results, topic)
            
            # Step 3: Create summary
            summary = self._create_summary(topic, findings)
            
            # Step 4: Calculate confidence level
            confidence_level = self._calculate_confidence(findings)
            
            return ResearchOutput(
                topic=topic,
                findings=findings,
                summary=summary,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            # Handle errors gracefully
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Research failed due to API issues: {e}")
            
            # Return minimal result for other errors
            return ResearchOutput(
                topic=topic,
                findings=[],
                summary=f"Limited research data available for {topic} due to technical issues.",
                confidence_level=0.1
            )

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
    
    async def _search_web(self, deps: SharedDependencies, topic: str) -> List[Dict[str, Any]]:
        """Internal method to search the web for information."""
        try:
            # Use Tavily client from dependencies
            search_response = deps.tavily_client.search(
                query=topic,
                search_depth="advanced",
                max_results=10,
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
    
    async def _extract_facts(self, deps: SharedDependencies, search_results: List[Dict[str, Any]], topic: str) -> List[ResearchFinding]:
        """Internal method to extract facts from search results."""
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
    
    def _create_summary(self, topic: str, findings: List[ResearchFinding]) -> str:
        """Create a summary of the research findings."""
        if not findings:
            return f"Limited research data available for {topic}."
        
        summary_parts = [
            f"Research on {topic} reveals several key insights:",
        ]
        
        # Group findings by category for better summary
        categories = {}
        for finding in findings[:10]:  # Use top 10 findings
            if finding.category not in categories:
                categories[finding.category] = []
            categories[finding.category].append(finding.fact)
        
        # Add category-based insights
        if 'statistic' in categories:
            summary_parts.append("Statistical data shows important trends and measurements.")
        if 'study' in categories:
            summary_parts.append("Multiple studies provide evidence-based insights.")
        if 'expert_opinion' in categories:
            summary_parts.append("Expert perspectives offer professional guidance.")
        if 'benefit' in categories:
            summary_parts.append("Research highlights significant benefits and advantages.")
        if 'risk' in categories:
            summary_parts.append("Important considerations and potential risks are identified.")
        
        return " ".join(summary_parts)
    
    def _calculate_confidence(self, findings: List[ResearchFinding]) -> float:
        """Calculate confidence level based on research findings quality."""
        if not findings:
            return 0.0
        
        # Base confidence on number of findings and their relevance scores
        avg_relevance = sum(f.relevance_score for f in findings) / len(findings)
        finding_count_factor = min(len(findings) / 20, 1.0)  # Max confidence at 20+ findings
        
        # Boost confidence for diverse categories
        categories = set(f.category for f in findings)
        category_diversity_factor = min(len(categories) / 6, 1.0)  # Max boost at 6 categories
        
        # Calculate final confidence
        confidence = (avg_relevance * 0.5) + (finding_count_factor * 0.3) + (category_diversity_factor * 0.2)
        
        return min(confidence, 1.0)
