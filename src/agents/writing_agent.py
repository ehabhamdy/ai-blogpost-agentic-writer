"""Writing Agent for creating well-structured blog posts from research data."""

import asyncio
import logging
from typing import List, Dict, Any
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from ..models.data_models import BlogDraft, ResearchOutput, ResearchFinding
from ..utils.dependencies import SharedDependencies

from ..utils.exceptions import WritingError, ValidationError
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WritingContext:
    """Context class that holds research data for the Writing Agent."""
    topic: str
    research_findings: List[ResearchFinding]
    research_summary: str
    research_confidence: float


class WritingAgent:
    """Agent responsible for creating well-structured blog posts from research data."""
    
    def __init__(self, model: Model):
        """Initialize the Writing Agent with a model."""
        self.agent = Agent(
            model=model,
            output_type=BlogDraft,
            system_prompt="""You are a professional content writer specializing in creating 
            engaging, well-structured blog posts. Your goal is to transform research data into 
            compelling, readable content that informs and engages readers.
            
            Focus on:
            - Clear, engaging writing that flows naturally
            - Proper blog structure with introduction, body, and conclusion
            - Incorporating research findings seamlessly into the narrative
            - Maintaining reader engagement throughout
            - Creating compelling titles and smooth transitions
            
            IMPORTANT: Your output MUST include these required fields:
            - title: A compelling blog post title
            - introduction: An engaging opening paragraph
            - body_sections: A list of content sections for the main body
            - conclusion: A closing paragraph that summarizes key points
            
            Note: The word count will be calculated automatically by the system.
            
            Use the available tools to structure content and enhance readability."""
        )
        
        # Register tools
        self.agent.tool(self.structure_content)
        self.agent.tool(self.enhance_readability)
    
    async def create_blog_draft(
        self, 
        topic: str, 
        research_data: ResearchOutput, 
        deps: SharedDependencies
    ) -> BlogDraft:
        """Create a blog draft from research data."""
        # Validate inputs
        if not topic or not topic.strip():
            raise ValidationError(
                "Topic cannot be empty",
                field_name="topic",
                invalid_value=topic
            )
        
        if not research_data or not research_data.findings:
            logger.warning(f"Creating draft for '{topic}' with limited research data")
        
        try:
            # Create a context that includes the research data
            context = WritingContext(
                topic=topic,
                research_findings=research_data.findings,
                research_summary=research_data.summary,
                research_confidence=research_data.confidence_level
            )
            
            result = await self.agent.run(
                f"""Create a comprehensive blog post about: {topic}
                
                You have access to research data through the context. Use the structure_content tool 
                to organize the research findings into a well-structured blog post.
                
                The research includes {len(research_data.findings)} findings with a confidence level of {research_data.confidence_level:.2f}.
                Research summary: {research_data.summary}
                
                Steps to follow:
                1. Use structure_content tool to organize the research findings
                2. Create an engaging blog post (800-1200 words) incorporating the structured content
                3. Ensure proper flow between introduction, body sections, and conclusion
                
                Create a blog post that is approximately 800-1200 words.""",
                deps=context
            )
            
            # Validate and ensure complete output
            draft = result.output
            
            # Check for required fields and add them if missing
            if not draft.title or not draft.introduction or not draft.body_sections:
                raise ValidationError(
                    "Generated draft is missing required sections",
                    field_name="blog_draft",
                    invalid_value="incomplete_draft"
                )
            
            # Ensure conclusion exists
            if not hasattr(draft, 'conclusion') or not draft.conclusion:
                # Generate a simple conclusion from the body content
                body_text = " ".join(draft.body_sections)
                draft.conclusion = f"In conclusion, {draft.title.split(':')[0] if ':' in draft.title else draft.title} offers significant benefits worth considering. As we've explored in this article, the evidence supports incorporating this practice into your routine for optimal results."
                logger.warning("Added missing conclusion to draft")
            
            # Always calculate word count programmatically for accuracy
            all_text = f"{draft.title} {draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
            draft.word_count = len(all_text.split())
            logger.info(f"Calculated word count: {draft.word_count}")
            
            return draft
            
        except Exception as e:
            # Use ModelRetry for retryable errors
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Writing failed due to API issues: {e}")
            
            # Handle validation errors and other exceptions
            if isinstance(e, ValidationError):
                raise e
            else:
                raise WritingError(
                    f"Failed to create blog draft for topic '{topic}': {e}",
                    topic=topic,
                    draft_stage="initial",
                    original_error=e
                )
    
    async def revise_blog_draft(
        self,
        original_draft: BlogDraft,
        feedback: str,
        research_data: ResearchOutput,
        deps: SharedDependencies
    ) -> BlogDraft:
        """Revise a blog draft based on feedback."""
        # Validate inputs
        if not original_draft:
            raise ValidationError(
                "Original draft cannot be None",
                field_name="original_draft",
                invalid_value=None
            )
        
        if not feedback or not feedback.strip():
            raise ValidationError(
                "Feedback cannot be empty",
                field_name="feedback",
                invalid_value=feedback
            )
        
        try:
            # Create a context that includes the research data
            context = WritingContext(
                topic=research_data.topic,
                research_findings=research_data.findings,
                research_summary=research_data.summary,
                research_confidence=research_data.confidence_level
            )
            
            result = await self.agent.run(
                f"""Revise the following blog post based on the provided feedback:
                
                Original Title: {original_draft.title}
                Original Introduction: {original_draft.introduction}
                Original Body Sections: {len(original_draft.body_sections)} sections
                Original Conclusion: {original_draft.conclusion}
                
                Feedback to address: {feedback}
                
                You have access to research data through the context. Use the available tools 
                to support your revisions and ensure accuracy. Maintain the overall structure 
                while improving based on the feedback.""",
                deps=context
            )
            
            # Validate and ensure complete revised output
            revised_draft = result.output
            
            # Check for required fields and add them if missing
            if not revised_draft.title or not revised_draft.introduction or not revised_draft.body_sections:
                raise ValidationError(
                    "Revised draft is missing required sections",
                    field_name="revised_draft",
                    invalid_value="incomplete_revision"
                )
            
            # Ensure conclusion exists
            if not hasattr(revised_draft, 'conclusion') or not revised_draft.conclusion:
                # Generate a simple conclusion from the body content or keep original
                if hasattr(original_draft, 'conclusion') and original_draft.conclusion:
                    revised_draft.conclusion = original_draft.conclusion
                else:
                    body_text = " ".join(revised_draft.body_sections)
                    revised_draft.conclusion = f"In conclusion, {revised_draft.title.split(':')[0] if ':' in revised_draft.title else revised_draft.title} offers significant benefits worth considering. As we've explored in this article, the evidence supports incorporating this practice into your routine for optimal results."
                logger.warning("Added missing conclusion to revised draft")
            
            # Always calculate word count programmatically for accuracy
            all_text = f"{revised_draft.title} {revised_draft.introduction} {' '.join(revised_draft.body_sections)} {revised_draft.conclusion}"
            revised_draft.word_count = len(all_text.split())
            logger.info(f"Calculated word count for revised draft: {revised_draft.word_count}")
            
            return revised_draft
            
        except Exception as e:
            # Use ModelRetry for retryable errors
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Revision failed due to API issues: {e}")
            
            # Handle validation errors and other exceptions
            if isinstance(e, ValidationError):
                raise e
            else:
                raise WritingError(
                    f"Failed to revise blog draft '{original_draft.title}': {e}",
                    topic=research_data.topic,
                    draft_stage="revision",
                    original_error=e
                )
    
    async def structure_content(
        self, 
        ctx: RunContext[WritingContext]
    ) -> Dict[str, Any]:
        """Organize research data into blog sections.
        
        Uses the research findings and topic from the context to create a structured
        content organization for the blog post.
        
        Returns:
            Dictionary with organized content structure
        """
        # Get data from context
        research_findings = ctx.deps.research_findings
        topic = ctx.deps.topic
        
        # Group findings by category
        categorized_findings = {}
        for finding in research_findings:
            category = finding.category
            if category not in categorized_findings:
                categorized_findings[category] = []
            categorized_findings[category].append(finding)
        
        # Create content structure
        structure = {
            'title_suggestions': self._generate_title_suggestions(topic, research_findings),
            'introduction_points': self._extract_introduction_points(research_findings),
            'body_sections': self._organize_body_sections(categorized_findings, topic),
            'conclusion_points': self._extract_conclusion_points(research_findings),
            'key_statistics': self._extract_key_statistics(research_findings),
            'expert_quotes': self._extract_expert_opinions(research_findings)
        }
        
        return structure
    
    async def enhance_readability(
        self, 
        ctx: RunContext[WritingContext], 
        content: str,
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """Improve content flow and readability.
        
        Args:
            content: The content to enhance
            target_audience: Target audience level (general, technical, beginner)
            
        Returns:
            Dictionary with readability improvements
        """
        improvements = {
            'transition_suggestions': self._suggest_transitions(content),
            'sentence_improvements': self._improve_sentences(content),
            'paragraph_structure': self._analyze_paragraph_structure(content),
            'readability_score': self._calculate_readability_score(content),
            'vocabulary_suggestions': self._suggest_vocabulary_improvements(content, target_audience)
        }
        
        return improvements
    
    def _generate_title_suggestions(self, topic: str, findings: List[ResearchFinding]) -> List[str]:
        """Generate compelling title suggestions based on topic and findings."""
        titles = []
        
        # Basic title
        titles.append(f"The Complete Guide to {topic.title()}")
        
        # Benefit-focused titles
        benefit_findings = [f for f in findings if f.category == 'benefit']
        if benefit_findings:
            titles.append(f"How {topic.title()} Can Transform Your Health")
            titles.append(f"The Science-Backed Benefits of {topic.title()}")
        
        # Statistic-focused titles
        stat_findings = [f for f in findings if f.category == 'statistic']
        if stat_findings:
            titles.append(f"What the Research Really Says About {topic.title()}")
            titles.append(f"The Numbers Don't Lie: {topic.title()} Facts")
        
        # Question-based titles
        titles.append(f"Is {topic.title()} Right for You? A Complete Analysis")
        titles.append(f"Everything You Need to Know About {topic.title()}")
        
        return titles[:5]  # Return top 5 suggestions
    
    def _extract_introduction_points(self, findings: List[ResearchFinding]) -> List[str]:
        """Extract key points for the introduction."""
        points = []
        
        # Get high-relevance findings for introduction
        high_relevance = [f for f in findings if f.relevance_score > 0.7]
        
        # Add overview points
        if high_relevance:
            points.append("Brief overview of the topic's importance")
            points.append("Key statistics or compelling facts")
            points.append("What readers will learn from the article")
        
        # Add specific compelling facts
        for finding in high_relevance[:3]:  # Top 3 most relevant
            if finding.category in ['statistic', 'study']:
                points.append(f"Mention: {finding.fact[:100]}...")
        
        return points
    
    def _organize_body_sections(self, categorized_findings: Dict[str, List[ResearchFinding]], topic: str) -> List[Dict[str, Any]]:
        """Organize findings into logical body sections."""
        sections = []
        
        # Define section order and titles
        section_mapping = {
            'study': 'What the Research Shows',
            'statistic': 'Key Statistics and Data',
            'benefit': 'Benefits and Advantages',
            'risk': 'Potential Risks and Considerations',
            'expert_opinion': 'Expert Perspectives',
            'general_fact': 'Important Facts to Know'
        }
        
        for category, title in section_mapping.items():
            if category in categorized_findings and categorized_findings[category]:
                findings = categorized_findings[category]
                # Sort by relevance
                findings.sort(key=lambda x: x.relevance_score, reverse=True)
                
                section = {
                    'title': title,
                    'category': category,
                    'findings': findings[:5],  # Top 5 findings per section
                    'key_points': [f.fact for f in findings[:3]]  # Top 3 as key points
                }
                sections.append(section)
        
        return sections
    
    def _extract_conclusion_points(self, findings: List[ResearchFinding]) -> List[str]:
        """Extract key points for the conclusion."""
        points = []
        
        # Get most relevant findings
        top_findings = sorted(findings, key=lambda x: x.relevance_score, reverse=True)[:5]
        
        points.append("Summarize key takeaways")
        points.append("Reinforce main benefits or findings")
        points.append("Provide actionable next steps for readers")
        
        # Add specific summary points
        for finding in top_findings[:2]:
            if finding.category in ['benefit', 'study']:
                points.append(f"Highlight: {finding.fact[:80]}...")
        
        return points
    
    def _extract_key_statistics(self, findings: List[ResearchFinding]) -> List[str]:
        """Extract key statistics for emphasis."""
        stats = []
        stat_findings = [f for f in findings if f.category == 'statistic']
        
        # Sort by relevance and extract top statistics
        stat_findings.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for finding in stat_findings[:5]:
            stats.append(finding.fact)
        
        return stats
    
    def _extract_expert_opinions(self, findings: List[ResearchFinding]) -> List[Dict[str, str]]:
        """Extract expert opinions for quotes."""
        quotes = []
        expert_findings = [f for f in findings if f.category == 'expert_opinion']
        
        for finding in expert_findings[:3]:  # Top 3 expert opinions
            quotes.append({
                'quote': finding.fact,
                'source': finding.source_url,
                'relevance': finding.relevance_score
            })
        
        return quotes
    
    def _suggest_transitions(self, content: str) -> List[str]:
        """Suggest transition phrases for better flow."""
        transitions = [
            "Furthermore, research indicates that...",
            "In addition to these benefits...",
            "However, it's important to consider...",
            "On the other hand...",
            "Building on this evidence...",
            "More importantly...",
            "As a result of these findings...",
            "Despite these advantages...",
            "To put this in perspective...",
            "Given these considerations..."
        ]
        
        # Return relevant transitions based on content length
        return transitions[:5]
    
    def _improve_sentences(self, content: str) -> List[str]:
        """Suggest sentence improvements."""
        suggestions = []
        
        sentences = content.split('. ')
        for sentence in sentences[:5]:  # Analyze first 5 sentences
            if len(sentence) > 30:  # Long sentences
                suggestions.append(f"Consider breaking down: '{sentence[:50]}...'")
        
        return suggestions
    
    def _analyze_paragraph_structure(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph structure."""
        paragraphs = content.split('\n\n')
        
        analysis = {
            'paragraph_count': len(paragraphs),
            'average_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            'suggestions': []
        }
        
        # Add suggestions based on analysis
        if analysis['average_length'] > 100:
            analysis['suggestions'].append("Consider shorter paragraphs for better readability")
        
        if analysis['paragraph_count'] < 3:
            analysis['suggestions'].append("Consider adding more paragraph breaks")
        
        return analysis
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score."""
        words = content.split()
        sentences = content.split('.')
        
        if not sentences or not words:
            return 0.0
        
        # Simple readability calculation (higher is better)
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Ideal range is 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            score = 1.0
        elif avg_words_per_sentence < 15:
            score = 0.8
        else:
            score = max(0.5, 1.0 - (avg_words_per_sentence - 20) * 0.02)
        
        return min(score, 1.0)
    
    def _suggest_vocabulary_improvements(self, content: str, target_audience: str) -> List[str]:
        """Suggest vocabulary improvements based on target audience."""
        suggestions = []
        
        # Complex words that might need simplification
        complex_words = {
            'utilize': 'use',
            'demonstrate': 'show',
            'facilitate': 'help',
            'implement': 'put in place',
            'subsequently': 'then',
            'approximately': 'about'
        }
        
        if target_audience == "general":
            for complex_word, simple_word in complex_words.items():
                if complex_word in content.lower():
                    suggestions.append(f"Consider replacing '{complex_word}' with '{simple_word}'")
        
        return suggestions[:3]  # Return top 3 suggestions