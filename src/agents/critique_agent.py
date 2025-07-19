"""Critique Agent for providing editorial analysis and feedback on blog drafts."""

import asyncio
from typing import List, Dict, Any
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from ..models.data_models import (
    CritiqueOutput, 
    CritiqueFeedback, 
    CritiqueSeverity, 
    BlogDraft, 
    ResearchOutput,
    ResearchFinding
)
from ..utils.dependencies import SharedDependencies
from dataclasses import dataclass


@dataclass
class CritiqueContext:
    """Context class that holds draft and research data for the Critique Agent."""
    blog_draft: BlogDraft
    research_data: ResearchOutput
    quality_threshold: float


class CritiqueAgent:
    """Agent responsible for providing editorial analysis and feedback on blog drafts."""
    
    def __init__(self, model: Model):
        """Initialize the Critique Agent with a model."""
        self.agent = Agent(
            model=model,
            result_type=CritiqueOutput,
            system_prompt="""You are a professional editor and content critic with expertise in 
            evaluating blog posts for clarity, accuracy, structure, and overall quality. Your role 
            is to provide constructive, actionable feedback that helps improve content quality.
            
            Focus on:
            - Clear communication and readability
            - Factual accuracy and proper use of research
            - Content structure and logical flow
            - Grammar, style, and tone consistency
            - Overall engagement and reader value
            
            Provide specific, actionable feedback with severity ratings. Be thorough but constructive.
            Use the available tools to analyze different aspects of the content systematically."""
        )
        
        # Register tools
        self.agent.tool(self.analyze_clarity)
        self.agent.tool(self.verify_facts)
        self.agent.tool(self.assess_structure)
    
    async def critique_blog_draft(
        self, 
        blog_draft: BlogDraft, 
        research_data: ResearchOutput,
        deps: SharedDependencies
    ) -> CritiqueOutput:
        """Provide comprehensive critique of a blog draft."""
        try:
            # Create a context that includes both the draft and research data
            context = CritiqueContext(
                blog_draft=blog_draft,
                research_data=research_data,
                quality_threshold=deps.quality_threshold
            )
            
            result = await self.agent.run(
                f"""Provide a comprehensive critique of this blog post:
                
                Title: {blog_draft.title}
                Word Count: {blog_draft.word_count}
                Sections: Introduction + {len(blog_draft.body_sections)} body sections + Conclusion
                
                Research Context:
                - Topic: {research_data.topic}
                - Research findings: {len(research_data.findings)} items
                - Research confidence: {research_data.confidence_level:.2f}
                
                Use all available tools to analyze:
                1. analyze_clarity - Check communication clarity and readability
                2. verify_facts - Cross-reference claims with research data
                3. assess_structure - Evaluate content organization and flow
                
                Provide specific, actionable feedback with appropriate severity levels.
                Quality threshold for approval: {deps.quality_threshold}/10""",
                deps=context
            )
            return result.data
        except Exception as e:
            # Retry with exponential backoff for recoverable errors
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Critique failed due to API issues: {e}")
            raise e
    
    async def analyze_clarity(
        self, 
        ctx: RunContext[CritiqueContext]
    ) -> Dict[str, Any]:
        """Check for clear communication and readability.
        
        Analyzes the blog draft for clarity issues including:
        - Sentence complexity and length
        - Paragraph structure and flow
        - Vocabulary appropriateness
        - Transition quality
        - Overall readability
        
        Returns:
            Dictionary with clarity analysis results
        """
        draft = ctx.deps.blog_draft
        
        # Analyze different sections
        clarity_analysis = {
            'title_clarity': self._analyze_title_clarity(draft.title),
            'introduction_clarity': self._analyze_section_clarity(draft.introduction, "introduction"),
            'body_clarity': self._analyze_body_sections_clarity(draft.body_sections),
            'conclusion_clarity': self._analyze_section_clarity(draft.conclusion, "conclusion"),
            'overall_readability': self._calculate_overall_readability(draft),
            'transition_quality': self._assess_transitions(draft),
            'vocabulary_assessment': self._assess_vocabulary_level(draft)
        }
        
        return clarity_analysis
    
    async def verify_facts(
        self, 
        ctx: RunContext[CritiqueContext]
    ) -> Dict[str, Any]:
        """Cross-reference claims with original research data.
        
        Verifies factual accuracy by:
        - Checking if claims are supported by research findings
        - Identifying unsupported statements
        - Assessing proper attribution of sources
        - Evaluating statistical accuracy
        
        Returns:
            Dictionary with fact verification results
        """
        draft = ctx.deps.blog_draft
        research = ctx.deps.research_data
        
        # Extract claims from the draft
        all_content = f"{draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        
        verification_results = {
            'supported_claims': self._identify_supported_claims(all_content, research.findings),
            'unsupported_claims': self._identify_unsupported_claims(all_content, research.findings),
            'source_attribution': self._check_source_attribution(all_content, research.findings),
            'statistical_accuracy': self._verify_statistics(all_content, research.findings),
            'research_utilization': self._assess_research_utilization(all_content, research.findings),
            'fact_density': self._calculate_fact_density(all_content, research.findings)
        }
        
        return verification_results
    
    async def assess_structure(
        self, 
        ctx: RunContext[CritiqueContext]
    ) -> Dict[str, Any]:
        """Evaluate content organization and flow.
        
        Assesses structural elements including:
        - Logical progression of ideas
        - Section balance and coherence
        - Introduction and conclusion effectiveness
        - Overall narrative flow
        
        Returns:
            Dictionary with structure assessment results
        """
        draft = ctx.deps.blog_draft
        
        structure_assessment = {
            'introduction_effectiveness': self._assess_introduction(draft.introduction),
            'body_organization': self._assess_body_organization(draft.body_sections),
            'conclusion_effectiveness': self._assess_conclusion(draft.conclusion),
            'logical_flow': self._assess_logical_flow(draft),
            'section_balance': self._assess_section_balance(draft),
            'narrative_coherence': self._assess_narrative_coherence(draft),
            'content_depth': self._assess_content_depth(draft)
        }
        
        return structure_assessment
    
    # Helper methods for analyze_clarity
    def _analyze_title_clarity(self, title: str) -> Dict[str, Any]:
        """Analyze title clarity and effectiveness."""
        analysis = {
            'length': len(title),
            'word_count': len(title.split()),
            'clarity_score': 0.0,
            'issues': []
        }
        
        # Check title length (ideal: 50-60 characters)
        if len(title) > 70:
            analysis['issues'].append("Title may be too long for optimal SEO")
        elif len(title) < 30:
            analysis['issues'].append("Title may be too short to be descriptive")
        
        # Check for clarity indicators
        if any(word in title.lower() for word in ['guide', 'how', 'what', 'why', 'benefits']):
            analysis['clarity_score'] += 0.3
        
        # Check for compelling elements
        if any(word in title.lower() for word in ['complete', 'ultimate', 'essential', 'proven']):
            analysis['clarity_score'] += 0.2
        
        # Base clarity score
        analysis['clarity_score'] = min(analysis['clarity_score'] + 0.5, 1.0)
        
        return analysis
    
    def _analyze_section_clarity(self, content: str, section_type: str) -> Dict[str, Any]:
        """Analyze clarity of a specific section."""
        sentences = content.split('. ')
        words = content.split()
        
        analysis = {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'clarity_score': 0.0,
            'issues': []
        }
        
        # Check sentence length (ideal: 15-20 words)
        if analysis['avg_sentence_length'] > 25:
            analysis['issues'].append(f"{section_type.title()} has overly complex sentences")
            analysis['clarity_score'] -= 0.2
        elif analysis['avg_sentence_length'] < 10:
            analysis['issues'].append(f"{section_type.title()} sentences may be too choppy")
            analysis['clarity_score'] -= 0.1
        else:
            analysis['clarity_score'] += 0.3
        
        # Check for passive voice indicators
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for word in words if word.lower() in passive_indicators)
        if passive_count > len(words) * 0.1:  # More than 10% passive voice
            analysis['issues'].append(f"{section_type.title()} uses too much passive voice")
            analysis['clarity_score'] -= 0.1
        
        # Base score
        analysis['clarity_score'] = max(analysis['clarity_score'] + 0.7, 0.0)
        analysis['clarity_score'] = min(analysis['clarity_score'], 1.0)
        
        return analysis
    
    def _analyze_body_sections_clarity(self, body_sections: List[str]) -> Dict[str, Any]:
        """Analyze clarity across all body sections."""
        total_clarity = 0.0
        section_analyses = []
        
        for i, section in enumerate(body_sections):
            section_analysis = self._analyze_section_clarity(section, f"body_section_{i+1}")
            section_analyses.append(section_analysis)
            total_clarity += section_analysis['clarity_score']
        
        return {
            'average_clarity': total_clarity / len(body_sections) if body_sections else 0.0,
            'section_analyses': section_analyses,
            'consistency_score': self._calculate_section_consistency(section_analyses)
        }
    
    def _calculate_overall_readability(self, draft: BlogDraft) -> Dict[str, Any]:
        """Calculate overall readability metrics."""
        all_content = f"{draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        words = all_content.split()
        sentences = all_content.split('.')
        
        readability = {
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'readability_score': 0.0
        }
        
        # Simple readability calculation
        if 15 <= readability['avg_words_per_sentence'] <= 20:
            readability['readability_score'] = 1.0
        elif readability['avg_words_per_sentence'] < 15:
            readability['readability_score'] = 0.8
        else:
            readability['readability_score'] = max(0.5, 1.0 - (readability['avg_words_per_sentence'] - 20) * 0.02)
        
        return readability
    
    def _assess_transitions(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess quality of transitions between sections."""
        transition_words = [
            'however', 'furthermore', 'moreover', 'additionally', 'consequently',
            'therefore', 'meanwhile', 'similarly', 'in contrast', 'on the other hand'
        ]
        
        all_content = f"{draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        words = all_content.lower().split()
        
        transition_count = sum(1 for word in words if word in transition_words)
        
        return {
            'transition_count': transition_count,
            'transition_density': transition_count / len(words) if words else 0,
            'quality_score': min(transition_count / 10, 1.0)  # Normalize to 0-1
        }
    
    def _assess_vocabulary_level(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess vocabulary complexity and appropriateness."""
        all_content = f"{draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        words = all_content.split()
        
        # Simple vocabulary complexity indicators
        complex_words = [word for word in words if len(word) > 8]
        
        return {
            'total_words': len(words),
            'complex_words': len(complex_words),
            'complexity_ratio': len(complex_words) / len(words) if words else 0,
            'appropriateness_score': 1.0 - min(len(complex_words) / len(words), 0.3) if words else 0
        }
    
    # Helper methods for verify_facts
    def _identify_supported_claims(self, content: str, findings: List[ResearchFinding]) -> List[Dict[str, Any]]:
        """Identify claims that are supported by research findings."""
        supported_claims = []
        
        # Simple keyword matching approach
        for finding in findings:
            finding_keywords = finding.fact.lower().split()[:5]  # First 5 words as keywords
            
            for keyword in finding_keywords:
                if len(keyword) > 3 and keyword in content.lower():
                    supported_claims.append({
                        'claim_context': keyword,
                        'supporting_finding': finding.fact,
                        'source': finding.source_url,
                        'relevance': finding.relevance_score
                    })
                    break  # Avoid duplicate matches for same finding
        
        return supported_claims[:10]  # Return top 10 matches
    
    def _identify_unsupported_claims(self, content: str, findings: List[ResearchFinding]) -> List[str]:
        """Identify potential claims that lack research support."""
        # Look for claim indicators
        claim_indicators = [
            'studies show', 'research indicates', 'according to', 'data suggests',
            'experts believe', 'evidence shows', 'statistics reveal'
        ]
        
        unsupported = []
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in claim_indicators):
                # Check if this claim has supporting research
                has_support = False
                for finding in findings:
                    if any(word in sentence_lower for word in finding.fact.lower().split()[:3]):
                        has_support = True
                        break
                
                if not has_support:
                    unsupported.append(sentence.strip())
        
        return unsupported[:5]  # Return top 5 unsupported claims
    
    def _check_source_attribution(self, content: str, findings: List[ResearchFinding]) -> Dict[str, Any]:
        """Check if sources are properly attributed."""
        # Look for source attribution patterns
        attribution_patterns = ['according to', 'source:', 'study by', 'research from']
        
        attribution_count = sum(1 for pattern in attribution_patterns if pattern in content.lower())
        
        return {
            'attribution_count': attribution_count,
            'total_findings': len(findings),
            'attribution_ratio': attribution_count / len(findings) if findings else 0,
            'attribution_score': min(attribution_count / max(len(findings) * 0.3, 1), 1.0)
        }
    
    def _verify_statistics(self, content: str, findings: List[ResearchFinding]) -> Dict[str, Any]:
        """Verify statistical claims against research data."""
        import re
        
        # Find numbers in content (simple regex for percentages and numbers)
        numbers_in_content = re.findall(r'\d+(?:\.\d+)?%?', content)
        
        stat_findings = [f for f in findings if f.category == 'statistic']
        numbers_in_research = []
        for finding in stat_findings:
            numbers_in_research.extend(re.findall(r'\d+(?:\.\d+)?%?', finding.fact))
        
        return {
            'numbers_in_content': len(numbers_in_content),
            'numbers_in_research': len(numbers_in_research),
            'statistical_findings': len(stat_findings),
            'verification_score': min(len(numbers_in_research) / max(len(numbers_in_content), 1), 1.0) if numbers_in_content else 1.0
        }
    
    def _assess_research_utilization(self, content: str, findings: List[ResearchFinding]) -> Dict[str, Any]:
        """Assess how well the research findings are utilized."""
        utilized_findings = 0
        
        for finding in findings:
            # Check if key terms from the finding appear in content
            finding_words = finding.fact.lower().split()
            key_words = [word for word in finding_words if len(word) > 4][:3]  # Top 3 key words
            
            if any(word in content.lower() for word in key_words):
                utilized_findings += 1
        
        return {
            'total_findings': len(findings),
            'utilized_findings': utilized_findings,
            'utilization_rate': utilized_findings / len(findings) if findings else 0,
            'utilization_score': utilized_findings / len(findings) if findings else 1.0
        }
    
    def _calculate_fact_density(self, content: str, findings: List[ResearchFinding]) -> Dict[str, Any]:
        """Calculate the density of factual information in the content."""
        words = content.split()
        fact_indicators = ['study', 'research', 'data', 'evidence', 'according', 'shows', 'indicates']
        
        fact_word_count = sum(1 for word in words if word.lower() in fact_indicators)
        
        return {
            'total_words': len(words),
            'fact_indicators': fact_word_count,
            'fact_density': fact_word_count / len(words) if words else 0,
            'density_score': min(fact_word_count / (len(words) * 0.05), 1.0) if words else 0  # Target 5% fact density
        }
    
    # Helper methods for assess_structure
    def _assess_introduction(self, introduction: str) -> Dict[str, Any]:
        """Assess the effectiveness of the introduction."""
        words = introduction.split()
        sentences = introduction.split('.')
        
        assessment = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'effectiveness_score': 0.0,
            'strengths': [],
            'weaknesses': []
        }
        
        # Check length (ideal: 100-150 words)
        if 100 <= len(words) <= 150:
            assessment['effectiveness_score'] += 0.3
            assessment['strengths'].append("Appropriate length")
        elif len(words) < 100:
            assessment['weaknesses'].append("Introduction may be too brief")
        else:
            assessment['weaknesses'].append("Introduction may be too lengthy")
        
        # Check for hook elements
        hook_indicators = ['imagine', 'what if', 'did you know', 'surprising', 'shocking']
        if any(indicator in introduction.lower() for indicator in hook_indicators):
            assessment['effectiveness_score'] += 0.2
            assessment['strengths'].append("Contains engaging hook")
        
        # Check for preview of content
        preview_indicators = ['will explore', 'will discuss', 'will cover', 'this article']
        if any(indicator in introduction.lower() for indicator in preview_indicators):
            assessment['effectiveness_score'] += 0.2
            assessment['strengths'].append("Previews article content")
        
        # Base score
        assessment['effectiveness_score'] = min(assessment['effectiveness_score'] + 0.3, 1.0)
        
        return assessment
    
    def _assess_body_organization(self, body_sections: List[str]) -> Dict[str, Any]:
        """Assess the organization of body sections."""
        assessment = {
            'section_count': len(body_sections),
            'organization_score': 0.0,
            'balance_score': 0.0,
            'issues': []
        }
        
        if not body_sections:
            assessment['issues'].append("No body sections found")
            return assessment
        
        # Check section count (ideal: 3-5 sections)
        if 3 <= len(body_sections) <= 5:
            assessment['organization_score'] += 0.4
        elif len(body_sections) < 3:
            assessment['issues'].append("Too few body sections for comprehensive coverage")
        else:
            assessment['issues'].append("Too many body sections may overwhelm readers")
        
        # Check section balance
        word_counts = [len(section.split()) for section in body_sections]
        avg_words = sum(word_counts) / len(word_counts)
        
        # Check if sections are reasonably balanced (within 50% of average)
        balanced_sections = sum(1 for count in word_counts if abs(count - avg_words) <= avg_words * 0.5)
        assessment['balance_score'] = balanced_sections / len(body_sections)
        
        if assessment['balance_score'] < 0.7:
            assessment['issues'].append("Sections are unevenly balanced")
        
        # Overall organization score
        assessment['organization_score'] = min(assessment['organization_score'] + assessment['balance_score'] * 0.6, 1.0)
        
        return assessment
    
    def _assess_conclusion(self, conclusion: str) -> Dict[str, Any]:
        """Assess the effectiveness of the conclusion."""
        words = conclusion.split()
        
        assessment = {
            'word_count': len(words),
            'effectiveness_score': 0.0,
            'strengths': [],
            'weaknesses': []
        }
        
        # Check length (ideal: 80-120 words)
        if 80 <= len(words) <= 120:
            assessment['effectiveness_score'] += 0.3
            assessment['strengths'].append("Appropriate length")
        elif len(words) < 80:
            assessment['weaknesses'].append("Conclusion may be too brief")
        else:
            assessment['weaknesses'].append("Conclusion may be too lengthy")
        
        # Check for summary elements
        summary_indicators = ['in summary', 'to conclude', 'overall', 'in conclusion']
        if any(indicator in conclusion.lower() for indicator in summary_indicators):
            assessment['effectiveness_score'] += 0.2
            assessment['strengths'].append("Contains clear summary")
        
        # Check for call to action
        cta_indicators = ['try', 'start', 'consider', 'take action', 'next step']
        if any(indicator in conclusion.lower() for indicator in cta_indicators):
            assessment['effectiveness_score'] += 0.2
            assessment['strengths'].append("Includes call to action")
        
        # Base score
        assessment['effectiveness_score'] = min(assessment['effectiveness_score'] + 0.3, 1.0)
        
        return assessment
    
    def _assess_logical_flow(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess the logical flow of the entire piece."""
        # Simple assessment based on section transitions and coherence
        flow_score = 0.0
        
        # Check if introduction leads naturally to body
        intro_words = set(draft.introduction.lower().split())
        first_body_words = set(draft.body_sections[0].lower().split()) if draft.body_sections else set()
        
        # Calculate word overlap as a proxy for flow
        overlap = len(intro_words.intersection(first_body_words))
        if overlap > 3:
            flow_score += 0.3
        
        # Check flow between body sections
        if len(draft.body_sections) > 1:
            section_overlaps = []
            for i in range(len(draft.body_sections) - 1):
                current_words = set(draft.body_sections[i].lower().split())
                next_words = set(draft.body_sections[i + 1].lower().split())
                overlap = len(current_words.intersection(next_words))
                section_overlaps.append(overlap)
            
            avg_overlap = sum(section_overlaps) / len(section_overlaps)
            if avg_overlap > 2:
                flow_score += 0.4
        
        # Check if conclusion ties back to introduction
        intro_words = set(draft.introduction.lower().split())
        conclusion_words = set(draft.conclusion.lower().split())
        conclusion_overlap = len(intro_words.intersection(conclusion_words))
        
        if conclusion_overlap > 3:
            flow_score += 0.3
        
        return {
            'flow_score': min(flow_score, 1.0),
            'coherence_indicators': {
                'intro_to_body_overlap': overlap if 'overlap' in locals() else 0,
                'body_section_overlaps': section_overlaps if 'section_overlaps' in locals() else [],
                'intro_to_conclusion_overlap': conclusion_overlap
            }
        }
    
    def _assess_section_balance(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess balance between different sections."""
        sections = [draft.introduction] + draft.body_sections + [draft.conclusion]
        word_counts = [len(section.split()) for section in sections]
        
        total_words = sum(word_counts)
        
        balance_assessment = {
            'total_words': total_words,
            'section_word_counts': word_counts,
            'balance_score': 0.0,
            'recommendations': []
        }
        
        if total_words == 0:
            return balance_assessment
        
        # Calculate ideal proportions
        intro_proportion = word_counts[0] / total_words
        body_proportion = sum(word_counts[1:-1]) / total_words if len(word_counts) > 2 else 0
        conclusion_proportion = word_counts[-1] / total_words
        
        # Ideal proportions: intro 10-15%, body 70-80%, conclusion 10-15%
        if 0.10 <= intro_proportion <= 0.15:
            balance_assessment['balance_score'] += 0.3
        else:
            balance_assessment['recommendations'].append("Adjust introduction length")
        
        if 0.70 <= body_proportion <= 0.80:
            balance_assessment['balance_score'] += 0.4
        else:
            balance_assessment['recommendations'].append("Adjust body section balance")
        
        if 0.10 <= conclusion_proportion <= 0.15:
            balance_assessment['balance_score'] += 0.3
        else:
            balance_assessment['recommendations'].append("Adjust conclusion length")
        
        return balance_assessment
    
    def _assess_narrative_coherence(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess overall narrative coherence."""
        # Check for consistent theme and messaging
        all_content = f"{draft.title} {draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        words = all_content.lower().split()
        
        # Find most common meaningful words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        meaningful_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Simple coherence metric based on word repetition
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top themes (most frequent words)
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        coherence_score = 0.0
        if top_themes:
            # If top themes appear throughout the document, it's more coherent
            theme_distribution = []
            for theme, count in top_themes:
                sections_with_theme = 0
                sections = [draft.introduction] + draft.body_sections + [draft.conclusion]
                for section in sections:
                    if theme in section.lower():
                        sections_with_theme += 1
                theme_distribution.append(sections_with_theme / len(sections))
            
            coherence_score = sum(theme_distribution) / len(theme_distribution)
        
        return {
            'coherence_score': coherence_score,
            'top_themes': top_themes,
            'theme_consistency': coherence_score > 0.6
        }
    
    def _assess_content_depth(self, draft: BlogDraft) -> Dict[str, Any]:
        """Assess the depth and comprehensiveness of content."""
        all_content = f"{draft.introduction} {' '.join(draft.body_sections)} {draft.conclusion}"
        
        # Indicators of depth
        depth_indicators = [
            'research', 'study', 'evidence', 'data', 'analysis', 'expert',
            'detailed', 'comprehensive', 'thorough', 'in-depth'
        ]
        
        depth_count = sum(1 for word in all_content.lower().split() if word in depth_indicators)
        total_words = len(all_content.split())
        
        depth_assessment = {
            'depth_indicators': depth_count,
            'total_words': total_words,
            'depth_ratio': depth_count / total_words if total_words else 0,
            'depth_score': min(depth_count / 10, 1.0),  # Normalize to 0-1
            'comprehensiveness': 'high' if depth_count > 15 else 'medium' if depth_count > 5 else 'low'
        }
        
        return depth_assessment
    
    def _calculate_section_consistency(self, section_analyses: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across sections."""
        if not section_analyses:
            return 0.0
        
        clarity_scores = [analysis['clarity_score'] for analysis in section_analyses]
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        
        # Calculate variance (lower variance = more consistent)
        variance = sum((score - avg_clarity) ** 2 for score in clarity_scores) / len(clarity_scores)
        
        # Convert variance to consistency score (0-1, higher is better)
        consistency_score = max(0.0, 1.0 - variance * 2)
        
        return consistency_score