"""Simple Critique Agent for providing basic feedback on blog drafts."""

import logging
from typing import List, Dict, Any
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model

from ..models.data_models import (
    CritiqueOutput, 
    BlogDraft, 
    ResearchOutput
)
from ..utils.dependencies import SharedDependencies
from ..utils.exceptions import CritiqueError, ValidationError

logger = logging.getLogger(__name__)


class CritiqueAgent:
    """Simple agent for basic blog draft critique."""
    
    def __init__(self, model: Model):
        """Initialize the Critique Agent with a model."""
        self.agent = Agent(
            model=model,
            output_type=CritiqueOutput,
            system_prompt="""You are an editor providing simple, actionable feedback on blog posts.
            
            Focus on:
            - Overall readability and clarity
            - Basic structure (intro, body, conclusion)
            - Whether content matches the research provided
            - Simple grammar and flow issues
            
            Keep feedback concise and actionable. Rate overall quality from 1-10."""
        )
    
    async def critique_blog_draft(
        self, 
        blog_draft: BlogDraft, 
        research_data: ResearchOutput,
        deps: SharedDependencies
    ) -> CritiqueOutput:
        """Provide simple critique of a blog draft."""
        # Basic validation
        if not blog_draft or not blog_draft.title:
            raise ValidationError("Blog draft is missing or incomplete", "blog_draft", blog_draft)
        
        if not research_data:
            raise ValidationError("Research data is required", "research_data", None)
        
        try:
            # Simple prompt with key information
            prompt = f"""Review this blog post and provide feedback:

Title: {blog_draft.title}
Introduction: {blog_draft.introduction[:200]}...
Body sections: {len(blog_draft.body_sections)} sections
Conclusion: {blog_draft.conclusion[:200]}...

Research topic: {research_data.topic}
Available research findings: {len(research_data.findings)}

Provide:
1. Overall quality score (1-10)
2. 2-3 key feedback points
3. Whether it meets quality threshold of {deps.quality_threshold}/10"""

            result = await self.agent.run(prompt)
            return result.output
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Critique failed: {e}")
            raise CritiqueError(f"Failed to critique '{blog_draft.title}': {e}", blog_draft.title, e)

