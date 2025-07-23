"""Orchestrator Agent for coordinating the multi-agent blog generation workflow."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from ..models.data_models import (
    BlogGenerationResult,
    BlogDraft,
    ResearchOutput,
    CritiqueOutput,
    CritiqueSeverity
)
from ..utils.dependencies import SharedDependencies

from ..utils.exceptions import (
    OrchestrationError, 
    ResearchError, 
    WritingError, 
    CritiqueError,
    ValidationError
)
from .research_agent import ResearchAgent
from .writing_agent import WritingAgent
from .critique_agent import CritiqueAgent
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationContext:
    """Context class that holds workflow state for the Orchestrator Agent."""
    topic: str
    research_agent: ResearchAgent
    writing_agent: WritingAgent
    critique_agent: CritiqueAgent
    start_time: float
    usage_tracking: Dict[str, Any]
    shared_deps: SharedDependencies  # Store the actual shared dependencies


class OrchestratorAgent:
    """Agent responsible for coordinating the complete blog generation workflow."""
    
    def __init__(self, model: Model):
        """Initialize the Orchestrator Agent with a model."""
        self.agent = Agent(
            model=model,
            output_type=BlogGenerationResult,
            system_prompt="""You are the orchestrator of a multi-agent blog generation system. 
            Your role is to coordinate between Research, Writing, and Critique agents to produce 
            high-quality blog posts through an iterative workflow.
            
            Your responsibilities:
            - Delegate research tasks to gather comprehensive information
            - Coordinate writing tasks to create well-structured drafts
            - Manage critique and revision cycles for quality improvement
            - Make decisions about when to continue or stop revision cycles
            - Track usage and performance metrics throughout the process
            - Ensure the final output meets quality standards
            
            Use the available tools to delegate work to specialized agents and make 
            informed decisions about the workflow progression. Always prioritize quality 
            while managing iteration limits and processing efficiency."""
        )
        
        # Register tools
        self.agent.tool(self.delegate_research)
        self.agent.tool(self.delegate_writing)
        self.agent.tool(self.delegate_critique)
        self.agent.tool(self.make_revision_decision)
    
    async def generate_blog_post(
        self,
        topic: str,
        research_agent: ResearchAgent,
        writing_agent: WritingAgent,
        critique_agent: CritiqueAgent,
        deps: SharedDependencies
    ) -> BlogGenerationResult:
        """Generate a complete blog post through the multi-agent workflow."""
        # Validate inputs
        if not topic or not topic.strip():
            raise ValidationError(
                "Topic cannot be empty",
                field_name="topic",
                invalid_value=topic
            )
        
        if not all([research_agent, writing_agent, critique_agent]):
            raise ValidationError(
                "All agents must be provided",
                field_name="agents",
                invalid_value="missing_agents"
            )
        
        start_time = time.time()
        intermediate_results = {}  # Store intermediate results for graceful degradation
        
        try:
            # Create orchestration context
            context = OrchestrationContext(
                topic=topic,
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                start_time=start_time,
                usage_tracking={
                    'research_calls': 0,
                    'writing_calls': 0,
                    'critique_calls': 0,
                    'total_tokens': 0,
                    'api_calls': 0,
                    'revision_cycles': 0
                },
                shared_deps=deps  # Store the actual shared dependencies
            )
            
            result = await self.agent.run(
                f"""Orchestrate the complete blog generation workflow for the topic: "{topic}"
                
                Follow this iterative revision workflow:
                1. Use delegate_research to gather comprehensive research on the topic
                2. Use delegate_writing to create an initial blog draft from the research
                3. Use delegate_critique to review the draft and provide feedback
                4. Use make_revision_decision to determine if revisions are needed
                5. If revisions are needed, use delegate_writing again with feedback and original draft
                6. Repeat steps 3-5 until quality is acceptable or max iterations ({deps.max_iterations}) reached
                7. Return the final BlogGenerationResult with all metadata
                
                Quality threshold: {deps.quality_threshold}/10
                Maximum iterations: {deps.max_iterations}
                
                Ensure high-quality output while managing processing efficiency through iterative revision.""",
                deps=context
            )
            
            return result.output
            
        except Exception as e:
            logger.error(f"Blog generation failed for topic '{topic}': {e}")
            
            # Use ModelRetry for retryable errors
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                raise ModelRetry(f"Orchestration failed due to API issues: {e}")
            
            # Handle agent-specific errors
            if isinstance(e, (ResearchError, WritingError, CritiqueError)):
                # Re-raise agent-specific errors with orchestration context
                raise OrchestrationError(
                    f"Blog generation failed during {e.__class__.__name__.replace('Error', '').lower()} phase: {e}",
                    workflow_stage=e.__class__.__name__.replace('Error', '').lower(),
                    original_error=e
                )
            elif isinstance(e, (ValidationError, OrchestrationError)):
                raise e
            else:
                # Try to preserve intermediate results if available
                if intermediate_results:
                    logger.warning(f"Returning partial results due to error: {e}")
                    # Could implement partial result recovery here
                
                raise OrchestrationError(
                    f"Blog generation failed for topic '{topic}': {e}",
                    workflow_stage="orchestration",
                    original_error=e
                )
    
    async def delegate_research(
        self,
        ctx: RunContext[OrchestrationContext],
        topic: str
    ) -> ResearchOutput:
        """Delegate research task to Research Agent and handle usage tracking.
        
        Args:
            topic: The research topic to investigate
            
        Returns:
            ResearchOutput with comprehensive research findings
        """
        try:
            # Update usage tracking
            ctx.deps.usage_tracking['research_calls'] += 1
            ctx.deps.usage_tracking['api_calls'] += 1
            
            # Use the actual shared dependencies from the orchestration context
            # This ensures the research agent gets the real Tavily client
            shared_deps = ctx.deps.shared_deps
            
            # Delegate to research agent
            research_result = await ctx.deps.research_agent.research_topic(
                topic=topic,
                deps=shared_deps
            )
            
            # Track token usage (simplified tracking)
            ctx.deps.usage_tracking['total_tokens'] += len(research_result.summary.split()) * 2
            
            return research_result
            
        except Exception as e:
            logger.error(f"Research delegation failed for topic '{topic}': {e}")
            
            if "rate limit" in str(e).lower():
                raise ModelRetry(f"Research delegation failed due to rate limiting: {e}")
            elif "timeout" in str(e).lower():
                raise ModelRetry(f"Research delegation timed out: {e}")
            elif isinstance(e, ResearchError):
                # Re-raise research errors as-is
                raise e
            else:
                # Return minimal research result for graceful degradation
                logger.warning(f"Returning minimal research result for topic '{topic}' due to error: {e}")
                return ResearchOutput(
                    topic=topic,
                    findings=[],
                    summary=f"Limited research available for {topic} due to technical issues: {str(e)[:100]}",
                    confidence_level=0.1
                )
    
    async def delegate_writing(
        self,
        ctx: RunContext[OrchestrationContext],
        research_data: ResearchOutput,
        feedback: Optional[str] = None,
        original_draft: Optional[BlogDraft] = None
    ) -> BlogDraft:
        """Delegate writing task to Writing Agent with research data.
        
        Args:
            research_data: Research findings to base the writing on
            feedback: Optional feedback for revisions
            original_draft: Optional original draft for revision
            
        Returns:
            BlogDraft with structured blog content
        """
        try:
            # Update usage tracking
            ctx.deps.usage_tracking['writing_calls'] += 1
            ctx.deps.usage_tracking['api_calls'] += 1
            
            # Use the actual shared dependencies (writing agent doesn't need Tavily client)
            shared_deps = ctx.deps.shared_deps
            
            # Delegate to writing agent
            if feedback and original_draft:
                # This is a revision
                writing_result = await ctx.deps.writing_agent.revise_blog_draft(
                    original_draft=original_draft,
                    feedback=feedback,
                    research_data=research_data,
                    deps=shared_deps
                )
            else:
                # This is initial draft creation
                writing_result = await ctx.deps.writing_agent.create_blog_draft(
                    topic=research_data.topic,
                    research_data=research_data,
                    deps=shared_deps
                )
            
            # Track token usage (simplified tracking)
            total_content = f"{writing_result.title} {writing_result.introduction} {' '.join(writing_result.body_sections)} {writing_result.conclusion}"
            ctx.deps.usage_tracking['total_tokens'] += len(total_content.split()) * 2
            
            return writing_result
            
        except Exception as e:
            logger.error(f"Writing delegation failed for topic '{research_data.topic}': {e}")
            
            if "rate limit" in str(e).lower():
                raise ModelRetry(f"Writing delegation failed due to rate limiting: {e}")
            elif "timeout" in str(e).lower():
                raise ModelRetry(f"Writing delegation timed out: {e}")
            elif isinstance(e, WritingError):
                # Re-raise writing errors as-is
                raise e
            else:
                # Return minimal draft for graceful degradation
                logger.warning(f"Returning minimal draft for topic '{research_data.topic}' due to error: {e}")
                return BlogDraft(
                    title=f"Blog Post: {research_data.topic}",
                    introduction=f"This article explores {research_data.topic}. {str(e)[:100]}",
                    body_sections=[f"Content about {research_data.topic} will be added here due to technical limitations."],
                    conclusion=f"In conclusion, {research_data.topic} is an important topic that requires further exploration.",
                    word_count=50
                )
    
    async def delegate_critique(
        self,
        ctx: RunContext[OrchestrationContext],
        blog_draft: BlogDraft,
        research_data: ResearchOutput
    ) -> CritiqueOutput:
        """Delegate critique task to Critique Agent with draft and research.
        
        Args:
            blog_draft: The blog draft to critique
            research_data: Original research data for fact-checking
            
        Returns:
            CritiqueOutput with detailed feedback and approval status
        """
        try:
            # Update usage tracking
            ctx.deps.usage_tracking['critique_calls'] += 1
            ctx.deps.usage_tracking['api_calls'] += 1
            
            # Use the actual shared dependencies (critique agent doesn't need Tavily client)
            shared_deps = ctx.deps.shared_deps
            
            # Delegate to critique agent
            critique_result = await ctx.deps.critique_agent.critique_blog_draft(
                blog_draft=blog_draft,
                research_data=research_data,
                deps=shared_deps
            )
            
            # Track token usage (simplified tracking)
            feedback_text = " ".join([f.issue + " " + f.suggestion for f in critique_result.feedback_items])
            ctx.deps.usage_tracking['total_tokens'] += len(feedback_text.split()) * 2
            
            return critique_result
            
        except Exception as e:
            logger.error(f"Critique delegation failed for draft '{blog_draft.title}': {e}")
            
            if "rate limit" in str(e).lower():
                raise ModelRetry(f"Critique delegation failed due to rate limiting: {e}")
            elif "timeout" in str(e).lower():
                raise ModelRetry(f"Critique delegation timed out: {e}")
            elif isinstance(e, CritiqueError):
                # Re-raise critique errors as-is
                raise e
            else:
                # Return minimal critique for graceful degradation
                logger.warning(f"Returning minimal critique for draft '{blog_draft.title}' due to error: {e}")
                return CritiqueOutput(
                    overall_quality=6.0,
                    feedback_items=[],
                    approval_status="approved",
                    summary_feedback=f"Unable to provide detailed critique due to technical issues: {str(e)[:100]}"
                )
    
    async def make_revision_decision(
        self,
        ctx: RunContext[OrchestrationContext],
        critique_output: CritiqueOutput,
        current_iteration: int,
        max_iterations: int,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """Determine if revision cycles should continue based on critique feedback.
        
        Args:
            critique_output: The critique results to evaluate
            current_iteration: Current iteration number (1-based)
            max_iterations: Maximum allowed iterations
            quality_threshold: Minimum quality score for approval
            
        Returns:
            Dictionary with revision decision and reasoning
        """
        decision = {
            'should_revise': False,
            'reasoning': '',
            'quality_score': critique_output.overall_quality,
            'approval_status': critique_output.approval_status,
            'iteration': current_iteration,
            'max_iterations': max_iterations
        }
        
        # Check if we've reached maximum iterations
        if current_iteration >= max_iterations:
            decision['reasoning'] = f"Maximum iterations ({max_iterations}) reached. Stopping revision cycle."
            return decision
        
        # Check if critique explicitly approved the draft
        if critique_output.approval_status == "approved":
            decision['reasoning'] = "Draft approved by critique agent."
            return decision
        
        # Check if quality score meets threshold
        if critique_output.overall_quality >= quality_threshold:
            decision['reasoning'] = f"Quality score ({critique_output.overall_quality:.1f}) meets threshold ({quality_threshold})."
            return decision
        
        # Check severity of feedback items
        major_issues = [f for f in critique_output.feedback_items if f.severity == CritiqueSeverity.MAJOR]
        moderate_issues = [f for f in critique_output.feedback_items if f.severity == CritiqueSeverity.MODERATE]
        
        # Decide based on issue severity and quality score
        if major_issues and critique_output.overall_quality < quality_threshold - 1.0:
            decision['should_revise'] = True
            decision['reasoning'] = f"Major issues found ({len(major_issues)}) and quality score ({critique_output.overall_quality:.1f}) significantly below threshold."
        elif moderate_issues and critique_output.overall_quality < quality_threshold:
            decision['should_revise'] = True
            decision['reasoning'] = f"Moderate issues found ({len(moderate_issues)}) and quality score ({critique_output.overall_quality:.1f}) below threshold."
        elif critique_output.overall_quality < quality_threshold - 2.0:
            decision['should_revise'] = True
            decision['reasoning'] = f"Quality score ({critique_output.overall_quality:.1f}) significantly below threshold ({quality_threshold})."
        else:
            decision['reasoning'] = f"Quality acceptable ({critique_output.overall_quality:.1f}) despite minor issues."
        
        return decision
    
    def _format_feedback_for_revision(self, critique_output: CritiqueOutput) -> str:
        """Format critique feedback for the writing agent revision."""
        feedback_parts = [critique_output.summary_feedback]
        
        # Group feedback by severity
        major_feedback = [f for f in critique_output.feedback_items if f.severity == CritiqueSeverity.MAJOR]
        moderate_feedback = [f for f in critique_output.feedback_items if f.severity == CritiqueSeverity.MODERATE]
        minor_feedback = [f for f in critique_output.feedback_items if f.severity == CritiqueSeverity.MINOR]
        
        if major_feedback:
            feedback_parts.append("\nCRITICAL ISSUES TO ADDRESS:")
            for feedback in major_feedback:
                feedback_parts.append(f"- {feedback.section}: {feedback.issue} -> {feedback.suggestion}")
        
        if moderate_feedback:
            feedback_parts.append("\nIMPORTANT IMPROVEMENTS:")
            for feedback in moderate_feedback:
                feedback_parts.append(f"- {feedback.section}: {feedback.issue} -> {feedback.suggestion}")
        
        if minor_feedback:
            feedback_parts.append("\nMINOR ENHANCEMENTS:")
            for feedback in minor_feedback[:3]:  # Limit minor feedback to top 3
                feedback_parts.append(f"- {feedback.section}: {feedback.issue} -> {feedback.suggestion}")
        
        return "\n".join(feedback_parts)
    
    def _calculate_final_metrics(
        self,
        start_time: float,
        usage_tracking: Dict[str, Any],
        final_quality: float
    ) -> Dict[str, Any]:
        """Calculate final processing metrics."""
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'total_processing_time': total_time,
            'quality_score': final_quality,
            'usage_stats': usage_tracking.copy(),
            'efficiency_score': final_quality / max(total_time / 60, 1)  # Quality per minute
        }