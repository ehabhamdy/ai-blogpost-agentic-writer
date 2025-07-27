"""Core data models for the AI Blog Generation Team."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Callable, Any, Dict
from enum import Enum
import time
from datetime import datetime


class ProcessingStage(str, Enum):
    """Enum for tracking the current processing stage of blog generation."""
    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressStatus(BaseModel):
    """Model for tracking progress status throughout the workflow."""
    stage: ProcessingStage = Field(description="Current processing stage")
    message: str = Field(description="Human-readable status message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this status was set")
    agent_name: Optional[str] = Field(default=None, description="Name of the agent currently processing")
    progress_percentage: float = Field(ge=0, le=100, description="Estimated progress percentage")
    error_details: Optional[str] = Field(default=None, description="Error details if stage is FAILED")


class ProcessingMetrics(BaseModel):
    """Model for tracking processing time and performance metrics."""
    stage_name: str = Field(description="Name of the processing stage")
    start_time: datetime = Field(description="When the stage started")
    end_time: Optional[datetime] = Field(default=None, description="When the stage completed")
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds")
    agent_name: str = Field(description="Name of the agent that handled this stage")
    success: bool = Field(default=True, description="Whether the stage completed successfully")
    error_message: Optional[str] = Field(default=None, description="Error message if stage failed")
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used in this stage")
    api_calls: Optional[int] = Field(default=None, description="Number of API calls made in this stage")


class WorkflowProgress(BaseModel):
    """Model for tracking overall workflow progress and metrics."""
    topic: str = Field(description="The blog topic being processed")
    current_status: ProgressStatus = Field(description="Current progress status")
    stage_metrics: List[ProcessingMetrics] = Field(default_factory=list, description="Metrics for each completed stage")
    total_start_time: datetime = Field(default_factory=datetime.now, description="When the workflow started")
    total_end_time: Optional[datetime] = Field(default=None, description="When the workflow completed")
    total_duration_seconds: Optional[float] = Field(default=None, description="Total workflow duration")
    revision_count: int = Field(default=0, description="Number of revision cycles completed")
    final_quality_score: Optional[float] = Field(default=None, description="Final quality score if completed")


class ResearchFinding(BaseModel):
    """Individual research finding with source attribution."""
    fact: str = Field(description="The factual information found")
    source_url: str = Field(description="URL of the source")
    relevance_score: float = Field(ge=0, le=1, description="Relevance to topic")
    category: str = Field(description="Category of information (statistic, study, expert_opinion, etc.)")


class ResearchOutput(BaseModel):
    """Structured output from Research Agent."""
    topic: str = Field(description="The research topic")
    findings: List[ResearchFinding] = Field(description="List of research findings")
    summary: str = Field(description="Brief summary of key insights")
    confidence_level: float = Field(ge=0, le=1, description="Confidence in research quality")


class BlogDraft(BaseModel):
    """Blog post draft structure."""
    title: str = Field(description="Blog post title")
    introduction: str = Field(description="Opening paragraph")
    body_sections: List[str] = Field(description="Main content sections")
    conclusion: str = Field(description="Closing paragraph")
    word_count: int = Field(description="Approximate word count")


class CritiqueSeverity(str, Enum):
    """Severity levels for critique feedback."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class CritiqueFeedback(BaseModel):
    """Individual piece of feedback."""
    section: str = Field(description="Which section the feedback applies to")
    issue: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Specific improvement suggestion")
    severity: CritiqueSeverity = Field(description="Severity of the issue")


class CritiqueOutput(BaseModel):
    """Structured output from Critique Agent."""
    overall_quality: float = Field(ge=0, le=10, description="Overall quality score")
    feedback_items: List[CritiqueFeedback] = Field(description="Specific feedback items")
    approval_status: Literal["approved", "needs_revision"] = Field(description="Whether draft is approved")
    summary_feedback: str = Field(description="Overall assessment summary")


class BlogGenerationResult(BaseModel):
    """Final result from the blog generation process."""
    final_post: BlogDraft = Field(description="The final blog post")
    research_data: ResearchOutput = Field(description="Research used")
    revision_count: int = Field(description="Number of revision cycles")
    total_processing_time: float = Field(description="Total time in seconds")
    quality_score: float = Field(description="Final quality assessment")
    workflow_progress: WorkflowProgress = Field(description="Complete workflow progress and metrics")