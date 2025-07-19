"""Core data models for the AI Blog Generation Team."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


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