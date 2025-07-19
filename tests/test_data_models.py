"""Unit tests for data models."""

import pytest
from pydantic import ValidationError
from src.models.data_models import (
    ResearchFinding,
    ResearchOutput,
    BlogDraft,
    CritiqueSeverity,
    CritiqueFeedback,
    CritiqueOutput,
    BlogGenerationResult,
)


class TestResearchFinding:
    """Test ResearchFinding model."""

    def test_valid_research_finding(self):
        """Test creating a valid research finding."""
        finding = ResearchFinding(
            fact="Intermittent fasting can improve metabolic health",
            source_url="https://example.com/study",
            relevance_score=0.9,
            category="study"
        )
        assert finding.fact == "Intermittent fasting can improve metabolic health"
        assert finding.source_url == "https://example.com/study"
        assert finding.relevance_score == 0.9
        assert finding.category == "study"

    def test_relevance_score_validation(self):
        """Test relevance score must be between 0 and 1."""
        # Test valid scores
        ResearchFinding(
            fact="Test fact",
            source_url="https://example.com",
            relevance_score=0.0,
            category="test"
        )
        ResearchFinding(
            fact="Test fact",
            source_url="https://example.com",
            relevance_score=1.0,
            category="test"
        )
        
        # Test invalid scores
        with pytest.raises(ValidationError):
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=-0.1,
                category="test"
            )
        
        with pytest.raises(ValidationError):
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=1.1,
                category="test"
            )

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            ResearchFinding()


class TestResearchOutput:
    """Test ResearchOutput model."""

    def test_valid_research_output(self):
        """Test creating a valid research output."""
        findings = [
            ResearchFinding(
                fact="Test fact 1",
                source_url="https://example1.com",
                relevance_score=0.8,
                category="study"
            ),
            ResearchFinding(
                fact="Test fact 2",
                source_url="https://example2.com",
                relevance_score=0.7,
                category="statistic"
            )
        ]
        
        output = ResearchOutput(
            topic="Test topic",
            findings=findings,
            summary="Test summary",
            confidence_level=0.85
        )
        
        assert output.topic == "Test topic"
        assert len(output.findings) == 2
        assert output.summary == "Test summary"
        assert output.confidence_level == 0.85

    def test_confidence_level_validation(self):
        """Test confidence level must be between 0 and 1."""
        findings = [
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=0.8,
                category="study"
            )
        ]
        
        # Test valid confidence levels
        ResearchOutput(
            topic="Test",
            findings=findings,
            summary="Test",
            confidence_level=0.0
        )
        ResearchOutput(
            topic="Test",
            findings=findings,
            summary="Test",
            confidence_level=1.0
        )
        
        # Test invalid confidence levels
        with pytest.raises(ValidationError):
            ResearchOutput(
                topic="Test",
                findings=findings,
                summary="Test",
                confidence_level=-0.1
            )
        
        with pytest.raises(ValidationError):
            ResearchOutput(
                topic="Test",
                findings=findings,
                summary="Test",
                confidence_level=1.1
            )

    def test_empty_findings_list(self):
        """Test that empty findings list is allowed."""
        output = ResearchOutput(
            topic="Test topic",
            findings=[],
            summary="No findings",
            confidence_level=0.1
        )
        assert len(output.findings) == 0


class TestBlogDraft:
    """Test BlogDraft model."""

    def test_valid_blog_draft(self):
        """Test creating a valid blog draft."""
        draft = BlogDraft(
            title="Test Blog Post",
            introduction="This is the introduction.",
            body_sections=["Section 1 content", "Section 2 content"],
            conclusion="This is the conclusion.",
            word_count=150
        )
        
        assert draft.title == "Test Blog Post"
        assert draft.introduction == "This is the introduction."
        assert len(draft.body_sections) == 2
        assert draft.conclusion == "This is the conclusion."
        assert draft.word_count == 150

    def test_empty_body_sections(self):
        """Test that empty body sections list is allowed."""
        draft = BlogDraft(
            title="Test",
            introduction="Intro",
            body_sections=[],
            conclusion="Conclusion",
            word_count=50
        )
        assert len(draft.body_sections) == 0

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            BlogDraft()


class TestCritiqueSeverity:
    """Test CritiqueSeverity enum."""

    def test_valid_severity_values(self):
        """Test that all severity values are valid."""
        assert CritiqueSeverity.MINOR == "minor"
        assert CritiqueSeverity.MODERATE == "moderate"
        assert CritiqueSeverity.MAJOR == "major"


class TestCritiqueFeedback:
    """Test CritiqueFeedback model."""

    def test_valid_critique_feedback(self):
        """Test creating valid critique feedback."""
        feedback = CritiqueFeedback(
            section="introduction",
            issue="Unclear opening statement",
            suggestion="Add a hook to engage readers",
            severity=CritiqueSeverity.MODERATE
        )
        
        assert feedback.section == "introduction"
        assert feedback.issue == "Unclear opening statement"
        assert feedback.suggestion == "Add a hook to engage readers"
        assert feedback.severity == CritiqueSeverity.MODERATE

    def test_severity_validation(self):
        """Test that severity must be a valid enum value."""
        # Valid severity
        CritiqueFeedback(
            section="test",
            issue="test issue",
            suggestion="test suggestion",
            severity=CritiqueSeverity.MINOR
        )
        
        # Invalid severity should be caught by Pydantic
        with pytest.raises(ValidationError):
            CritiqueFeedback(
                section="test",
                issue="test issue",
                suggestion="test suggestion",
                severity="invalid"
            )


class TestCritiqueOutput:
    """Test CritiqueOutput model."""

    def test_valid_critique_output(self):
        """Test creating valid critique output."""
        feedback_items = [
            CritiqueFeedback(
                section="introduction",
                issue="Test issue",
                suggestion="Test suggestion",
                severity=CritiqueSeverity.MINOR
            )
        ]
        
        output = CritiqueOutput(
            overall_quality=8.5,
            feedback_items=feedback_items,
            approval_status="approved",
            summary_feedback="Good overall quality"
        )
        
        assert output.overall_quality == 8.5
        assert len(output.feedback_items) == 1
        assert output.approval_status == "approved"
        assert output.summary_feedback == "Good overall quality"

    def test_quality_score_validation(self):
        """Test quality score must be between 0 and 10."""
        feedback_items = []
        
        # Test valid scores
        CritiqueOutput(
            overall_quality=0.0,
            feedback_items=feedback_items,
            approval_status="needs_revision",
            summary_feedback="Test"
        )
        CritiqueOutput(
            overall_quality=10.0,
            feedback_items=feedback_items,
            approval_status="approved",
            summary_feedback="Test"
        )
        
        # Test invalid scores
        with pytest.raises(ValidationError):
            CritiqueOutput(
                overall_quality=-0.1,
                feedback_items=feedback_items,
                approval_status="needs_revision",
                summary_feedback="Test"
            )
        
        with pytest.raises(ValidationError):
            CritiqueOutput(
                overall_quality=10.1,
                feedback_items=feedback_items,
                approval_status="approved",
                summary_feedback="Test"
            )

    def test_approval_status_validation(self):
        """Test approval status must be literal value."""
        feedback_items = []
        
        # Valid statuses
        CritiqueOutput(
            overall_quality=8.0,
            feedback_items=feedback_items,
            approval_status="approved",
            summary_feedback="Test"
        )
        CritiqueOutput(
            overall_quality=6.0,
            feedback_items=feedback_items,
            approval_status="needs_revision",
            summary_feedback="Test"
        )
        
        # Invalid status
        with pytest.raises(ValidationError):
            CritiqueOutput(
                overall_quality=8.0,
                feedback_items=feedback_items,
                approval_status="invalid_status",
                summary_feedback="Test"
            )


class TestBlogGenerationResult:
    """Test BlogGenerationResult model."""

    def test_valid_blog_generation_result(self):
        """Test creating valid blog generation result."""
        # Create test data
        findings = [
            ResearchFinding(
                fact="Test fact",
                source_url="https://example.com",
                relevance_score=0.8,
                category="study"
            )
        ]
        
        research_data = ResearchOutput(
            topic="Test topic",
            findings=findings,
            summary="Test summary",
            confidence_level=0.8
        )
        
        final_post = BlogDraft(
            title="Test Post",
            introduction="Test intro",
            body_sections=["Test body"],
            conclusion="Test conclusion",
            word_count=100
        )
        
        result = BlogGenerationResult(
            final_post=final_post,
            research_data=research_data,
            revision_count=2,
            total_processing_time=45.5,
            quality_score=8.7
        )
        
        assert result.final_post.title == "Test Post"
        assert result.research_data.topic == "Test topic"
        assert result.revision_count == 2
        assert result.total_processing_time == 45.5
        assert result.quality_score == 8.7

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            BlogGenerationResult()