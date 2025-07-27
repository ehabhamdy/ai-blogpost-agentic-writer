"""Tests for progress tracking and status reporting functionality."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List

from src.models.data_models import (
    ProcessingStage,
    ProgressStatus,
    ProcessingMetrics,
    WorkflowProgress
)
from src.utils.progress_tracker import ProgressTracker, create_structured_logger


class TestProgressTracker:
    """Test cases for the ProgressTracker class."""
    
    def test_initialization(self):
        """Test that ProgressTracker initializes correctly."""
        topic = "Test Topic"
        callback = Mock()
        
        tracker = ProgressTracker(topic, callback)
        
        assert tracker.topic == topic
        assert tracker.status_callback == callback
        assert tracker.workflow_progress.topic == topic
        assert tracker.workflow_progress.current_status.stage == ProcessingStage.INITIALIZING
        assert tracker.workflow_progress.current_status.progress_percentage == 0.0
    
    def test_update_status(self):
        """Test status updates work correctly."""
        tracker = ProgressTracker("Test Topic")
        callback = Mock()
        tracker.status_callback = callback
        
        # Update status
        tracker.update_status(
            ProcessingStage.RESEARCHING,
            "Starting research",
            agent_name="ResearchAgent",
            progress_percentage=25.0
        )
        
        # Check status was updated
        status = tracker.workflow_progress.current_status
        assert status.stage == ProcessingStage.RESEARCHING
        assert status.message == "Starting research"
        assert status.agent_name == "ResearchAgent"
        assert status.progress_percentage == 25.0
        assert status.error_details is None
        
        # Check callback was called
        callback.assert_called_once_with(status)
    
    def test_update_status_with_error(self):
        """Test status updates with error details."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.update_status(
            ProcessingStage.FAILED,
            "Process failed",
            error_details="Network timeout"
        )
        
        status = tracker.workflow_progress.current_status
        assert status.stage == ProcessingStage.FAILED
        assert status.error_details == "Network timeout"
    
    def test_start_stage(self):
        """Test starting a new processing stage."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.start_stage(
            ProcessingStage.WRITING,
            "WritingAgent",
            "Creating blog draft"
        )
        
        # Check status was updated
        status = tracker.workflow_progress.current_status
        assert status.stage == ProcessingStage.WRITING
        assert status.agent_name == "WritingAgent"
        assert status.message == "Creating blog draft"
        
        # Check internal tracking
        assert tracker._current_stage_start is not None
        assert tracker._current_stage_metrics is not None
        assert tracker._current_stage_metrics.stage_name == ProcessingStage.WRITING.value
        assert tracker._current_stage_metrics.agent_name == "WritingAgent"
    
    def test_complete_stage(self):
        """Test completing a processing stage."""
        tracker = ProgressTracker("Test Topic")
        
        # Start a stage
        tracker.start_stage(ProcessingStage.RESEARCHING, "ResearchAgent")
        
        # Wait a bit to ensure duration > 0
        time.sleep(0.01)
        
        # Complete the stage
        tracker.complete_stage(
            success=True,
            tokens_used=100,
            api_calls=2
        )
        
        # Check metrics were recorded
        assert len(tracker.workflow_progress.stage_metrics) == 1
        metrics = tracker.workflow_progress.stage_metrics[0]
        assert metrics.stage_name == ProcessingStage.RESEARCHING.value
        assert metrics.agent_name == "ResearchAgent"
        assert metrics.success is True
        assert metrics.tokens_used == 100
        assert metrics.api_calls == 2
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds > 0
        
        # Check internal state was cleared
        assert tracker._current_stage_metrics is None
        assert tracker._current_stage_start is None
    
    def test_complete_stage_with_error(self):
        """Test completing a stage with an error."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.start_stage(ProcessingStage.WRITING, "WritingAgent")
        tracker.complete_stage(
            success=False,
            error_message="API rate limit exceeded"
        )
        
        metrics = tracker.workflow_progress.stage_metrics[0]
        assert metrics.success is False
        assert metrics.error_message == "API rate limit exceeded"
    
    def test_complete_workflow_success(self):
        """Test completing the workflow successfully."""
        tracker = ProgressTracker("Test Topic")
        
        # Simulate some stages
        tracker.start_stage(ProcessingStage.RESEARCHING, "ResearchAgent")
        tracker.complete_stage(success=True)
        
        tracker.start_stage(ProcessingStage.WRITING, "WritingAgent")
        tracker.complete_stage(success=True)
        
        # Complete workflow
        final_progress = tracker.complete_workflow(
            final_quality_score=8.5,
            revision_count=2
        )
        
        assert final_progress.final_quality_score == 8.5
        assert final_progress.revision_count == 2
        assert final_progress.total_end_time is not None
        assert final_progress.total_duration_seconds is not None
        assert final_progress.current_status.stage == ProcessingStage.COMPLETED
        assert final_progress.current_status.progress_percentage == 100.0
    
    def test_fail_workflow(self):
        """Test failing the workflow."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.start_stage(ProcessingStage.RESEARCHING, "ResearchAgent")
        
        final_progress = tracker.fail_workflow(
            "Network connection failed",
            "ResearchAgent"
        )
        
        assert final_progress.current_status.stage == ProcessingStage.FAILED
        assert final_progress.current_status.error_details == "Network connection failed"
        assert final_progress.current_status.agent_name == "ResearchAgent"
        assert final_progress.total_end_time is not None
        
        # Check that current stage was completed with failure
        assert len(final_progress.stage_metrics) == 1
        assert final_progress.stage_metrics[0].success is False
    
    def test_progress_percentage_calculation(self):
        """Test automatic progress percentage calculation."""
        tracker = ProgressTracker("Test Topic")
        
        test_cases = [
            (ProcessingStage.INITIALIZING, 5.0),
            (ProcessingStage.RESEARCHING, 25.0),
            (ProcessingStage.WRITING, 50.0),
            (ProcessingStage.CRITIQUING, 70.0),
            (ProcessingStage.REVISING, 85.0),
            (ProcessingStage.COMPLETED, 100.0),
            (ProcessingStage.FAILED, 0.0)
        ]
        
        for stage, expected_percentage in test_cases:
            tracker.update_status(stage, f"Testing {stage.value}")
            assert tracker.workflow_progress.current_status.progress_percentage == expected_percentage
    
    def test_revision_penalty_in_progress(self):
        """Test that revision cycles reduce progress percentage."""
        tracker = ProgressTracker("Test Topic")
        
        # Simulate multiple revision cycles
        tracker.workflow_progress.revision_count = 2
        
        tracker.update_status(ProcessingStage.REVISING, "Revising draft")
        
        # Should be less than base 85% due to revision penalty
        assert tracker.workflow_progress.current_status.progress_percentage < 85.0
        assert tracker.workflow_progress.current_status.progress_percentage >= 70.0  # Minimum
    
    def test_callback_error_handling(self):
        """Test that callback errors don't break the tracker."""
        def failing_callback(status):
            raise Exception("Callback failed")
        
        tracker = ProgressTracker("Test Topic", failing_callback)
        
        # This should not raise an exception
        tracker.update_status(ProcessingStage.RESEARCHING, "Testing callback error")
        
        # Status should still be updated
        assert tracker.workflow_progress.current_status.stage == ProcessingStage.RESEARCHING
    
    def test_get_current_progress(self):
        """Test getting current progress."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.update_status(ProcessingStage.WRITING, "Creating draft")
        
        progress = tracker.get_current_progress()
        assert isinstance(progress, WorkflowProgress)
        assert progress.current_status.stage == ProcessingStage.WRITING
    
    @patch('src.utils.progress_tracker.logger')
    def test_structured_logging(self, mock_logger):
        """Test that structured logging is used correctly."""
        tracker = ProgressTracker("Test Topic")
        
        tracker.update_status(
            ProcessingStage.RESEARCHING,
            "Starting research",
            agent_name="ResearchAgent"
        )
        
        # Check that logger.info was called with structured data
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args
        
        # Check the extra parameter contains structured data
        assert 'extra' in call_args.kwargs
        extra = call_args.kwargs['extra']
        assert extra['topic'] == "Test Topic"
        assert extra['stage'] == ProcessingStage.RESEARCHING.value
        assert extra['agent_name'] == "ResearchAgent"
        assert extra['status_message'] == "Starting research"
        assert 'timestamp' in extra


class TestStructuredLogger:
    """Test cases for structured logging functionality."""
    
    def test_create_structured_logger(self):
        """Test creating a structured logger."""
        logger = create_structured_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) > 0
    
    def test_logger_singleton_behavior(self):
        """Test that multiple calls return the same logger instance."""
        logger1 = create_structured_logger("same_name")
        logger2 = create_structured_logger("same_name")
        
        assert logger1 is logger2
        # Should not add duplicate handlers
        assert len(logger1.handlers) == len(logger2.handlers)


class TestProgressStatusModel:
    """Test cases for ProgressStatus model validation."""
    
    def test_valid_progress_status(self):
        """Test creating a valid ProgressStatus."""
        status = ProgressStatus(
            stage=ProcessingStage.WRITING,
            message="Creating blog draft",
            agent_name="WritingAgent",
            progress_percentage=50.0
        )
        
        assert status.stage == ProcessingStage.WRITING
        assert status.message == "Creating blog draft"
        assert status.agent_name == "WritingAgent"
        assert status.progress_percentage == 50.0
        assert status.error_details is None
        assert isinstance(status.timestamp, datetime)
    
    def test_progress_percentage_validation(self):
        """Test progress percentage validation."""
        # Valid percentages
        ProgressStatus(
            stage=ProcessingStage.WRITING,
            message="Test",
            progress_percentage=0.0
        )
        
        ProgressStatus(
            stage=ProcessingStage.WRITING,
            message="Test",
            progress_percentage=100.0
        )
        
        # Invalid percentages should raise validation error
        with pytest.raises(ValueError):
            ProgressStatus(
                stage=ProcessingStage.WRITING,
                message="Test",
                progress_percentage=-1.0
            )
        
        with pytest.raises(ValueError):
            ProgressStatus(
                stage=ProcessingStage.WRITING,
                message="Test",
                progress_percentage=101.0
            )


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics model."""
    
    def test_valid_processing_metrics(self):
        """Test creating valid ProcessingMetrics."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)
        
        metrics = ProcessingMetrics(
            stage_name="researching",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=5.0,
            agent_name="ResearchAgent",
            success=True,
            tokens_used=100,
            api_calls=2
        )
        
        assert metrics.stage_name == "researching"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration_seconds == 5.0
        assert metrics.agent_name == "ResearchAgent"
        assert metrics.success is True
        assert metrics.tokens_used == 100
        assert metrics.api_calls == 2
        assert metrics.error_message is None
    
    def test_metrics_with_error(self):
        """Test ProcessingMetrics with error information."""
        metrics = ProcessingMetrics(
            stage_name="writing",
            start_time=datetime.now(),
            agent_name="WritingAgent",
            success=False,
            error_message="API timeout"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "API timeout"


class TestWorkflowProgress:
    """Test cases for WorkflowProgress model."""
    
    def test_workflow_progress_initialization(self):
        """Test WorkflowProgress initialization."""
        current_status = ProgressStatus(
            stage=ProcessingStage.INITIALIZING,
            message="Starting workflow",
            progress_percentage=0.0
        )
        
        progress = WorkflowProgress(
            topic="Test Topic",
            current_status=current_status
        )
        
        assert progress.topic == "Test Topic"
        assert progress.current_status == current_status
        assert progress.stage_metrics == []
        assert progress.revision_count == 0
        assert isinstance(progress.total_start_time, datetime)
        assert progress.total_end_time is None
        assert progress.final_quality_score is None
    
    def test_workflow_progress_with_metrics(self):
        """Test WorkflowProgress with stage metrics."""
        current_status = ProgressStatus(
            stage=ProcessingStage.COMPLETED,
            message="Workflow completed",
            progress_percentage=100.0
        )
        
        metrics = [
            ProcessingMetrics(
                stage_name="researching",
                start_time=datetime.now(),
                agent_name="ResearchAgent"
            ),
            ProcessingMetrics(
                stage_name="writing",
                start_time=datetime.now(),
                agent_name="WritingAgent"
            )
        ]
        
        progress = WorkflowProgress(
            topic="Test Topic",
            current_status=current_status,
            stage_metrics=metrics,
            revision_count=1,
            final_quality_score=8.5
        )
        
        assert len(progress.stage_metrics) == 2
        assert progress.revision_count == 1
        assert progress.final_quality_score == 8.5


if __name__ == "__main__":
    pytest.main([__file__])