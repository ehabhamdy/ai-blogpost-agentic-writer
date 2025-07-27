"""Progress tracking utilities for the AI Blog Generation Team."""

import logging
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from ..models.data_models import (
    ProcessingStage,
    ProgressStatus,
    ProcessingMetrics,
    WorkflowProgress
)

# Set up structured logging
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Utility class for tracking progress and metrics throughout the workflow."""
    
    def __init__(self, topic: str, status_callback: Optional[Callable[[ProgressStatus], None]] = None):
        """Initialize the progress tracker.
        
        Args:
            topic: The blog topic being processed
            status_callback: Optional callback function to receive status updates
        """
        self.topic = topic
        self.status_callback = status_callback
        self.workflow_progress = WorkflowProgress(
            topic=topic,
            current_status=ProgressStatus(
                stage=ProcessingStage.INITIALIZING,
                message="Initializing blog generation workflow",
                progress_percentage=0.0
            )
        )
        self._current_stage_start: Optional[datetime] = None
        self._current_stage_metrics: Optional[ProcessingMetrics] = None
        
        # Log initialization
        logger.info(
            "Progress tracker initialized",
            extra={
                "topic": topic,
                "stage": ProcessingStage.INITIALIZING.value,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def update_status(
        self,
        stage: ProcessingStage,
        message: str,
        agent_name: Optional[str] = None,
        progress_percentage: Optional[float] = None,
        error_details: Optional[str] = None
    ) -> None:
        """Update the current progress status.
        
        Args:
            stage: Current processing stage
            message: Human-readable status message
            agent_name: Name of the agent currently processing
            progress_percentage: Estimated progress percentage (0-100)
            error_details: Error details if stage is FAILED
        """
        # Calculate progress percentage if not provided
        if progress_percentage is None:
            progress_percentage = self._calculate_progress_percentage(stage)
        
        # Create new status
        new_status = ProgressStatus(
            stage=stage,
            message=message,
            agent_name=agent_name,
            progress_percentage=progress_percentage,
            error_details=error_details
        )
        
        # Update workflow progress
        self.workflow_progress.current_status = new_status
        
        # Log status update with structured logging
        log_extra = {
            "topic": self.topic,
            "stage": stage.value,
            "agent_name": agent_name,
            "progress_percentage": progress_percentage,
            "status_message": message,
            "timestamp": new_status.timestamp.isoformat()
        }
        
        if error_details:
            log_extra["error_details"] = error_details
            logger.error("Progress status updated with error", extra=log_extra)
        else:
            logger.info("Progress status updated", extra=log_extra)
        
        # Call status callback if provided
        if self.status_callback:
            try:
                self.status_callback(new_status)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")
    
    def start_stage(
        self,
        stage: ProcessingStage,
        agent_name: str,
        message: Optional[str] = None
    ) -> None:
        """Start tracking a new processing stage.
        
        Args:
            stage: The processing stage being started
            agent_name: Name of the agent handling this stage
            message: Optional custom message, defaults to stage-based message
        """
        # Complete previous stage if exists
        if self._current_stage_metrics:
            self.complete_stage(success=True)
        
        # Set default message if not provided
        if message is None:
            message = self._get_default_stage_message(stage, agent_name)
        
        # Start new stage tracking
        self._current_stage_start = datetime.now()
        self._current_stage_metrics = ProcessingMetrics(
            stage_name=stage.value,
            start_time=self._current_stage_start,
            agent_name=agent_name
        )
        
        # Update status
        self.update_status(stage, message, agent_name)
        
        logger.info(
            f"Started stage: {stage.value}",
            extra={
                "topic": self.topic,
                "stage": stage.value,
                "agent_name": agent_name,
                "start_time": self._current_stage_start.isoformat()
            }
        )
    
    def complete_stage(
        self,
        success: bool = True,
        error_message: Optional[str] = None,
        tokens_used: Optional[int] = None,
        api_calls: Optional[int] = None
    ) -> None:
        """Complete the current processing stage.
        
        Args:
            success: Whether the stage completed successfully
            error_message: Error message if stage failed
            tokens_used: Number of tokens used in this stage
            api_calls: Number of API calls made in this stage
        """
        if not self._current_stage_metrics:
            logger.warning("No current stage to complete")
            return
        
        # Complete the metrics
        end_time = datetime.now()
        duration = (end_time - self._current_stage_metrics.start_time).total_seconds()
        
        self._current_stage_metrics.end_time = end_time
        self._current_stage_metrics.duration_seconds = duration
        self._current_stage_metrics.success = success
        self._current_stage_metrics.error_message = error_message
        self._current_stage_metrics.tokens_used = tokens_used
        self._current_stage_metrics.api_calls = api_calls
        
        # Add to workflow progress
        self.workflow_progress.stage_metrics.append(self._current_stage_metrics)
        
        # Log completion
        log_extra = {
            "topic": self.topic,
            "stage": self._current_stage_metrics.stage_name,
            "agent_name": self._current_stage_metrics.agent_name,
            "duration_seconds": duration,
            "success": success,
            "tokens_used": tokens_used,
            "api_calls": api_calls
        }
        
        if error_message:
            log_extra["error_message"] = error_message
            logger.error(f"Stage completed with error: {self._current_stage_metrics.stage_name}", extra=log_extra)
        else:
            logger.info(f"Stage completed successfully: {self._current_stage_metrics.stage_name}", extra=log_extra)
        
        # Clear current stage
        self._current_stage_metrics = None
        self._current_stage_start = None
    
    def complete_workflow(
        self,
        final_quality_score: Optional[float] = None,
        revision_count: int = 0
    ) -> WorkflowProgress:
        """Complete the entire workflow and return final progress.
        
        Args:
            final_quality_score: Final quality score if completed successfully
            revision_count: Number of revision cycles completed
            
        Returns:
            Complete workflow progress with all metrics
        """
        # Complete any remaining stage
        if self._current_stage_metrics:
            self.complete_stage(success=True)
        
        # Set completion time and metrics
        end_time = datetime.now()
        total_duration = (end_time - self.workflow_progress.total_start_time).total_seconds()
        
        self.workflow_progress.total_end_time = end_time
        self.workflow_progress.total_duration_seconds = total_duration
        self.workflow_progress.final_quality_score = final_quality_score
        self.workflow_progress.revision_count = revision_count
        
        # Update final status
        if self.workflow_progress.current_status.stage != ProcessingStage.FAILED:
            self.update_status(
                ProcessingStage.COMPLETED,
                "Blog generation completed successfully",
                progress_percentage=100.0
            )
        
        # Log workflow completion
        logger.info(
            "Workflow completed",
            extra={
                "topic": self.topic,
                "total_duration_seconds": total_duration,
                "final_quality_score": final_quality_score,
                "revision_count": revision_count,
                "total_stages": len(self.workflow_progress.stage_metrics),
                "success": self.workflow_progress.current_status.stage == ProcessingStage.COMPLETED
            }
        )
        
        return self.workflow_progress
    
    def fail_workflow(self, error_message: str, failing_agent: Optional[str] = None) -> WorkflowProgress:
        """Mark the workflow as failed.
        
        Args:
            error_message: Description of the failure
            failing_agent: Name of the agent that caused the failure
            
        Returns:
            Workflow progress with failure details
        """
        # Complete current stage with failure
        if self._current_stage_metrics:
            self.complete_stage(success=False, error_message=error_message)
        
        # Update status to failed
        self.update_status(
            ProcessingStage.FAILED,
            f"Blog generation failed: {error_message}",
            agent_name=failing_agent,
            error_details=error_message
        )
        
        # Set end time
        self.workflow_progress.total_end_time = datetime.now()
        if self.workflow_progress.total_start_time:
            self.workflow_progress.total_duration_seconds = (
                self.workflow_progress.total_end_time - self.workflow_progress.total_start_time
            ).total_seconds()
        
        logger.error(
            "Workflow failed",
            extra={
                "topic": self.topic,
                "error_message": error_message,
                "failing_agent": failing_agent,
                "total_duration_seconds": self.workflow_progress.total_duration_seconds
            }
        )
        
        return self.workflow_progress
    
    def get_current_progress(self) -> WorkflowProgress:
        """Get the current workflow progress.
        
        Returns:
            Current workflow progress
        """
        return self.workflow_progress
    
    def _calculate_progress_percentage(self, stage: ProcessingStage) -> float:
        """Calculate estimated progress percentage based on stage.
        
        Args:
            stage: Current processing stage
            
        Returns:
            Estimated progress percentage (0-100)
        """
        stage_percentages = {
            ProcessingStage.INITIALIZING: 5.0,
            ProcessingStage.RESEARCHING: 25.0,
            ProcessingStage.WRITING: 50.0,
            ProcessingStage.CRITIQUING: 70.0,
            ProcessingStage.REVISING: 85.0,
            ProcessingStage.COMPLETED: 100.0,
            ProcessingStage.FAILED: 0.0
        }
        
        base_percentage = stage_percentages.get(stage, 0.0)
        
        # Adjust for revision cycles
        if stage == ProcessingStage.REVISING and self.workflow_progress.revision_count > 0:
            # Reduce progress slightly for each revision cycle
            revision_penalty = min(self.workflow_progress.revision_count * 5, 15)
            base_percentage = max(base_percentage - revision_penalty, 70.0)
        
        return base_percentage
    
    def _get_default_stage_message(self, stage: ProcessingStage, agent_name: str) -> str:
        """Get default message for a processing stage.
        
        Args:
            stage: Processing stage
            agent_name: Name of the agent
            
        Returns:
            Default message for the stage
        """
        messages = {
            ProcessingStage.INITIALIZING: "Initializing blog generation workflow",
            ProcessingStage.RESEARCHING: f"{agent_name} is gathering research on the topic",
            ProcessingStage.WRITING: f"{agent_name} is creating the blog draft",
            ProcessingStage.CRITIQUING: f"{agent_name} is reviewing the draft for quality",
            ProcessingStage.REVISING: f"{agent_name} is revising the draft based on feedback",
            ProcessingStage.COMPLETED: "Blog generation completed successfully",
            ProcessingStage.FAILED: "Blog generation failed"
        }
        
        return messages.get(stage, f"{agent_name} is processing")


def create_structured_logger(name: str) -> logging.Logger:
    """Create a structured logger for agent interactions.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger with structured formatting
    """
    logger = logging.getLogger(name)
    
    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger