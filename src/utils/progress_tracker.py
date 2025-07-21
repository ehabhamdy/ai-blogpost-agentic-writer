"""Progress tracking utilities for multi-agent blog generation workflow."""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Enumeration of workflow stages for progress tracking."""
    INITIALIZING = "initializing"
    RESEARCHING = "researching"
    WRITING_INITIAL = "writing_initial"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


class AgentStatus(Enum):
    """Enumeration of individual agent statuses."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Data class representing a progress update."""
    stage: WorkflowStage
    agent: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    progress_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentProgress:
    """Data class tracking individual agent progress."""
    name: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    progress_percent: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return None


class ProgressTracker:
    """Comprehensive progress tracking for the blog generation workflow."""
    
    def __init__(self, callback: Optional[Callable[[ProgressUpdate], None]] = None):
        """Initialize the progress tracker.
        
        Args:
            callback: Optional callback function to receive progress updates
        """
        self.callback = callback
        self.current_stage = WorkflowStage.INITIALIZING
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.agents: Dict[str, AgentProgress] = {}
        self.updates: List[ProgressUpdate] = []
        self.revision_count = 0
        self.max_revisions = 3
        
        # Initialize agent progress trackers
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize progress tracking for all agents."""
        agent_names = ["research", "writing", "critique", "orchestrator"]
        for name in agent_names:
            self.agents[name] = AgentProgress(name=name)
    
    def update_stage(self, stage: WorkflowStage, message: str = "", metadata: Dict[str, Any] = None):
        """Update the current workflow stage.
        
        Args:
            stage: The new workflow stage
            message: Optional descriptive message
            metadata: Optional metadata dictionary
        """
        self.current_stage = stage
        
        # Calculate overall progress based on stage
        progress_map = {
            WorkflowStage.INITIALIZING: 5.0,
            WorkflowStage.RESEARCHING: 20.0,
            WorkflowStage.WRITING_INITIAL: 40.0,
            WorkflowStage.CRITIQUING: 60.0,
            WorkflowStage.REVISING: 80.0,
            WorkflowStage.FINALIZING: 95.0,
            WorkflowStage.COMPLETED: 100.0,
            WorkflowStage.ERROR: 0.0
        }
        
        progress_percent = progress_map.get(stage, 0.0)
        
        # Adjust for revision cycles
        if stage == WorkflowStage.REVISING and self.revision_count > 0:
            # Add extra progress for each revision cycle
            revision_progress = min(self.revision_count * 10, 30)
            progress_percent = min(progress_percent + revision_progress, 95.0)
        
        update = ProgressUpdate(
            stage=stage,
            agent="workflow",
            message=message or f"Workflow stage: {stage.value}",
            progress_percent=progress_percent,
            metadata=metadata or {}
        )
        
        self._send_update(update)
        
        if stage == WorkflowStage.COMPLETED:
            self.end_time = datetime.now()
        elif stage == WorkflowStage.ERROR:
            self.end_time = datetime.now()
    
    def update_agent(self, agent_name: str, status: AgentStatus, task: str = "", 
                    progress: float = 0.0, metadata: Dict[str, Any] = None):
        """Update individual agent progress.
        
        Args:
            agent_name: Name of the agent
            status: Current agent status
            task: Current task description
            progress: Progress percentage (0-100)
            metadata: Optional metadata dictionary
        """
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentProgress(name=agent_name)
        
        agent = self.agents[agent_name]
        
        # Update agent status
        if status == AgentStatus.WORKING and agent.status != AgentStatus.WORKING:
            agent.start_time = datetime.now()
        elif status in [AgentStatus.COMPLETED, AgentStatus.ERROR] and agent.status == AgentStatus.WORKING:
            agent.end_time = datetime.now()
        
        agent.status = status
        agent.current_task = task
        agent.progress_percent = progress
        if metadata:
            agent.metadata.update(metadata)
        
        # Send progress update
        update = ProgressUpdate(
            stage=self.current_stage,
            agent=agent_name,
            message=f"{agent_name.title()} Agent: {task}" if task else f"{agent_name.title()} Agent: {status.value}",
            progress_percent=self._calculate_overall_progress(),
            metadata={
                "agent_status": status.value,
                "agent_progress": progress,
                "agent_task": task,
                **(metadata or {})
            }
        )
        
        self._send_update(update)
    
    def update_revision_count(self, count: int):
        """Update the revision count."""
        self.revision_count = count
        
        update = ProgressUpdate(
            stage=self.current_stage,
            agent="workflow",
            message=f"Revision cycle {count}/{self.max_revisions}",
            progress_percent=self._calculate_overall_progress(),
            metadata={"revision_count": count, "max_revisions": self.max_revisions}
        )
        
        self._send_update(update)
    
    def set_error(self, agent_name: str, error_message: str):
        """Set an error for a specific agent.
        
        Args:
            agent_name: Name of the agent that encountered an error
            error_message: Error message description
        """
        if agent_name in self.agents:
            self.agents[agent_name].status = AgentStatus.ERROR
            self.agents[agent_name].error_message = error_message
            self.agents[agent_name].end_time = datetime.now()
        
        self.update_stage(WorkflowStage.ERROR, f"Error in {agent_name}: {error_message}")
    
    def _calculate_overall_progress(self) -> float:
        """Calculate overall progress based on agent statuses and current stage."""
        # Base progress from current stage
        stage_progress_map = {
            WorkflowStage.INITIALIZING: 5.0,
            WorkflowStage.RESEARCHING: 20.0,
            WorkflowStage.WRITING_INITIAL: 40.0,
            WorkflowStage.CRITIQUING: 60.0,
            WorkflowStage.REVISING: 80.0,
            WorkflowStage.FINALIZING: 95.0,
            WorkflowStage.COMPLETED: 100.0,
            WorkflowStage.ERROR: 0.0
        }
        
        base_progress = stage_progress_map.get(self.current_stage, 0.0)
        
        # Add agent-specific progress within the current stage
        active_agents = [agent for agent in self.agents.values() 
                        if agent.status == AgentStatus.WORKING]
        
        if active_agents:
            agent_progress = sum(agent.progress_percent for agent in active_agents) / len(active_agents)
            # Add up to 10% based on agent progress within current stage
            stage_bonus = (agent_progress / 100.0) * 10.0
            base_progress = min(base_progress + stage_bonus, 100.0)
        
        return base_progress
    
    def _send_update(self, update: ProgressUpdate):
        """Send progress update to callback and store in history.
        
        Args:
            update: The progress update to send
        """
        self.updates.append(update)
        
        # Log the update
        logger.info(f"[{update.stage.value.upper()}] {update.message} ({update.progress_percent:.1f}%)")
        
        # Send to callback if provided
        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a comprehensive status summary.
        
        Returns:
            Dictionary containing current status information
        """
        duration = None
        if self.start_time:
            end = self.end_time or datetime.now()
            duration = (end - self.start_time).total_seconds()
        
        return {
            "current_stage": self.current_stage.value,
            "overall_progress": self._calculate_overall_progress(),
            "duration_seconds": duration,
            "revision_count": self.revision_count,
            "max_revisions": self.max_revisions,
            "agents": {
                name: {
                    "status": agent.status.value,
                    "current_task": agent.current_task,
                    "progress": agent.progress_percent,
                    "duration": agent.duration,
                    "error": agent.error_message
                }
                for name, agent in self.agents.items()
            },
            "recent_updates": [
                {
                    "timestamp": update.timestamp.isoformat(),
                    "stage": update.stage.value,
                    "agent": update.agent,
                    "message": update.message,
                    "progress": update.progress_percent
                }
                for update in self.updates[-5:]  # Last 5 updates
            ]
        }
    
    def print_status(self):
        """Print current status to console in a formatted way."""
        summary = self.get_status_summary()
        
        print("\n" + "="*60)
        print(f"🎯 BLOG GENERATION PROGRESS")
        print("="*60)
        print(f"Stage: {summary['current_stage'].upper()}")
        print(f"Overall Progress: {summary['overall_progress']:.1f}%")
        
        if summary['duration_seconds']:
            print(f"Duration: {summary['duration_seconds']:.1f}s")
        
        if summary['revision_count'] > 0:
            print(f"Revisions: {summary['revision_count']}/{summary['max_revisions']}")
        
        print("\n📊 AGENT STATUS:")
        for name, agent in summary['agents'].items():
            status_emoji = {
                'idle': '⏸️',
                'working': '🔄',
                'completed': '✅',
                'error': '❌'
            }.get(agent['status'], '❓')
            
            print(f"  {status_emoji} {name.title()}: {agent['status']}")
            if agent['current_task']:
                print(f"    Task: {agent['current_task']}")
            if agent['progress'] > 0:
                print(f"    Progress: {agent['progress']:.1f}%")
            if agent['error']:
                print(f"    Error: {agent['error']}")
        
        print("\n📝 RECENT UPDATES:")
        for update in summary['recent_updates']:
            timestamp = datetime.fromisoformat(update['timestamp']).strftime("%H:%M:%S")
            print(f"  [{timestamp}] {update['message']}")
        
        print("="*60)


class StreamingProgressTracker(ProgressTracker):
    """Enhanced progress tracker with real-time streaming capabilities."""
    
    def __init__(self, callback: Optional[Callable[[ProgressUpdate], None]] = None,
                 print_updates: bool = True, update_interval: float = 1.0):
        """Initialize streaming progress tracker.
        
        Args:
            callback: Optional callback function for progress updates
            print_updates: Whether to print updates to console
            update_interval: Minimum interval between console updates (seconds)
        """
        super().__init__(callback)
        self.print_updates = print_updates
        self.update_interval = update_interval
        self.last_print_time = 0.0
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
    
    async def start_streaming(self):
        """Start the streaming progress updates."""
        self._running = True
        self._update_task = asyncio.create_task(self._stream_updates())
    
    async def stop_streaming(self):
        """Stop the streaming progress updates."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    async def _stream_updates(self):
        """Stream progress updates at regular intervals."""
        while self._running:
            try:
                current_time = time.time()
                if (current_time - self.last_print_time) >= self.update_interval:
                    if self.print_updates:
                        self._print_compact_status()
                    self.last_print_time = current_time
                
                await asyncio.sleep(0.5)  # Check every 0.5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in streaming updates: {e}")
    
    def _print_compact_status(self):
        """Print a compact status update."""
        summary = self.get_status_summary()
        
        # Create a compact status line
        stage = summary['current_stage'].upper()
        progress = summary['overall_progress']
        
        # Get active agent info
        active_agents = [
            f"{name}({info['progress']:.0f}%)" 
            for name, info in summary['agents'].items() 
            if info['status'] == 'working'
        ]
        
        active_info = f" | Active: {', '.join(active_agents)}" if active_agents else ""
        
        # Progress bar
        bar_length = 20
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        status_line = f"\r🎯 [{bar}] {progress:5.1f}% | {stage}{active_info}"
        
        # Print without newline to overwrite previous line
        print(status_line, end='', flush=True)
        
        # Add newline for important milestones
        if progress in [20, 40, 60, 80, 100] and abs(progress - getattr(self, '_last_milestone', 0)) > 15:
            print()  # Add newline for milestone
            self._last_milestone = progress


# Convenience functions for easy integration
def create_progress_tracker(callback: Optional[Callable[[ProgressUpdate], None]] = None,
                          streaming: bool = False) -> ProgressTracker:
    """Create a progress tracker instance.
    
    Args:
        callback: Optional callback function for progress updates
        streaming: Whether to use streaming progress tracker
        
    Returns:
        ProgressTracker instance
    """
    if streaming:
        return StreamingProgressTracker(callback=callback)
    else:
        return ProgressTracker(callback=callback)


def default_progress_callback(update: ProgressUpdate):
    """Default progress callback that prints formatted updates.
    
    Args:
        update: The progress update to handle
    """
    timestamp = update.timestamp.strftime("%H:%M:%S")
    print(f"[{timestamp}] {update.message} ({update.progress_percent:.1f}%)")