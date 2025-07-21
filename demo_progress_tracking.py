#!/usr/bin/env python3
"""
Demo script showcasing the comprehensive progress tracking system for AI Blog Generation.

This demo shows:
1. Basic progress tracking with console output
2. Streaming progress tracking with real-time updates
3. Custom progress callbacks
4. Integration with the complete blog generation workflow
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.research_agent import ResearchAgent
from src.agents.writing_agent import WritingAgent
from src.agents.critique_agent import CritiqueAgent
from src.utils.dependencies import SharedDependencies
from src.utils.progress_tracker import (
    ProgressTracker, 
    StreamingProgressTracker, 
    ProgressUpdate,
    WorkflowStage,
    AgentStatus,
    default_progress_callback,
    create_progress_tracker
)

# Load environment variables
load_dotenv()


def custom_progress_callback(update: ProgressUpdate):
    """Custom progress callback with enhanced formatting."""
    # Color codes for different stages
    stage_colors = {
        WorkflowStage.INITIALIZING: '\033[94m',  # Blue
        WorkflowStage.RESEARCHING: '\033[93m',   # Yellow
        WorkflowStage.WRITING_INITIAL: '\033[92m',  # Green
        WorkflowStage.CRITIQUING: '\033[95m',    # Magenta
        WorkflowStage.REVISING: '\033[96m',      # Cyan
        WorkflowStage.FINALIZING: '\033[97m',    # White
        WorkflowStage.COMPLETED: '\033[92m',     # Green
        WorkflowStage.ERROR: '\033[91m'          # Red
    }
    
    reset_color = '\033[0m'
    
    # Get color for current stage
    color = stage_colors.get(update.stage, '\033[0m')
    
    # Format timestamp
    timestamp = update.timestamp.strftime("%H:%M:%S")
    
    # Create progress bar
    progress = update.progress_percent
    bar_length = 20
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Format the message
    stage_name = update.stage.value.upper().replace('_', ' ')
    agent_name = update.agent.upper() if update.agent != 'workflow' else 'SYSTEM'
    
    print(f"{color}[{timestamp}] {agent_name:12} | [{bar}] {progress:5.1f}% | {stage_name}: {update.message}{reset_color}")


async def demo_basic_progress_tracking():
    """Demonstrate basic progress tracking functionality."""
    print("\n" + "="*80)
    print("🎯 DEMO 1: BASIC PROGRESS TRACKING")
    print("="*80)
    
    # Create a basic progress tracker
    tracker = ProgressTracker(callback=custom_progress_callback)
    
    print("📊 Simulating workflow stages...")
    
    # Simulate workflow progression
    stages = [
        (WorkflowStage.INITIALIZING, "Setting up blog generation workflow"),
        (WorkflowStage.RESEARCHING, "Gathering information from web sources"),
        (WorkflowStage.WRITING_INITIAL, "Creating initial blog draft"),
        (WorkflowStage.CRITIQUING, "Analyzing content quality and structure"),
        (WorkflowStage.REVISING, "Improving draft based on feedback"),
        (WorkflowStage.FINALIZING, "Preparing final output"),
        (WorkflowStage.COMPLETED, "Blog generation completed successfully")
    ]
    
    for i, (stage, message) in enumerate(stages):
        tracker.update_stage(stage, message)
        
        # Simulate agent work
        if stage == WorkflowStage.RESEARCHING:
            tracker.update_agent("research", AgentStatus.WORKING, "Searching web for information", 50.0)
            await asyncio.sleep(1)
            tracker.update_agent("research", AgentStatus.COMPLETED, "Found 15 research findings", 100.0, 
                               {"findings_count": 15, "confidence": 0.85})
        
        elif stage == WorkflowStage.WRITING_INITIAL:
            tracker.update_agent("writing", AgentStatus.WORKING, "Creating blog structure", 25.0)
            await asyncio.sleep(0.5)
            tracker.update_agent("writing", AgentStatus.WORKING, "Writing introduction", 50.0)
            await asyncio.sleep(0.5)
            tracker.update_agent("writing", AgentStatus.WORKING, "Writing body sections", 75.0)
            await asyncio.sleep(0.5)
            tracker.update_agent("writing", AgentStatus.COMPLETED, "Draft completed (850 words)", 100.0,
                               {"word_count": 850})
        
        elif stage == WorkflowStage.CRITIQUING:
            tracker.update_agent("critique", AgentStatus.WORKING, "Analyzing content quality", 30.0)
            await asyncio.sleep(0.5)
            tracker.update_agent("critique", AgentStatus.WORKING, "Checking factual accuracy", 60.0)
            await asyncio.sleep(0.5)
            tracker.update_agent("critique", AgentStatus.COMPLETED, "Quality score: 7.2/10", 100.0,
                               {"quality_score": 7.2, "feedback_count": 3})
        
        elif stage == WorkflowStage.REVISING:
            tracker.update_revision_count(1)
            tracker.update_agent("writing", AgentStatus.WORKING, "Revising based on feedback", 80.0)
            await asyncio.sleep(0.8)
            tracker.update_agent("writing", AgentStatus.COMPLETED, "Revision completed (920 words)", 100.0,
                               {"word_count": 920})
        
        await asyncio.sleep(0.3)
    
    # Show final status
    print("\n📋 FINAL STATUS SUMMARY:")
    tracker.print_status()


async def demo_streaming_progress_tracking():
    """Demonstrate streaming progress tracking with real-time updates."""
    print("\n" + "="*80)
    print("🎯 DEMO 2: STREAMING PROGRESS TRACKING")
    print("="*80)
    
    # Create a streaming progress tracker
    tracker = StreamingProgressTracker(
        callback=custom_progress_callback,
        print_updates=True,
        update_interval=0.5
    )
    
    print("🔄 Starting streaming progress updates...")
    
    # Start streaming
    await tracker.start_streaming()
    
    try:
        # Simulate a longer workflow with multiple agents working
        tracker.update_stage(WorkflowStage.INITIALIZING, "Initializing multi-agent system")
        await asyncio.sleep(1)
        
        # Research phase
        tracker.update_stage(WorkflowStage.RESEARCHING, "Starting research phase")
        tracker.update_agent("research", AgentStatus.WORKING, "Connecting to Tavily API", 10.0)
        await asyncio.sleep(2)
        tracker.update_agent("research", AgentStatus.WORKING, "Searching for recent articles", 40.0)
        await asyncio.sleep(2)
        tracker.update_agent("research", AgentStatus.WORKING, "Processing search results", 70.0)
        await asyncio.sleep(1.5)
        tracker.update_agent("research", AgentStatus.COMPLETED, "Research completed", 100.0)
        
        # Writing phase
        tracker.update_stage(WorkflowStage.WRITING_INITIAL, "Creating initial draft")
        tracker.update_agent("writing", AgentStatus.WORKING, "Structuring content", 20.0)
        await asyncio.sleep(2)
        tracker.update_agent("writing", AgentStatus.WORKING, "Writing introduction", 40.0)
        await asyncio.sleep(2)
        tracker.update_agent("writing", AgentStatus.WORKING, "Developing main points", 70.0)
        await asyncio.sleep(2)
        tracker.update_agent("writing", AgentStatus.COMPLETED, "Initial draft ready", 100.0)
        
        # Critique phase
        tracker.update_stage(WorkflowStage.CRITIQUING, "Analyzing draft quality")
        tracker.update_agent("critique", AgentStatus.WORKING, "Checking clarity", 30.0)
        await asyncio.sleep(1.5)
        tracker.update_agent("critique", AgentStatus.WORKING, "Verifying facts", 60.0)
        await asyncio.sleep(1.5)
        tracker.update_agent("critique", AgentStatus.WORKING, "Assessing structure", 90.0)
        await asyncio.sleep(1)
        tracker.update_agent("critique", AgentStatus.COMPLETED, "Critique completed", 100.0)
        
        # Revision phase
        tracker.update_stage(WorkflowStage.REVISING, "Revising based on feedback")
        tracker.update_revision_count(1)
        tracker.update_agent("writing", AgentStatus.WORKING, "Addressing feedback", 50.0)
        await asyncio.sleep(2)
        tracker.update_agent("writing", AgentStatus.COMPLETED, "Revision completed", 100.0)
        
        # Final phase
        tracker.update_stage(WorkflowStage.COMPLETED, "Blog generation completed")
        
    finally:
        # Stop streaming
        await tracker.stop_streaming()
    
    print("\n\n📋 FINAL STREAMING STATUS:")
    tracker.print_status()


async def demo_real_blog_generation_with_progress():
    """Demonstrate progress tracking with actual blog generation."""
    print("\n" + "="*80)
    print("🎯 DEMO 3: REAL BLOG GENERATION WITH PROGRESS TRACKING")
    print("="*80)
    
    # Check for API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key or not tavily_key:
        print("❌ API keys not available. Skipping real blog generation demo.")
        print("   Set OPENAI_API_KEY and TAVILY_API_KEY environment variables to run this demo.")
        return
    
    print("🚀 Running actual blog generation with progress tracking...")
    
    try:
        # Initialize models and agents
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            # Create shared dependencies
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=2,
                quality_threshold=7.0
            )
            
            # Create agents
            orchestrator_agent = OrchestratorAgent(model)
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            critique_agent = CritiqueAgent(model)
            
            # Create streaming progress tracker
            progress_tracker = StreamingProgressTracker(
                callback=custom_progress_callback,
                print_updates=True,
                update_interval=1.0
            )
            
            # Start streaming
            await progress_tracker.start_streaming()
            
            try:
                # Test topic
                topic = "benefits of morning exercise"
                
                print(f"📝 Generating blog post about: '{topic}'")
                print("🔄 Watch the progress updates below...\n")
                
                # Run the complete workflow with progress tracking
                result = await orchestrator_agent.generate_blog_post(
                    topic=topic,
                    research_agent=research_agent,
                    writing_agent=writing_agent,
                    critique_agent=critique_agent,
                    deps=deps,
                    progress_tracker=progress_tracker
                )
                
                # Display results
                print("\n" + "="*80)
                print("✅ BLOG GENERATION COMPLETED!")
                print("="*80)
                print(f"📊 Final Quality Score: {result.quality_score:.1f}/10")
                print(f"🔄 Revision Cycles: {result.revision_count}")
                print(f"⏱️  Total Time: {result.total_processing_time:.1f} seconds")
                print(f"📝 Word Count: {result.final_post.word_count}")
                print(f"📖 Title: {result.final_post.title}")
                
                # Show progress summary
                print("\n📋 PROGRESS SUMMARY:")
                progress_tracker.print_status()
                
            finally:
                await progress_tracker.stop_streaming()
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_custom_progress_integration():
    """Demonstrate custom progress tracking integration."""
    print("\n" + "="*80)
    print("🎯 DEMO 4: CUSTOM PROGRESS INTEGRATION")
    print("="*80)
    
    # Custom callback that saves progress to a file
    progress_log = []
    
    def logging_callback(update: ProgressUpdate):
        """Custom callback that logs progress to a list."""
        progress_log.append({
            'timestamp': update.timestamp.isoformat(),
            'stage': update.stage.value,
            'agent': update.agent,
            'message': update.message,
            'progress': update.progress_percent,
            'metadata': update.metadata
        })
        
        # Also print to console
        custom_progress_callback(update)
    
    # Create tracker with custom callback
    tracker = create_progress_tracker(callback=logging_callback, streaming=False)
    
    print("📊 Simulating workflow with custom logging...")
    
    # Simulate workflow
    stages = [
        (WorkflowStage.INITIALIZING, "Custom workflow initialization"),
        (WorkflowStage.RESEARCHING, "Custom research process"),
        (WorkflowStage.WRITING_INITIAL, "Custom writing process"),
        (WorkflowStage.COMPLETED, "Custom workflow completed")
    ]
    
    for stage, message in stages:
        tracker.update_stage(stage, message, {"custom_data": f"stage_{stage.value}"})
        await asyncio.sleep(0.5)
    
    # Show logged progress
    print(f"\n📝 LOGGED {len(progress_log)} PROGRESS UPDATES:")
    for i, log_entry in enumerate(progress_log[-3:], 1):  # Show last 3
        print(f"  {i}. [{log_entry['timestamp'][:19]}] {log_entry['stage']}: {log_entry['message']}")
    
    # Show status summary
    print("\n📊 STATUS SUMMARY:")
    summary = tracker.get_status_summary()
    print(f"  Overall Progress: {summary['overall_progress']:.1f}%")
    print(f"  Duration: {summary['duration_seconds']:.1f}s")
    print(f"  Total Updates: {len(progress_log)}")


async def main():
    """Run all progress tracking demos."""
    print("🎯 AI BLOG GENERATION - PROGRESS TRACKING DEMOS")
    print("="*80)
    print("This demo showcases comprehensive progress tracking capabilities:")
    print("• Basic progress tracking with callbacks")
    print("• Streaming real-time progress updates")
    print("• Integration with actual blog generation")
    print("• Custom progress tracking implementations")
    print("="*80)
    
    try:
        # Run all demos
        await demo_basic_progress_tracking()
        await asyncio.sleep(2)
        
        await demo_streaming_progress_tracking()
        await asyncio.sleep(2)
        
        await demo_custom_progress_integration()
        await asyncio.sleep(2)
        
        # Real blog generation demo (requires API keys)
        await demo_real_blog_generation_with_progress()
        
        print("\n" + "="*80)
        print("🎉 ALL PROGRESS TRACKING DEMOS COMPLETED!")
        print("="*80)
        print("Key Features Demonstrated:")
        print("✅ Real-time progress updates with visual indicators")
        print("✅ Stage-based workflow tracking")
        print("✅ Individual agent status monitoring")
        print("✅ Revision cycle tracking")
        print("✅ Custom callback integration")
        print("✅ Streaming progress updates")
        print("✅ Comprehensive status reporting")
        print("✅ Error handling and recovery")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())