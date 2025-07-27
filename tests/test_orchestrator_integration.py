#!/usr/bin/env python3
"""
Integration test for Orchestrator Agent to verify complete workflow coordination.

This test verifies the full multi-agent blog generation workflow including:
- Research delegation and coordination
- Writing delegation with iterative revisions
- Critique delegation and quality assessment
- Revision decision making and workflow management

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
- TAVILY_API_KEY: Tavily API key (for research functionality)
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

# Load environment variables
load_dotenv()


async def test_orchestrator_agent_integration():
    """Test Orchestrator Agent integration with real models and full workflow."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set. Skipping integration test.")
        return
    
    print("🎯 Starting Orchestrator Agent integration test...")
    print("This test will run the complete multi-agent blog generation workflow.")
    print()
    
    try:
        # Initialize models - using different models for different agents
        model_4o_mini = OpenAIModel('gpt-4o-mini')
        model_4o = OpenAIModel('gpt-4o')
        
        async with httpx.AsyncClient() as http_client:
            # Create shared dependencies
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=2,  # Limit iterations for testing
                quality_threshold=6.0  # Reasonable threshold for testing
            )
            
            # Create all agents with appropriate models
            print("🔧 Initializing agents...")
            orchestrator_agent = OrchestratorAgent(model_4o)  # Use GPT-4o for orchestration
            research_agent = ResearchAgent(model_4o_mini)     # Use mini for research
            writing_agent = WritingAgent(model_4o)            # Use GPT-4o for writing
            critique_agent = CritiqueAgent(model_4o_mini)     # Use mini for critique
            print("✓ All agents created successfully")
            print()
            
            # Test topic
            topic = "benefits of meditation for mental health"
            print(f"📝 Testing complete workflow for topic: '{topic}'")
            print()
            
            # Run the complete blog generation workflow
            print("🚀 Starting blog generation workflow...")
            result = await orchestrator_agent.generate_blog_post(
                topic=topic,
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                deps=deps
            )
            
            print("✅ Blog generation workflow completed successfully!")
            print()
            
            # Display results
            print("📊 WORKFLOW RESULTS:")
            print("=" * 50)
            print(f"Final Quality Score: {result.quality_score:.1f}/10")
            print(f"Revision Cycles: {result.revision_count}")
            print(f"Total Processing Time: {result.total_processing_time:.2f} seconds")
            print(f"Total Word Count: {result.final_post.word_count}")
            print()
            
            # Print the complete blog post
            print("📝 COMPLETE BLOG POST:")
            print("=" * 80)
            print(f"TITLE: {result.final_post.title}")
            print()
            print("INTRODUCTION:")
            print(result.final_post.introduction)
            print()
            
            for i, section in enumerate(result.final_post.body_sections, 1):
                print(f"SECTION {i}:")
                print(section)
                print()
            
            print("CONCLUSION:")
            print(result.final_post.conclusion)
            print()
            print("=" * 80)
            
            # Verify word count
            complete_text = f"{result.final_post.title} {result.final_post.introduction} {' '.join(result.final_post.body_sections)} {result.final_post.conclusion}"
            actual_word_count = len(complete_text.split())
            print(f"Reported word count: {result.final_post.word_count}")
            print(f"Actual word count: {actual_word_count}")
            print()
            
            # Display research summary
            print("🔍 RESEARCH SUMMARY:")
            print("=" * 50)
            print(f"Research Topic: {result.research_data.topic}")
            print(f"Research Findings: {len(result.research_data.findings)}")
            print(f"Research Confidence: {result.research_data.confidence_level:.2f}")
            print(f"Research Summary: {result.research_data.summary[:200]}...")
            print()
            
            print("✅ Integration test completed successfully!")
            print(f"Generated a {result.final_post.word_count}-word blog post with quality score {result.quality_score:.1f}")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_orchestrator_agent_integration())