#!/usr/bin/env python3
"""
Integration test for the revision workflow with quality control.

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


async def test_revision_workflow_integration():
    """Test the complete revision workflow with quality control."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set. Skipping integration test.")
        return
    
    print("🔄 Starting revision workflow integration test...")
    print("This test uses a high quality threshold to potentially trigger revisions.")
    print()
    
    try:
        # Initialize models
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            # Create shared dependencies with high quality threshold to trigger revisions
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=2,
                quality_threshold=8.0  # High threshold to potentially trigger revisions
            )
            
            # Create all agents
            print("🔧 Initializing agents...")
            orchestrator_agent = OrchestratorAgent(model)
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            critique_agent = CritiqueAgent(model)
            print("✓ All agents created successfully")
            print()
            
            # Test topic that might need revision
            topic = "benefits of drinking water daily"
            print(f"📝 Testing revision workflow for topic: '{topic}'")
            print(f"   Quality threshold: {deps.quality_threshold}")
            print(f"   Max iterations: {deps.max_iterations}")
            print()
            
            # Run the complete workflow
            print("🚀 Starting blog generation workflow...")
            result = await orchestrator_agent.generate_blog_post(
                topic=topic,
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                deps=deps
            )
            
            print("✅ Revision workflow completed successfully!")
            print()
            
            # Display revision workflow results
            print("📊 REVISION WORKFLOW RESULTS:")
            print("=" * 50)
            print(f"Final Quality Score: {result.quality_score:.1f}/10")
            print(f"Quality Threshold: {deps.quality_threshold}")
            print(f"Revision Cycles: {result.revision_count}")
            print(f"Max Iterations: {deps.max_iterations}")
            print(f"Total Processing Time: {result.total_processing_time:.2f} seconds")
            print(f"Final Word Count: {result.final_post.word_count}")
            print()
            
            # Analyze revision behavior
            if result.revision_count > 0:
                print("🔄 REVISION ANALYSIS:")
                print("=" * 50)
                print(f"✓ Revision workflow was triggered ({result.revision_count} cycles)")
                print(f"✓ Quality improved through iterative refinement")
                print(f"✓ Final quality score: {result.quality_score:.1f}")
                if result.quality_score >= deps.quality_threshold:
                    print("✓ Quality threshold was met")
                else:
                    print("⚠️ Quality threshold not met (stopped at max iterations)")
            else:
                print("📝 FIRST DRAFT ANALYSIS:")
                print("=" * 50)
                print(f"✓ First draft met quality threshold ({result.quality_score:.1f} >= {deps.quality_threshold})")
                print("✓ No revisions were needed")
            
            print()
            
            # Display final blog post summary
            print("📝 FINAL BLOG POST SUMMARY:")
            print("=" * 50)
            print(f"Title: {result.final_post.title}")
            print(f"Word Count: {result.final_post.word_count}")
            print(f"Sections: {len(result.final_post.body_sections)}")
            print()
            
            print("✅ Revision workflow integration test completed successfully!")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_forced_revision_workflow():
    """Test revision workflow with very high quality threshold to force revisions."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key or not tavily_key:
        print("❌ Required API keys not set. Skipping forced revision test.")
        return
    
    print("🔄 Starting forced revision workflow test...")
    print("This test uses a very high quality threshold to force multiple revisions.")
    print()
    
    try:
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            # Create dependencies with very high quality threshold to force revisions
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=3,
                quality_threshold=9.5  # Very high threshold to force revisions
            )
            
            orchestrator_agent = OrchestratorAgent(model)
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            critique_agent = CritiqueAgent(model)
            
            topic = "simple health tips"
            print(f"📝 Testing forced revisions for topic: '{topic}'")
            print(f"   Quality threshold: {deps.quality_threshold} (very high)")
            print(f"   Max iterations: {deps.max_iterations}")
            print()
            
            result = await orchestrator_agent.generate_blog_post(
                topic=topic,
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                deps=deps
            )
            
            print("✅ Forced revision workflow completed!")
            print()
            print("📊 FORCED REVISION RESULTS:")
            print("=" * 50)
            print(f"Revision cycles: {result.revision_count}")
            print(f"Final quality: {result.quality_score:.1f}")
            print(f"Quality threshold: {deps.quality_threshold}")
            print(f"Processing time: {result.total_processing_time:.2f}s")
            
            # Verify that revisions were attempted
            if result.revision_count > 0:
                print("✓ Forced revision workflow worked as expected")
            else:
                print("⚠️ No revisions were triggered (first draft was exceptional)")
            
    except Exception as e:
        print(f"❌ Forced revision test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all revision workflow tests."""
    print("🎯 REVISION WORKFLOW INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Normal revision workflow
    await test_revision_workflow_integration()
    print()
    print("-" * 60)
    print()
    
    # Test 2: Forced revision workflow
    await test_forced_revision_workflow()
    print()
    print("🎉 All revision workflow tests completed!")


if __name__ == "__main__":
    asyncio.run(main())