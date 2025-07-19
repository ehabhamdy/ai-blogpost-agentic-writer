#!/usr/bin/env python3
"""
Integration test for Research Agent.
Run this script with proper API keys set to test the full functionality.

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
- TAVILY_API_KEY: Tavily API key
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.research_agent import ResearchAgent
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


async def test_research_agent():
    """Test the research agent with real APIs."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set. Skipping integration test.")
        return
    
    print("🔍 Starting Research Agent integration test...")
    
    try:
        # Initialize model and dependencies
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=3,
                quality_threshold=7.0
            )
            
            # Create research agent
            research_agent = ResearchAgent(model)
            print("✓ Research agent created successfully")
            
            # Test research on a simple topic
            topic = "benefits of intermittent fasting"
            print(f"🔍 Researching topic: {topic}")
            
            result = await research_agent.research_topic(topic, deps)
            
            print(f"✓ Research completed successfully!")
            print(f"  - Topic: {result.topic}")
            print(f"  - Number of findings: {len(result.findings)}")
            print(f"  - Confidence level: {result.confidence_level}")
            print(f"  - Summary: {result.summary[:100]}...")
            
            # Show a few findings
            print("\n📋 Sample findings:")
            for i, finding in enumerate(result.findings[:3]):
                print(f"  {i+1}. [{finding.category}] {finding.fact[:80]}...")
                print(f"     Relevance: {finding.relevance_score:.2f}")
                print(f"     Source: {finding.source_url}")
                print()
            
            print("✅ Integration test completed successfully!")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_research_agent())