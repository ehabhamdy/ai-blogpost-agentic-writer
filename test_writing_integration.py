#!/usr/bin/env python3
"""
Integration test for Writing Agent to verify it works with Pydantic AI.

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.writing_agent import WritingAgent
from src.models.data_models import ResearchOutput, ResearchFinding
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()

async def test_writing_agent_integration():
    """Test Writing Agent integration with real models."""
    
    # Skip if no API keys available
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping integration test - no OPENAI_API_KEY found")
        return
    
    # Create model and dependencies
    model = OpenAIModel("gpt-4o-mini")  # Use cheaper model for testing
    
    async with httpx.AsyncClient() as http_client:
        deps = SharedDependencies(
            http_client=http_client,
            tavily_client=TavilyClient(api_key="dummy"),  # Won't be used in this test
            max_iterations=3,
            quality_threshold=7.0
        )
        
        # Create Writing Agent
        writing_agent = WritingAgent(model)
        
        # Create sample research data
        sample_findings = [
            ResearchFinding(
                fact="Intermittent fasting can reduce body weight by 3-8% over 3-24 weeks",
                source_url="https://example.com/study1",
                relevance_score=0.9,
                category="statistic"
            ),
            ResearchFinding(
                fact="Dr. Smith, a leading researcher, recommends 16:8 intermittent fasting for beginners",
                source_url="https://example.com/expert1",
                relevance_score=0.8,
                category="expert_opinion"
            ),
            ResearchFinding(
                fact="Studies show intermittent fasting improves insulin sensitivity",
                source_url="https://example.com/study2",
                relevance_score=0.85,
                category="study"
            )
        ]
        
        research_output = ResearchOutput(
            topic="intermittent fasting",
            findings=sample_findings,
            summary="Intermittent fasting shows promising results for weight loss and metabolic health",
            confidence_level=0.8
        )
        
        try:
            # Test blog draft creation
            print("Testing blog draft creation...")
            blog_draft = await writing_agent.create_blog_draft(
                topic="intermittent fasting",
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Successfully created blog draft:")
            print(f"   Title: {blog_draft.title}")
            print(f"   Introduction length: {len(blog_draft.introduction)} chars")
            print(f"   Body sections: {len(blog_draft.body_sections)}")
            print(f"   Conclusion length: {len(blog_draft.conclusion)} chars")
            print(f"   Word count: {blog_draft.word_count}")
            
            # Test revision functionality
            print("\nTesting blog draft revision...")
            revised_draft = await writing_agent.revise_blog_draft(
                original_draft=blog_draft,
                feedback="Make the introduction more engaging and add more specific statistics",
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Successfully revised blog draft:")
            print(f"   Revised title: {revised_draft.title}")
            print(f"   Revised introduction length: {len(revised_draft.introduction)} chars")
            print(f"   Revised word count: {revised_draft.word_count}")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_writing_agent_integration())