#!/usr/bin/env python3
"""
Integration test for Writing Agent.

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
import httpx

from src.agents.writing_agent import WritingAgent
from src.models.data_models import ResearchOutput, ResearchFinding
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


async def test_writing_agent_integration():
    """Test Writing Agent integration with real models."""
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return
    
    print("✍️ Starting Writing Agent integration test...")
    
    try:
        # Create model and dependencies
        model = OpenAIModel("gpt-4o-mini")  # Use cost-effective model for testing
        
        async with httpx.AsyncClient() as http_client:
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=None,  # Not needed for writing tests
                max_iterations=3,
                quality_threshold=7.0
            )
            
            # Create Writing Agent
            writing_agent = WritingAgent(model)
            print("✓ Writing agent created successfully")
            
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
                topic="intermittent fasting benefits",
                findings=sample_findings,
                summary="Intermittent fasting shows promising results for weight loss and metabolic health improvements.",
                confidence_level=0.8
            )
            
            # Test blog draft creation
            print("📝 Testing blog draft creation...")
            blog_draft = await writing_agent.create_blog_draft(
                topic="intermittent fasting benefits",
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Successfully created blog draft:")
            print(f"   Title: {blog_draft.title}")
            print(f"   Introduction length: {len(blog_draft.introduction)} chars")
            print(f"   Body sections: {len(blog_draft.body_sections)}")
            print(f"   Conclusion length: {len(blog_draft.conclusion)} chars")
            print(f"   Word count: {blog_draft.word_count}")
            print()
            
            # Test revision functionality
            print("🔄 Testing blog draft revision...")
            revised_draft = await writing_agent.revise_blog_draft(
                original_draft=blog_draft,
                feedback="Make the introduction more engaging and add more specific statistics throughout the article.",
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Successfully revised blog draft:")
            print(f"   Revised title: {revised_draft.title}")
            print(f"   Revised word count: {revised_draft.word_count}")
            print()
            
            print("✅ Integration test completed successfully!")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_writing_agent_integration())