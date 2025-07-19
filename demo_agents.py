#!/usr/bin/env python3
"""
Demo script showing Research Agent and Writing Agent working together.

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
from src.agents.writing_agent import WritingAgent
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


async def demo_agents_workflow():
    """Demonstrate the complete workflow from research to blog post."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set. Please set it in your .env file.")
        return
    
    print("🚀 Starting AI Blog Generation Team Demo...")
    print("=" * 60)
    
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
            
            # Create agents
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            print("✓ Agents initialized successfully")
            
            # Step 1: Research
            topic = "benefits of meditation for mental health"
            print(f"\n🔍 STEP 1: Researching '{topic}'...")
            
            research_result = await research_agent.research_topic(topic, deps)
            
            print(f"✓ Research completed!")
            print(f"  - Found {len(research_result.findings)} research findings")
            print(f"  - Confidence level: {research_result.confidence_level:.2f}")
            print(f"  - Summary: {research_result.summary}")
            
            # Show top findings
            print(f"\n📊 Top Research Findings:")
            for i, finding in enumerate(research_result.findings[:3]):
                print(f"  {i+1}. [{finding.category.upper()}] {finding.fact[:80]}...")
                print(f"     Relevance: {finding.relevance_score:.2f} | Source: {finding.source_url}")
            
            # Step 2: Writing
            print(f"\n✍️  STEP 2: Creating blog post...")
            
            blog_draft = await writing_agent.create_blog_draft(
                topic=topic,
                research_data=research_result,
                deps=deps
            )
            
            print(f"✓ Blog post created!")
            print(f"  - Title: {blog_draft.title}")
            print(f"  - Word count: {blog_draft.word_count}")
            print(f"  - Body sections: {len(blog_draft.body_sections)}")
            
            # Display the blog post
            print(f"\n📝 GENERATED BLOG POST:")
            print("=" * 60)
            print(f"# {blog_draft.title}")
            print()
            print("## Introduction")
            print(blog_draft.introduction)
            print()
            
            for i, section in enumerate(blog_draft.body_sections, 1):
                print(f"## Section {i}")
                print(section)
                print()
            
            print("## Conclusion")
            print(blog_draft.conclusion)
            print()
            print("=" * 60)
            
            # Step 3: Demonstrate revision
            print(f"\n🔄 STEP 3: Demonstrating revision capability...")
            
            feedback = "Make the introduction more engaging and add specific statistics about meditation benefits"
            
            revised_draft = await writing_agent.revise_blog_draft(
                original_draft=blog_draft,
                feedback=feedback,
                research_data=research_result,
                deps=deps
            )
            
            print(f"✓ Blog post revised!")
            print(f"  - Original word count: {blog_draft.word_count}")
            print(f"  - Revised word count: {revised_draft.word_count}")
            print(f"  - Feedback applied: {feedback}")
            
            print(f"\n📝 REVISED INTRODUCTION:")
            print("-" * 40)
            print(revised_draft.introduction)
            print("-" * 40)
            
            print(f"\n✅ Demo completed successfully!")
            print(f"The AI Blog Generation Team successfully:")
            print(f"  1. Researched the topic and found {len(research_result.findings)} relevant findings")
            print(f"  2. Created a {blog_draft.word_count}-word blog post")
            print(f"  3. Revised the content based on feedback")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(demo_agents_workflow())