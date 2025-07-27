#!/usr/bin/env python3
"""
Integration test for Critique Agent.

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
import httpx

from src.agents.critique_agent import CritiqueAgent
from src.models.data_models import (
    BlogDraft, 
    ResearchOutput, 
    ResearchFinding,
    CritiqueSeverity
)
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


async def test_critique_agent_integration():
    """Test Critique Agent integration with real models."""
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Skipping integration test.")
        return
    
    print("🔍 Starting Critique Agent integration test...")
    
    try:
        # Create model and dependencies
        model = OpenAIModel("gpt-4o-mini")  # Use cost-effective model for testing
        
        async with httpx.AsyncClient() as http_client:
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=None,  # Not needed for critique tests
                max_iterations=3,
                quality_threshold=7.0
            )
            
            # Create Critique Agent
            critique_agent = CritiqueAgent(model)
            print("✓ Critique agent created successfully")
            
            # Create sample research data
            sample_findings = [
                ResearchFinding(
                    fact="Intermittent fasting can reduce body weight by 3-8% over 3-24 weeks according to multiple studies",
                    source_url="https://example.com/study1",
                    relevance_score=0.9,
                    category="statistic"
                ),
                ResearchFinding(
                    fact="Dr. Jason Fung, a leading researcher in metabolic health, recommends 16:8 intermittent fasting for beginners",
                    source_url="https://example.com/expert1",
                    relevance_score=0.8,
                    category="expert_opinion"
                ),
                ResearchFinding(
                    fact="Clinical studies demonstrate that intermittent fasting improves insulin sensitivity",
                    source_url="https://example.com/study2",
                    relevance_score=0.85,
                    category="study"
                )
            ]
            
            research_output = ResearchOutput(
                topic="intermittent fasting benefits",
                findings=sample_findings,
                summary="Intermittent fasting shows promising results for weight loss and metabolic health improvements.",
                confidence_level=0.85
            )
            
            # Create a sample blog draft to critique
            sample_blog_draft = BlogDraft(
                title="The Complete Guide to Intermittent Fasting: Benefits and Getting Started",
                introduction="Intermittent fasting has gained tremendous popularity in recent years as a powerful tool for weight loss and improved health. This comprehensive guide will explore the science-backed benefits, different methods, and practical tips for getting started with intermittent fasting safely and effectively.",
                body_sections=[
                    "Research consistently shows that intermittent fasting can lead to significant weight loss, with studies demonstrating 3-8% body weight reduction over 3-24 weeks. Dr. Jason Fung, a leading expert in metabolic health, emphasizes that intermittent fasting works by allowing insulin levels to drop, enabling the body to access stored fat for energy.",
                    "The most popular and beginner-friendly approach is the 16:8 method, which involves fasting for 16 hours and eating within an 8-hour window each day. For example, you might eat between 12 PM and 8 PM, then fast from 8 PM until 12 PM the next day.",
                    "While intermittent fasting offers many benefits, it's important to be aware of potential side effects during the initial adaptation period. Some people may experience fatigue, headaches, or difficulty concentrating as their body adjusts to the new eating pattern."
                ],
                conclusion="Intermittent fasting can be an effective tool for weight loss and improved metabolic health when implemented properly. Start with the 16:8 method, stay consistent, and pay attention to how your body responds. If you have any health conditions, consult with a healthcare professional before starting.",
                word_count=650
            )
            
            print(f"📝 Sample blog draft created:")
            print(f"   Title: {sample_blog_draft.title}")
            print(f"   Word count: {sample_blog_draft.word_count}")
            print(f"   Body sections: {len(sample_blog_draft.body_sections)}")
            print()
            
            # Test critique functionality
            print("🔍 Testing blog draft critique...")
            critique_result = await critique_agent.critique_blog_draft(
                blog_draft=sample_blog_draft,
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Successfully critiqued blog draft:")
            print(f"   Overall quality score: {critique_result.overall_quality}/10")
            print(f"   Approval status: {critique_result.approval_status}")
            print(f"   Number of feedback items: {len(critique_result.feedback_items)}")
            print(f"   Summary feedback: {critique_result.summary_feedback}")
            print()
            
            # Display detailed feedback
            if critique_result.feedback_items:
                print("📋 Detailed feedback:")
                for i, feedback in enumerate(critique_result.feedback_items[:3], 1):  # Show first 3 items
                    severity_emoji = {
                        CritiqueSeverity.MINOR: "🟡",
                        CritiqueSeverity.MODERATE: "🟠", 
                        CritiqueSeverity.MAJOR: "🔴"
                    }.get(feedback.severity, "⚪")
                    
                    print(f"   {i}. {severity_emoji} [{feedback.section}] {feedback.severity.upper()}")
                    print(f"      Issue: {feedback.issue}")
                    print(f"      Suggestion: {feedback.suggestion}")
                    print()
            
            print("✅ Integration test completed successfully!")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_critique_agent_integration())