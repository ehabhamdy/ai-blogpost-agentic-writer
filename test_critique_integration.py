#!/usr/bin/env python3
"""
Integration test for Critique Agent to verify it works with Pydantic AI.

Required environment variables:
- OPENAI_API_KEY: OpenAI API key
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.critique_agent import CritiqueAgent
from src.models.data_models import (
    BlogDraft, 
    ResearchOutput, 
    ResearchFinding,
    CritiqueOutput,
    CritiqueSeverity
)
from src.utils.dependencies import SharedDependencies

# Load environment variables
load_dotenv()


async def test_critique_agent_integration():
    """Test Critique Agent integration with real models."""
    
    # Skip if no API keys available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Skipping integration test - no OPENAI_API_KEY found")
        return
    
    print("🔍 Starting Critique Agent integration test...")
    
    try:
        # Create model and dependencies
        model = OpenAIModel("gpt-4o-mini")  # Use cheaper model for testing
        
        async with httpx.AsyncClient() as http_client:
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key="dummy"),  # Won't be used in this test
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
                    fact="Clinical studies demonstrate that intermittent fasting improves insulin sensitivity and reduces inflammation markers",
                    source_url="https://example.com/study2",
                    relevance_score=0.85,
                    category="study"
                ),
                ResearchFinding(
                    fact="The 16:8 method involves fasting for 16 hours and eating within an 8-hour window daily",
                    source_url="https://example.com/method1",
                    relevance_score=0.7,
                    category="general_fact"
                ),
                ResearchFinding(
                    fact="Some individuals may experience fatigue, headaches, or difficulty concentrating during the initial adaptation period",
                    source_url="https://example.com/risks1",
                    relevance_score=0.6,
                    category="risk"
                )
            ]
            
            research_output = ResearchOutput(
                topic="intermittent fasting",
                findings=sample_findings,
                summary="Intermittent fasting shows promising results for weight loss and metabolic health, with the 16:8 method being most popular for beginners",
                confidence_level=0.85
            )
            
            # Create a sample blog draft to critique
            sample_blog_draft = BlogDraft(
                title="The Complete Guide to Intermittent Fasting: Benefits, Methods, and Getting Started",
                introduction="Intermittent fasting has gained tremendous popularity in recent years as a powerful tool for weight loss and improved health. This comprehensive guide will explore the science-backed benefits, different methods, and practical tips for getting started with intermittent fasting safely and effectively.",
                body_sections=[
                    "Research consistently shows that intermittent fasting can lead to significant weight loss, with studies demonstrating 3-8% body weight reduction over 3-24 weeks. Dr. Jason Fung, a leading expert in metabolic health, emphasizes that intermittent fasting works by allowing insulin levels to drop, enabling the body to access stored fat for energy. Clinical studies have also shown improvements in insulin sensitivity and reductions in inflammation markers.",
                    "The most popular and beginner-friendly approach is the 16:8 method, which involves fasting for 16 hours and eating within an 8-hour window each day. For example, you might eat between 12 PM and 8 PM, then fast from 8 PM until 12 PM the next day. This method is sustainable because it typically just involves skipping breakfast and having your first meal at lunch. Other methods include the 5:2 approach, where you eat normally five days a week and restrict calories on two non-consecutive days.",
                    "While intermittent fasting offers many benefits, it's important to be aware of potential side effects, especially during the initial adaptation period. Some people may experience fatigue, headaches, or difficulty concentrating as their body adjusts to the new eating pattern. These symptoms typically resolve within a few weeks. It's crucial to stay hydrated, get adequate sleep, and listen to your body. Certain individuals, including pregnant women, people with diabetes, or those with a history of eating disorders, should consult with a healthcare provider before starting intermittent fasting."
                ],
                conclusion="Intermittent fasting can be an effective tool for weight loss and improved metabolic health when implemented properly. Start with the 16:8 method, stay consistent, and pay attention to how your body responds. Remember that sustainable results come from long-term lifestyle changes, not quick fixes. If you're considering intermittent fasting, especially if you have any health conditions, consult with a healthcare professional to ensure it's right for you.",
                word_count=850
            )
            
            print(f"📝 Sample blog draft created:")
            print(f"   Title: {sample_blog_draft.title}")
            print(f"   Word count: {sample_blog_draft.word_count}")
            print(f"   Body sections: {len(sample_blog_draft.body_sections)}")
            
            # Test critique functionality
            print("\n🔍 Testing blog draft critique...")
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
            
            # Display detailed feedback
            if critique_result.feedback_items:
                print("\n📋 Detailed feedback:")
                for i, feedback in enumerate(critique_result.feedback_items[:5], 1):  # Show first 5 items
                    severity_emoji = {
                        CritiqueSeverity.MINOR: "🟡",
                        CritiqueSeverity.MODERATE: "🟠", 
                        CritiqueSeverity.MAJOR: "🔴"
                    }.get(feedback.severity, "⚪")
                    
                    print(f"   {i}. {severity_emoji} [{feedback.section}] {feedback.severity.upper()}")
                    print(f"      Issue: {feedback.issue}")
                    print(f"      Suggestion: {feedback.suggestion}")
                    print()
            
            # Test with a lower quality draft to see different feedback
            print("\n🔍 Testing with a lower quality draft...")
            low_quality_draft = BlogDraft(
                title="Fasting",
                introduction="Fasting is good.",
                body_sections=[
                    "People fast and lose weight.",
                    "There are different ways to fast."
                ],
                conclusion="Try fasting.",
                word_count=25
            )
            
            low_quality_critique = await critique_agent.critique_blog_draft(
                blog_draft=low_quality_draft,
                research_data=research_output,
                deps=deps
            )
            
            print(f"✅ Low quality draft critique completed:")
            print(f"   Overall quality score: {low_quality_critique.overall_quality}/10")
            print(f"   Approval status: {low_quality_critique.approval_status}")
            print(f"   Number of feedback items: {len(low_quality_critique.feedback_items)}")
            print(f"   Summary: {low_quality_critique.summary_feedback[:100]}...")
            
            print("\n✅ Integration test completed successfully!")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_critique_tools_integration():
    """Test individual critique tools with real model."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Skipping tools integration test - no OPENAI_API_KEY found")
        return
    
    print("\n🔧 Testing individual critique tools...")
    
    try:
        model = OpenAIModel("gpt-4o-mini")
        critique_agent = CritiqueAgent(model)
        
        # Sample data for testing tools
        sample_draft = BlogDraft(
            title="The Science Behind Intermittent Fasting",
            introduction="Intermittent fasting has become increasingly popular. This article explores the research behind this eating pattern.",
            body_sections=[
                "Studies show intermittent fasting can help with weight loss. Research indicates it may improve metabolic health.",
                "The 16:8 method is popular among beginners. It involves eating within an 8-hour window."
            ],
            conclusion="Intermittent fasting shows promise for health improvement. Consider consulting a doctor before starting.",
            word_count=200
        )
        
        sample_research = ResearchOutput(
            topic="intermittent fasting",
            findings=[
                ResearchFinding(
                    fact="Intermittent fasting reduces body weight by 3-8%",
                    source_url="https://example.com/study",
                    relevance_score=0.9,
                    category="statistic"
                )
            ],
            summary="Research shows benefits of intermittent fasting",
            confidence_level=0.8
        )
        
        # Test individual helper methods (these don't require API calls)
        print("🔧 Testing clarity analysis methods...")
        title_analysis = critique_agent._analyze_title_clarity(sample_draft.title)
        print(f"   Title clarity score: {title_analysis['clarity_score']:.2f}")
        
        section_analysis = critique_agent._analyze_section_clarity(sample_draft.introduction, "introduction")
        print(f"   Introduction clarity score: {section_analysis['clarity_score']:.2f}")
        
        readability = critique_agent._calculate_overall_readability(sample_draft)
        print(f"   Overall readability score: {readability['readability_score']:.2f}")
        
        print("🔧 Testing fact verification methods...")
        supported_claims = critique_agent._identify_supported_claims(
            sample_draft.introduction + " " + " ".join(sample_draft.body_sections),
            sample_research.findings
        )
        print(f"   Supported claims found: {len(supported_claims)}")
        
        attribution = critique_agent._check_source_attribution(
            sample_draft.introduction + " " + " ".join(sample_draft.body_sections),
            sample_research.findings
        )
        print(f"   Attribution score: {attribution['attribution_score']:.2f}")
        
        print("🔧 Testing structure assessment methods...")
        intro_assessment = critique_agent._assess_introduction(sample_draft.introduction)
        print(f"   Introduction effectiveness: {intro_assessment['effectiveness_score']:.2f}")
        
        body_assessment = critique_agent._assess_body_organization(sample_draft.body_sections)
        print(f"   Body organization score: {body_assessment['organization_score']:.2f}")
        
        conclusion_assessment = critique_agent._assess_conclusion(sample_draft.conclusion)
        print(f"   Conclusion effectiveness: {conclusion_assessment['effectiveness_score']:.2f}")
        
        print("✅ Tools integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Tools integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_critique_agent_integration())
    asyncio.run(test_critique_tools_integration())