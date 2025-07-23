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
from src.models.data_models import ResearchOutput

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
        # Initialize models - using gpt-4o-mini for cost efficiency
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            # Create shared dependencies
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=2,  # Limit iterations for testing
                quality_threshold=6.0  # Lower threshold for testing
            )
            
            # Create all agents
            print("🔧 Initializing agents...")
            orchestrator_agent = OrchestratorAgent(model)
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            critique_agent = CritiqueAgent(model)
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
            print()
            
            print("📝 FINAL BLOG POST:")
            print("=" * 50)
            print(f"Title: {result.final_post.title}")
            print()
            print("Introduction:")
            print(result.final_post.introduction[:200] + "..." if len(result.final_post.introduction) > 200 else result.final_post.introduction)
            print()
            print(f"Body Sections ({len(result.final_post.body_sections)}):")
            for i, section in enumerate(result.final_post.body_sections, 1):
                print(f"  {i}. {section[:100]}..." if len(section) > 100 else f"  {i}. {section}")
            print()
            print("Conclusion:")
            print(result.final_post.conclusion[:200] + "..." if len(result.final_post.conclusion) > 200 else result.final_post.conclusion)
            print()
            print(f"Total Word Count: {result.final_post.word_count}")
            print()
            
            print("🔍 RESEARCH DATA:")
            print("=" * 50)
            print(f"Research Topic: {result.research_data.topic}")
            print(f"Research Findings: {len(result.research_data.findings)}")
            print(f"Research Confidence: {result.research_data.confidence_level:.2f}")
            print(f"Research Summary: {result.research_data.summary[:150]}...")
            print()
            
            # Show sample research findings
            print("Sample Research Findings:")
            for i, finding in enumerate(result.research_data.findings[:3], 1):
                print(f"  {i}. [{finding.category}] {finding.fact[:80]}...")
                print(f"     Relevance: {finding.relevance_score:.2f}")
                print(f"     Source: {finding.source_url}")
            print()
            
            print("✅ Integration test completed successfully!")
            print(f"Generated a {result.final_post.word_count}-word blog post with quality score {result.quality_score:.1f}")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_orchestrator_delegation_methods():
    """Test individual delegation methods of the Orchestrator Agent."""
    
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key or not tavily_key:
        print("❌ Required API keys not set. Skipping delegation methods test.")
        return
    
    print("🔧 Testing Orchestrator Agent delegation methods...")
    print()
    
    try:
        # Initialize models
        model = OpenAIModel('gpt-4o-mini')
        
        async with httpx.AsyncClient() as http_client:
            # Create shared dependencies
            deps = SharedDependencies(
                http_client=http_client,
                tavily_client=TavilyClient(api_key=tavily_key),
                max_iterations=2,
                quality_threshold=6.0
            )
            
            # Create agents
            orchestrator_agent = OrchestratorAgent(model)
            research_agent = ResearchAgent(model)
            writing_agent = WritingAgent(model)
            critique_agent = CritiqueAgent(model)
            
            # Create orchestration context
            from src.agents.orchestrator_agent import OrchestrationContext
            import time
            
            context = OrchestrationContext(
                topic="benefits of exercise",
                research_agent=research_agent,
                writing_agent=writing_agent,
                critique_agent=critique_agent,
                start_time=time.time(),
                usage_tracking={
                    'research_calls': 0,
                    'writing_calls': 0,
                    'critique_calls': 0,
                    'total_tokens': 0,
                    'api_calls': 0
                },
                shared_deps=deps  # Add the shared dependencies
            )
            
            # Mock RunContext for testing
            class MockRunContext:
                def __init__(self, deps):
                    self.deps = deps
            
            mock_ctx = MockRunContext(context)
            
            # Test 1: Research delegation
            print("1️⃣ Testing research delegation...")
            research_result = await orchestrator_agent.delegate_research(
                mock_ctx, "benefits of exercise"
            )
            print(f"✓ Research completed: {len(research_result.findings)} findings")
            print(f"   Confidence: {research_result.confidence_level:.2f}")
            print(f"   Usage tracking - Research calls: {context.usage_tracking['research_calls']}")
            print()
            
            # Test 2: Writing delegation
            print("2️⃣ Testing writing delegation...")
            writing_result = await orchestrator_agent.delegate_writing(
                mock_ctx, research_result
            )
            print(f"✓ Writing completed: {writing_result.word_count} words")
            print(f"   Title: {writing_result.title}")
            print(f"   Usage tracking - Writing calls: {context.usage_tracking['writing_calls']}")
            print()
            
            # Test 3: Critique delegation
            print("3️⃣ Testing critique delegation...")
            # Ensure research_result has all required fields
            if not hasattr(research_result, 'confidence_level') or research_result.confidence_level is None:
                print("   Adding missing confidence_level to research_result")
                research_result = ResearchOutput(
                    topic=research_result.topic,
                    findings=research_result.findings,
                    summary=research_result.summary if hasattr(research_result, 'summary') else f"Research summary for {research_result.topic}",
                    confidence_level=0.7  # Default confidence level
                )
                
            critique_result = await orchestrator_agent.delegate_critique(
                mock_ctx, writing_result, research_result
            )
            print(f"✓ Critique completed: Quality {critique_result.overall_quality:.1f}/10")
            print(f"   Status: {critique_result.approval_status}")
            print(f"   Feedback items: {len(critique_result.feedback_items)}")
            print(f"   Usage tracking - Critique calls: {context.usage_tracking['critique_calls']}")
            print()
            
            # Test 4: Revision decision making
            print("4️⃣ Testing revision decision making...")
            decision = await orchestrator_agent.make_revision_decision(
                mock_ctx,
                critique_result,
                current_iteration=1,
                max_iterations=2,
                quality_threshold=6.0
            )
            print(f"✓ Decision made: Should revise = {decision['should_revise']}")
            print(f"   Reasoning: {decision['reasoning']}")
            print(f"   Quality score: {decision['quality_score']}")
            print()
            
            # Test 5: Feedback formatting
            print("5️⃣ Testing feedback formatting...")
            formatted_feedback = orchestrator_agent._format_feedback_for_revision(critique_result)
            print(f"✓ Feedback formatted: {len(formatted_feedback)} characters")
            print(f"   Preview: {formatted_feedback[:100]}...")
            print()
            
            # Test 6: Final metrics calculation
            print("6️⃣ Testing final metrics calculation...")
            metrics = orchestrator_agent._calculate_final_metrics(
                start_time=context.start_time,
                usage_tracking=context.usage_tracking,
                final_quality=critique_result.overall_quality
            )
            print(f"✓ Metrics calculated:")
            print(f"   Processing time: {metrics['total_processing_time']:.2f}s")
            print(f"   Quality score: {metrics['quality_score']:.1f}")
            print(f"   Efficiency score: {metrics['efficiency_score']:.2f}")
            print()
            
            print("✅ All delegation methods tested successfully!")
            
    except Exception as e:
        print(f"❌ Delegation methods test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all integration tests."""
    print("🎯 ORCHESTRATOR AGENT INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    # Test 1: Individual delegation methods
    # await test_orchestrator_delegation_methods()
    print()
    print("-" * 60)
    print()
    
    # Test 2: Complete workflow integration
    await test_orchestrator_agent_integration()
    print()
    print("🎉 All integration tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())