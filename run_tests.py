#!/usr/bin/env python3
"""
Test runner for AI Blog Generation Team integration tests.

This script provides an easy way to run all integration tests.
Make sure you have the required environment variables set:
- OPENAI_API_KEY: OpenAI API key
- TAVILY_API_KEY: Tavily API key (for research functionality)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def run_test(test_module_path: str, test_name: str):
    """Run a specific test module."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    try:
        # Import and run the test module
        spec = __import__(test_module_path.replace('/', '.').replace('.py', ''))
        if hasattr(spec, 'main'):
            await spec.main()
        else:
            # Look for the main test function
            for attr_name in dir(spec):
                if attr_name.startswith('test_') and callable(getattr(spec, attr_name)):
                    test_func = getattr(spec, attr_name)
                    if asyncio.iscoroutinefunction(test_func):
                        await test_func()
                    else:
                        test_func()
                    break
        
        print(f"✅ {test_name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {test_name} failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    print("🎯 AI BLOG GENERATION TEAM - INTEGRATION TESTS")
    print("=" * 60)
    
    # Check for required environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set. Please set this environment variable.")
        return
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY not set. Please set this environment variable.")
        return
    
    print("✓ Environment variables are set")
    print()
    
    # Define tests to run
    tests = [
        ("tests.test_research_integration", "Research Agent Integration Test"),
        ("tests.test_writing_integration", "Writing Agent Integration Test"),
        ("tests.test_critique_integration", "Critique Agent Integration Test"),
        ("tests.test_revision_workflow_integration", "Revision Workflow Integration Test"),
        ("tests.test_orchestrator_integration", "Orchestrator Agent Integration Test"),
    ]
    
    # Run tests
    results = []
    for test_module, test_name in tests:
        success = await run_test(test_module, test_name)
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())