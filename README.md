# AI Blog Generation Team

A sophisticated multi-agent system for generating high-quality blog posts using AI. The system coordinates multiple specialized agents to research, write, critique, and refine content through an iterative workflow.

## 🎯 Features

- **Multi-Agent Architecture**: Specialized agents for research, writing, critique, and orchestration
- **Iterative Quality Control**: Automatic revision cycles based on quality assessment
- **Real-time Research**: Integration with Tavily API for up-to-date information
- **Progress Tracking**: Comprehensive monitoring of workflow stages and metrics
- **Error Handling**: Robust error recovery and graceful degradation
- **Configurable Quality Thresholds**: Customizable quality standards and iteration limits

## 🏗️ Architecture

### Core Agents

1. **Research Agent** (`src/agents/research_agent.py`)
   - Conducts web research using Tavily API
   - Categorizes and scores research findings
   - Provides structured research output

2. **Writing Agent** (`src/agents/writing_agent.py`)
   - Creates initial blog drafts from research data
   - Handles revision requests with feedback
   - Ensures proper structure and word count (800-1200 words)

3. **Critique Agent** (`src/agents/critique_agent.py`)
   - Evaluates blog drafts for quality, accuracy, and engagement
   - Provides detailed feedback with severity levels
   - Makes approval/revision recommendations

4. **Orchestrator Agent** (`src/agents/orchestrator_agent.py`)
   - Coordinates the entire workflow
   - Manages agent interactions and decision making
   - Handles retry logic and error recovery

### Data Models (`src/models/data_models.py`)
- Structured data models for all agent inputs/outputs
- Progress tracking and workflow state management
- Comprehensive result aggregation

### Utilities
- **Dependencies** (`src/utils/dependencies.py`): Shared resources and configuration
- **Progress Tracking** (`src/utils/progress_tracker.py`): Workflow monitoring
- **Error Handling** (`src/utils/exceptions.py`): Custom exception types

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-blogpost-agentic-writer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or using uv
uv sync
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

### Basic Usage

```python
import asyncio
from pydantic_ai.models.openai import OpenAIModel
from tavily import TavilyClient
import httpx

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.research_agent import ResearchAgent
from src.agents.writing_agent import WritingAgent
from src.agents.critique_agent import CritiqueAgent
from src.utils.dependencies import SharedDependencies

async def generate_blog_post():
    model = OpenAIModel('gpt-4o-mini')
    
    async with httpx.AsyncClient() as http_client:
        deps = SharedDependencies(
            http_client=http_client,
            tavily_client=TavilyClient(api_key="your-tavily-key"),
            max_iterations=2,
            quality_threshold=6.0
        )
        
        # Create agents
        orchestrator = OrchestratorAgent(model)
        research_agent = ResearchAgent(model)
        writing_agent = WritingAgent(model)
        critique_agent = CritiqueAgent(model)
        
        # Generate blog post
        result = await orchestrator.generate_blog_post(
            topic="benefits of meditation for mental health",
            research_agent=research_agent,
            writing_agent=writing_agent,
            critique_agent=critique_agent,
            deps=deps
        )
        
        print(f"Generated {result.final_post.word_count}-word blog post")
        print(f"Quality score: {result.quality_score}/10")
        print(f"Revision cycles: {result.revision_count}")

# Run the example
asyncio.run(generate_blog_post())
```

## 🧪 Testing

### Run All Integration Tests
```bash
python run_tests.py
```

### Run Individual Tests
```bash
# Orchestrator integration test
python tests/test_orchestrator_integration.py

# Research agent test
python tests/test_research_integration.py

# Writing agent test
python tests/test_writing_integration.py

# Critique agent test
python tests/test_critique_integration.py

# Revision workflow test
python tests/test_revision_workflow_integration.py
```

### Unit Tests
```bash
pytest tests/ -v
```

See `tests/README.md` for detailed testing information.

## 📊 Workflow Process

1. **Research Phase**: Gather relevant information on the topic
2. **Initial Writing**: Create first draft based on research
3. **Quality Assessment**: Critique the draft for quality and accuracy
4. **Revision Decision**: Determine if revisions are needed
5. **Iterative Refinement**: Revise and re-critique until quality threshold is met
6. **Final Output**: Return completed blog post with metadata

## ⚙️ Configuration

### Quality Thresholds
- **6.0-7.0**: Acceptable quality for most use cases
- **7.0-8.0**: High quality content
- **8.0+**: Premium quality (may require multiple revisions)

### Iteration Limits
- **1-2 iterations**: Fast generation with basic quality control
- **3-5 iterations**: Balanced approach for quality content
- **5+ iterations**: Maximum quality with extended processing time

## 🔧 Development

### Project Structure
```
ai-blogpost-agentic-writer/
├── src/
│   ├── agents/          # Core agent implementations
│   ├── models/          # Data models and schemas
│   └── utils/           # Utilities and shared components
├── tests/               # Unit and integration tests
├── main.py             # Main application entry point
├── run_tests.py        # Test runner script
└── README.md           # This file
```

### Adding New Agents
1. Create agent class in `src/agents/`
2. Implement required methods and tools
3. Add data models in `src/models/data_models.py`
4. Create unit tests in `tests/`
5. Add integration tests if needed

## 📝 Example Output

The system generates comprehensive blog posts with:
- Engaging titles and introductions
- Well-structured body sections (3-5 sections)
- Evidence-based content with research backing
- Proper conclusions with actionable insights
- Word counts typically between 800-1200 words
- Quality scores and revision tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Research powered by [Tavily API](https://tavily.com/)
- Language models from [OpenAI](https://openai.com/)