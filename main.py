from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


# read model key from .env file 
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))

    agent = Agent(  
        model,
        system_prompt='Be concise, reply with one sentence.',  
    )

    result = agent.run_sync('Where does "hello world" come from?')  
    print(result.output)



if __name__ == "__main__":
    main()


