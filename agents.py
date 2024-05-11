import os
from langchain.chains import LLMChain
from langchain_community.llms import Anyscale
from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper

os.environ["ANYSCALE_API_KEY"] = "esecret_r1u6kcke1j42yfhmt1kjdh5pv1"
llm = Anyscale(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")

wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to get information from wikipedia about a single topic"
    ),
]


agent_executor = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=
                                  True)

output = agent_executor.run("Can you please provide a quick summary of Napoleon Bonaparte? \
                          Then do a separate search and tell me what the commonalities are with Serena Williams")

print (output)