from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from tools.internal_lookup import safe_internal_lookup
from tools.web_search import get_direct_llm_response
from tools.sales_pitch import generate_sales_pitch

def create_sales_agent(openai_api_key, model="gpt-4o", temperature=0.3):
    # Define internal lookup tool
    internal_tool = Tool(
        name="internal_customer_lookup",
        func=safe_internal_lookup,
        description="Look up customer info from internal documents."
    )

    # Define web search fallback
    web_tool = Tool(
        name="web_search",
        func=get_direct_llm_response,
        description="Fallback to LLM if internal search fails."
    )

    # Define the sales pitch generator tool (hybrid: RAG + generative)
    pitch_tool = Tool(
        name="draft_sales_proposal",
        func=generate_sales_pitch,
        description="Generates a 3-slide sales pitch for a given company using internal and generative context."
    )

    tools = [internal_tool, web_tool, pitch_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful AI sales assistant.\n"
         "You can search internal documents, use web search, and generate custom sales proposals.\n"
         "Use 'draft_sales_proposal' to create a 3-slide sales pitch."),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])

    llm = ChatOpenAI(model=model, api_key=openai_api_key, temperature=temperature)

    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        output_keys=["output"]
    )

    return executor