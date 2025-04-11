import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.tools.tavily_search import TavilySearchResults

def safe_internal_lookup(query):
    result = st.session_state.chain.invoke({"question": query})
    answer = result.get("answer", "")
    if "I do not know" in answer or "context does not" in answer:
        return "No relevant internal data found."
    return answer

def generate_sales_pitch(company_name: str) -> str:
    try:
        internal_summary = safe_internal_lookup(f"What do we know about {company_name}?")
    except:
        internal_summary = "No internal data available."

    prompt = f"""
    You are preparing a 3-slide sales pitch for {company_name}.
    Use this internal knowledge (if any):  
    {internal_summary}

    ---

    Slide 1: Company overview and strategic challenges 
    Slide 2: Relevant AI/Data Science use cases tailored to their industry and pain points  
    Slide 3: Proposal summary and next steps
    """

    llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model="gpt-4o",
        temperature=0.1,
        model_kwargs={"top_p": st.session_state.top_p}
    )

    return "### 🧾 Sales Proposal\n\n" + llm.invoke(prompt).content

def is_low_relevance(source_docs):
    return all(len(doc.page_content.strip()) < 30 for doc in source_docs)

def is_question_covered_by_docs(question, source_docs):
    question_keywords = question.lower().split()[:10]
    return any(
        keyword in doc.page_content.lower()
        for keyword in question_keywords
        for doc in source_docs
    )

def get_response_from_LLM(prompt):
    try:
        # === 1. Try RAG chain ===
        rag_result = st.session_state.chain.invoke({"question": prompt})
        answer = rag_result.get("answer", "")
        source_docs = rag_result.get("source_documents", [])

        answer_text = answer.lower().strip()

        fallback_phrases = [
        "i cannot provide an answer",
        "i do not have information",
        "no relevant information",
        "i'm sorry",
        "not enough information",
        "i don’t have any information",
        "not mentioned in the provided context",
        "context does not contain",
        "there is no information",
        "i couldn't find anything",
        "i'm unable to find information",
        "not available in the context",
        "there is nothing about",
        "i was not able to find",
        "i cannot find details"
        ]

        rag_helpful = (
            source_docs and
            not any(phrase in answer_text .lower() for phrase in fallback_phrases) and
            not is_low_relevance(source_docs) and
            is_question_covered_by_docs(prompt, source_docs)
        )

        if rag_helpful:
            return f"*Sourced via document context (RAG)*\n\n{answer}", source_docs, "internal"

        # === 2. Sales Agent Fallback ===
        search_tool = TavilySearchResults(tavily_api_key=st.secrets["TAVILY_API_KEY"])

        sales_tools = [
            Tool("internal_customer_lookup", safe_internal_lookup, "Retrieve info about known clients"),
            Tool("web_search", search_tool.run, "Search online for unknown companies"),
            Tool("draft_sales_proposal", generate_sales_pitch, "Create a 3-slide sales proposal")
        ]

        sales_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a sales intelligence assistant.\n"
             "- Use internal_customer_lookup for known clients.\n"
             "- Use web_search if the client is unknown.\n"
             "- Use draft_sales_proposal for custom pitches.\n"
             "Always include tool outputs in the final answer."),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        sales_llm = ChatOpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            model="gpt-4o",
            temperature=0.1,
            model_kwargs={"top_p": st.session_state.top_p}
        )

        sales_agent = create_openai_functions_agent(sales_llm, tools=sales_tools, prompt=sales_prompt)

        sales_executor = AgentExecutor.from_agent_and_tools(
            agent=sales_agent,
            tools=sales_tools,
            verbose=True,
            output_keys=["output"]
        )

        response = sales_executor.invoke({"input": prompt})
        sales_answer = response.get("output", "")
        if "No relevant internal data" not in sales_answer:
            return f"*Using Sales Agent*\n\n{sales_answer}", [], "internal"

        # === 3. BI Agent Final Fallback ===
        bi_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                    """ You have access to two tools: \n"

                        1. internal_customer_lookup (for internal documents and data)

                        2. web_search (for publicly available online information)
                     Behavior:

                        Always attempt to retrieve information using internal_customer_lookup first.

                        If the company is found in internal sources, use only that information.

                        Clearly indicate the source as internal.

                        If no relevant internal data is found, or if the customer is unknown, use web_search as a fallback.

                        Use the web to gather information such as: Industry and sector; Key personnel; Business model and services offered; Notable customers and partnerships.
                        
                     Base all responses strictly on the tool outputs.

                        Do not assume prior knowledge about the company, unless you can find information using internal_customer_lookup tool.

                        Do not generate speculative content.

                        Indicate which tool the information came from.

                        Your goal is to return a concise and factual summary of the company’s business context, with the source of information clearly stated. Do not make up facts."""),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        bi_agent = create_openai_functions_agent(sales_llm, tools=sales_tools[:2], prompt=bi_prompt)

        bi_executor = AgentExecutor.from_agent_and_tools(
            agent=bi_agent,
            tools=sales_tools[:2],
            verbose=True,
            output_keys=["output"]
        )

        bi_result = bi_executor.invoke({"input": prompt})
        return f"*Using Business Intelligence Agent*\n\n{bi_result['output']}", [], "web"

    except Exception as e:
        st.error(f"LLM Handler Error: {str(e)}")
        return f"❌ LLM Error: {e}", []