import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from tools.research_agent import create_company_research_tool

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
        rag_answer = rag_result.get("answer", "")
        source_docs = rag_result.get("source_documents", [])

        rag_text = rag_answer.lower().strip()

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
            "i cannot find details",
            "i don't have that information",
            "i don't have any specific information",
            "i don't have information"
        ]

        rag_helpful = (
            source_docs and
            not any(phrase in rag_text for phrase in fallback_phrases) and
            not is_low_relevance(source_docs) and
            is_question_covered_by_docs(prompt, source_docs)
        )

        # === 2. Setup agent tools ===
        search_tool = TavilySearchResults(tavily_api_key=st.secrets["TAVILY_API_KEY"])
        from tools.research_agent import create_company_research_tool
        research_tool = create_company_research_tool()

        sales_tools = [
            Tool("internal_customer_lookup", safe_internal_lookup, "Retrieve info about known clients"),
            Tool("web_search", search_tool.run, "Search online for unknown companies"),
            Tool("draft_sales_proposal", generate_sales_pitch, "Create a 3-slide sales proposal")
        ]

        # === 3. RAG Helpful → Agent Enhancement ===
        if rag_helpful:
            # Agent to enhance existing RAG response
            enrichment_tools = sales_tools + [Tool(
                name="company_research_web_agent",
                func=research_tool.func,
                description="Use this to retrieve structured company research from the web."
            )]

            agent_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an assistant tasked with enriching internal document answers using external tools.\n"
                 "You are provided with the original user question and a RAG-based answer.\n\n"
                 "If the RAG answer is sufficient, return it as-is.\n"
                 "Otherwise, enhance it using tools like `web_search`, `company_research_web_agent`, or `draft_sales_proposal`.\n"
                 "Always include tool outputs directly in your final answer."),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])

            agent_llm = ChatOpenAI(
                api_key=st.secrets["OPENAI_API_KEY"],
                model="gpt-4o",
                temperature=0.1,
                model_kwargs={"top_p": st.session_state.top_p}
            )

            enrich_agent = create_openai_functions_agent(agent_llm, tools=enrichment_tools, prompt=agent_prompt)

            enrich_executor = AgentExecutor.from_agent_and_tools(
                agent=enrich_agent,
                tools=enrichment_tools,
                verbose=True,
                output_keys=["output"]
            )

            agent_input = f"""User question: {prompt}
RAG answer: {rag_answer}"""

            st.info("🧠 Enhancing internal answer with external tools...")
            enhanced = enrich_executor.invoke({"input": agent_input})
            return f"*RAG + Enriched Answer*\n\n{enhanced['output']}", source_docs, "internal+agent"

        # === 4. Sales Agent Fallback ===
        sales_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a sales intelligence assistant.\n"
             "- Use internal_customer_lookup for known clients.\n"
             "- Use web_search if the client is unknown.\n"
             "- Use draft_sales_proposal for custom pitches.\n"
             "Always attempt to answer using these tools."
             "If no internal information is found, you MUST try `web_search`.\n"
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
        sales_answer_text = sales_answer.lower().strip()
        fallback_detected = any(phrase in sales_answer_text for phrase in fallback_phrases)

        if not fallback_detected:
            return f"*Using Sales Agent*\n\n{sales_answer}", [], "internal"

        # === 5. BI Agent Final Fallback ===
        bi_tools = [sales_tools[0], sales_tools[1], research_tool]

        bi_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                """You have access to three tools: \n
                1. internal_customer_lookup (for internal documents and data)\n
                2. web_search (for publicly available online information)\n
                3. research_tool (for research on the company)\n

                Always attempt to retrieve information using internal_customer_lookup first.\n
                If the company is found in internal sources, use that and research_tool to gather more info.\n
                If no internal data is found or the company is unknown, use web_search and research_tool.\n
                Base all responses strictly on tool outputs. Do not make up facts.\n
                Clearly indicate the source."""),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        bi_agent = create_openai_functions_agent(sales_llm, tools=bi_tools, prompt=bi_prompt)

        bi_executor = AgentExecutor.from_agent_and_tools(
            agent=bi_agent,
            tools=bi_tools,
            verbose=True,
            output_keys=["output"]
        )

        bi_result = bi_executor.invoke({"input": prompt})
        return f"*Using Business Intelligence Agent + Web Research*\n\n{bi_result['output']}", [], "web"

    except Exception as e:
        st.error(f"LLM Handler Error: {str(e)}")
        return f"❌ LLM Error: {e}", [], "error"