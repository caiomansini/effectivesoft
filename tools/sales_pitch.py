from langchain.chat_models import ChatOpenAI
from tools.internal_lookup import safe_internal_lookup
import streamlit as st


def generate_sales_pitch(company_name: str) -> str:
    # Try to get internal knowledge from the retriever first
    try:
        internal_summary = safe_internal_lookup(f"What do we know about {company_name}?")
    except:
        internal_summary = "No internal data available."

    prompt = f"""
        You are preparing a 3-slide sales pitch for {company_name}.
        Optionally uses a reference company (e.g., a known project) as inspiration.

        Use this internal knowledge (if any):  
        {internal_summary}

        ---

        Slide 1: Company overview and strategic challenges 
        Slide 2: Relevant AI/Data Science use cases tailored to their industry and pain points.  
        Slide 3: Proposal summary and next steps

        Tailor suggestions to the assumed industry and needs of the target company. Use a professional tone.
        """

    llm = ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        model='gpt-4o',
        temperature=0.1,
        model_kwargs={"top_p": st.session_state.top_p},
    )
    return "### 🧾 Sales Proposal\n\n" + llm.invoke(prompt).content