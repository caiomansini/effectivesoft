from langchain.chat_models import ChatOpenAI
import streamlit as st

def generate_sales_pitch(company_name: str) -> str:
    """
    Combines internal document insights with generative logic
    to create a 3-slide sales pitch tailored to a specific company.
    """
    # Step 1: Try retrieving internal knowledge
    context = ""
    if "chain" in st.session_state:
        try:
            result = st.session_state.chain.invoke({"question": f"What do we know about {company_name}?"})
            context = result.get("answer", "")
        except Exception as e:
            context = ""

    # Step 2: Ask the LLM to write the pitch using both internal and general reasoning
    prompt = f"""
    Using the following context (if any), write a short 3-slide sales pitch for {company_name}.

    Context:
    {context if context else "No internal data available."}

    Slide 1: Who they are and what industry they're in.
    Slide 2: Relevant AI/Data Science use cases tailored to their business.
    Slide 3: A call to action and proposed next steps.

    Write in a professional tone. Include any insights from the internal context, if available.
    """

    llm = ChatOpenAI(
        model=st.session_state.selected_model,
        temperature=0.4,
        openai_api_key=st.session_state.openai_api_key,
        max_tokens=700,
        model_kwargs={"top_p": st.session_state.top_p}
    )

    return llm.invoke(prompt).content