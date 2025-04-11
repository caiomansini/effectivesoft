import streamlit as st
from langchain.tools import Tool
from langchain.tools.tavily_search import TavilySearchResults

def get_company_research_prompt(company_name: str) -> str:
    return f"""
You are a research assistant helping gather useful, up-to-date information about a company.

If the user asked information about {company_name} use the knowledge base to answer the question plus search the web to enrich your answer using the script below."

Search the web and return a concise but rich summary of the company with the following details:

1. **Basic Info**
   - Full company name
   - Headquarters location
   - Founding year
   - Industry/sector

2. **Business Overview**
   - Description of core products or services
   - Target market or customers
   - Mission or vision statement
   - Notable recent news (past 6–12 months)


3. **Financials (if public or known)**
   - Revenue, funding rounds, investors

4. **Technology & Innovation**
   - Mention of AI/data initiatives
   - Patents or product launches


Return the information as a structured summary (bulleted or sections), including relevant URLs where possible. Do not fabricate any data.

If some data is unavailable, just omit that section.

Company Name: {company_name}
"""

def create_company_research_tool():
    tavily = TavilySearchResults(tavily_api_key=st.secrets["TAVILY_API_KEY"])

    return Tool(
        name="company_research_web_agent",
        description="Conduct detailed web research on a company for basic info, financials, leadership, and digital presence.",
        func=lambda company_name: tavily.run(get_company_research_prompt(company_name)),
    )