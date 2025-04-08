# EffectiveSoft
This repository contains the code for the Proof of Concept for the Chatbot


# Folder Structure

📁 EffectiveSoft_PoC/
├── app.py                  # Streamlit UI
├── chains/
│   └── rag_chain.py        # create_ConversationalRetrievalChain, memory
├── retrievers/
│   └── setup.py            # create_retriever, contextual, cohere
├── tools/
│   ├── internal_lookup.py  # safe_internal_lookup()
│   ├── sales_pitch.py      # generate_sales_pitch()
│   └── web_search.py       # wrapper for Tavily
├── agents/
│   └── sales_agent.py      # create_sales_agent()
├── utils/
│   ├── documents.py        # loaders, chunking, file handling
│   └── settings.py         # sidebar, configs, state
├── data/
│   └── tmp/                # uploaded files
