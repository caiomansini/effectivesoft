## EffectiveSoft
This repository contains the code for the Proof of Concept for the Chatbot

# Problem Statement
Sales support bot
The company requesting the project could be either known or unknown to us. As part of preparation, the company wants to develop a Chatbot that is capable of:

Known Companies:
1. Collect general details about the company
2. Collect the contacts of people who have worked with the company
3. Collect an overview of our previous collaboration experience with this company

Unknown Companies:
1. Collect general details about the company
2. Be able to submit an opportunity with known details
3. Be able to ask the system to draft a proposal presentation with relevant use cases included into it

# Folder Structure

``` bash
📁 effective_soft_chatbot/
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
```

# Instructions
1. Create a virtual environment: python -m venv effectivesoft_chatbot
2. Activate the virtual env
3. Enter the directory
4. Install all the required dependencies
5. Change the API KEYS in the .env file
6. Start the app: streamlit run app.py
7. Enjoy

