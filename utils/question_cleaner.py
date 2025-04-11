import re

def clean_retriever_question(question: str, company_name: str) -> str:
    """
    Cleans the company name from a user question to improve semantic retrieval.
    E.g., "Tell me about our project with Boli AI" → "Tell me about our project"
    """
    if not company_name:
        return question

    # Remove company name from question (case insensitive)
    pattern = re.compile(re.escape(company_name), re.IGNORECASE)
    cleaned = pattern.sub("", question).strip()

    # Remove redundant prepositions (like "with") if left behind
    cleaned = re.sub(r"\bwith\b\s*$", "", cleaned).strip()

    return cleaned