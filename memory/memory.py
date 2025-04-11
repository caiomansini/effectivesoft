import tiktoken
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory
)
from langchain.chat_models import ChatOpenAI


class PatchedSummaryMemory(ConversationSummaryBufferMemory):
    """
    Custom memory class that overrides LangChain's broken token counting
    by using tiktoken directly (works with GPT-4o and cl100k_base).
    """
    def get_num_tokens(self, messages):
        encoding = tiktoken.encoding_for_model(self.llm.model_name)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # per-message overhead
            for key, value in message.dict().items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
        num_tokens += 2  # priming
        return num_tokens


def create_memory(model_name="gpt-4o", memory_max_token=4096, api_key=None):
    """
    Dynamically selects the most appropriate memory class based on model support.
    - Uses token-aware PatchedSummaryMemory for gpt-3.5 / gpt-4
    - Falls back to ConversationBufferMemory for unsupported models (or you can reverse this)
    """
    summary_safe_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

    if model_name in summary_safe_models:
        return PatchedSummaryMemory(
            llm=ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=0.1
            ),
            max_token_limit=memory_max_token,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
    else:
        return ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )