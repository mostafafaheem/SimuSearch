from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage
from openrouter import OpenRouterClient

class ChatOpenRouter(BaseChatModel):
    def __init__(self, model_name: str = "gpt-4", openrouter_api_key: str = None, temperature: float = 0.1):
        self.client = OpenRouterClient(api_key=openrouter_api_key)  # instantiate the client
        self.model_name = model_name
        self.temperature = temperature

    def _call(self, messages: list[BaseMessage], stop=None) -> str:
        conv = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            conv.append({"role": role, "content": m.content})

        resp = self.client.chat.create(
            model=self.model_name,
            messages=conv,
            temperature=self.temperature
        )
        return resp.choices[0].message["content"]