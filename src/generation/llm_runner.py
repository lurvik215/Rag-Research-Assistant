import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


class LLMRunner:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, prompt: str, max_tokens: int = 512,
                 model: str = None) -> str:
        """
        Sends prompt to Groq. Uses model override if provided,
        otherwise falls back to default from config.
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
