import os
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


class LLMRunner:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Sends prompt to Groq and returns the response text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
