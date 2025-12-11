import os
import google.generativeai as genai
from brain.schemas import BrainContext

class LLMClient:
    def __init__(self, model_name: str = 'gemini-3-pro-preview'):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name) 

    def generate(self, system_prompt: str, user_content: str, temperature: float = 0.7) -> str:
        prompt = f"System: {system_prompt}\n\nUser: {user_content}"
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class BrainRegion:
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm

    def process(self, context: BrainContext) -> BrainContext:
        raise NotImplementedError
