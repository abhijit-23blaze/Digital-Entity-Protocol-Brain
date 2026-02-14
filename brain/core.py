import os
from google import genai
from google.genai import types
from brain.schemas import BrainContext

class LLMClient:
    """Wrapper around a Gemini model via the google-genai SDK.
    
    Args:
        model_name: Gemini model identifier (e.g. 'gemini-3-pro-preview').
        thinking: Thinking level — 'low', 'high', or None for model default.
                  'high' → deep reasoning (Left Hemisphere / logic).
                  'low'  → suppress reasoning (Right Hemisphere / creative).
                  None   → model default (high for Gemini 3).
    """
    def __init__(self, model_name: str = 'gemini-3-pro-preview', thinking: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking = thinking  # 'low', 'high', or None

    def generate(self, system_prompt: str, user_content: str, temperature: float = 1.0) -> str:
        """Generate content using the Gemini API.
        
        Note: Gemini 3 recommends temperature=1.0 (the default).
        """
        prompt = f"System: {system_prompt}\n\nUser: {user_content}"
        try:
            config_kwargs = {"temperature": temperature}

            # Apply thinking configuration if set
            if self.thinking:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
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
