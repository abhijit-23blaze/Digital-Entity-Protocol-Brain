import os
from google import genai
from google.genai import types
from typing import List, Tuple
from copy import deepcopy

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class GeminiLLM(BaseLLM):
    """Gemini LLM implementation using the google-genai SDK."""

    def __init__(self, global_config, **kwargs) -> None:
        super().__init__(global_config)
        self.llm_name = global_config.llm_name
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        
        self._init_llm_config()

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['generate_params'] = {
            "candidate_count": config_dict.get("num_gen_choices", 1),
            "temperature": config_dict.get("temperature", 1.0),
            "max_output_tokens": config_dict.get("max_new_tokens", 400),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict]:
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant: "

        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)

        config = types.GenerateContentConfig(
            candidate_count=params.get("candidate_count", 1),
            temperature=params.get("temperature", 1.0),
            max_output_tokens=params.get("max_output_tokens", 400),
        )

        try:
            response = self.client.models.generate_content(
                model=self.llm_name,
                contents=prompt,
                config=config,
            )
            
            response_message = response.text
            
            # Metadata extraction
            metadata = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop"
            }
            if response.usage_metadata:
                metadata["prompt_tokens"] = response.usage_metadata.prompt_token_count
                metadata["completion_tokens"] = response.usage_metadata.candidates_token_count
            
            return response_message, metadata

        except Exception as e:
            logger.error(f"Error in Gemini inference: {e}")
            return "", {"error": str(e)}
