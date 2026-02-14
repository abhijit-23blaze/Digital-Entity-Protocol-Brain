import os
import google.generativeai as genai
from typing import List, Tuple, Dict, Any
from copy import deepcopy
import json

from .base import BaseLLM, LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class GeminiLLM(BaseLLM):
    """Gemini LLM implementation using google.generativeai."""

    def __init__(self, global_config, **kwargs) -> None:
        super().__init__(global_config)
        self.llm_name = global_config.llm_name
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        
        self._init_llm_config()
        self.model = genai.GenerativeModel(self.llm_name)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['generate_params'] = {
            "candidate_count": config_dict.get("num_gen_choices", 1),
            "temperature": config_dict.get("temperature", 0.0),
            # Gemini uses 'max_output_tokens' instead of 'max_completion_tokens' or 'max_new_tokens'
            "max_output_tokens": config_dict.get("max_new_tokens", 400),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[List[TextChatMessage], dict]:
        # Convert HippoRAG messages (list of dicts) to Gemini format if necessary?
        # HippoRAG TextChatMessage is likely a dict or similar.
        # Check openai_gpt.py usage. It passes messages directly to openai client.
        # Gemini generate_content expects string or list of contents.
        # We need to format the chat history for Gemini.
        
        prompt = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            # Simple conversion to a single prompt string for now as HippoRAG seems to use it for single completions mostly?
            # Or if it's chat, we should structure it. 
            # Let's try to construct a prompt string compatible with Gemini's text capabilities 
            # or use chat history if we had a chat session, but here it's stateless 'infer'.
            prompt += f"{role.capitalize()}: {content}\n"
        
        # Add 'Model:' or similar if needed, or just let Gemini generate.
        prompt += "Assistant: "

        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
            
        generation_config = genai.types.GenerationConfig(
            candidate_count=params.get("candidate_count", 1),
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens", 400),
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_message = response.text
            
            # Metadata extraction
            metadata = {
                "prompt_tokens": 0, # Gemini API doesn't always return token counts easily in simple responses
                "completion_tokens": 0,
                "finish_reason": "stop" # Placeholder
            }
            if response.usage_metadata:
                metadata["prompt_tokens"] = response.usage_metadata.prompt_token_count
                metadata["completion_tokens"] = response.usage_metadata.candidates_token_count
            
            return response_message, metadata

        except Exception as e:
            logger.error(f"Error in Gemini inference: {e}")
            return "", {"error": str(e)}

