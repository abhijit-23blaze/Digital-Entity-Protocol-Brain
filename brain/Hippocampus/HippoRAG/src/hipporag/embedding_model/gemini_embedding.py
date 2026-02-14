import os
import numpy as np
from typing import List, Optional
from copy import deepcopy
from tqdm import tqdm
import google.generativeai as genai

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini Embedding implementation using google.generativeai."""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        
        # Default to a known gemini embedding model if generic name passed or if it's "gemini-embedding"
        if "gemini" in self.embedding_model_name and "/" not in self.embedding_model_name:
             self.embedding_model_name = "models/text-embedding-004"

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

        self._init_embedding_config()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {},
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "batch_size": self.global_config.embedding_batch_size,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        # Gemini batch embedding might be limited, let's process carefully.
        # genai.embed_content supports 'content' as a list.
        # "Batching is supported for up to 100 documents." - from docs, but let's be safe.
        
        # Preprocessing similar to OpenAI implementation
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=texts,
                task_type="retrieval_document", # or retrieval_query depending on context, but usually we index documents
                title=None
            )
            # result['embedding'] is a list of embeddings.
            embeddings = result['embedding']
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error in Gemini embedding: {e}")
            # Fallback or re-raise
            raise e

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)
        
        # Gemini task type adjustment if needed (instruct/query vs doc)
        # For now, simplistic approach.

        batch_size = params.pop("batch_size", 16)
        # Cap batch size to 100 for Gemini API limits if higher
        batch_size = min(batch_size, 100)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    results.append(self.encode(batch))
                except Exception as e:
                     logger.error(f"Batch encoding failed at index {i}: {e}")
                     # Try one by one if batch fails? Or just fail.
                     raise e
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        # Normalization
        if self.embedding_config.norm:
            # Avoid division by zero
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            results = results / (norms + 1e-10)

        return results
