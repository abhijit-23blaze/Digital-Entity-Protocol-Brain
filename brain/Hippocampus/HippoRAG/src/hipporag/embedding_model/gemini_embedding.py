import os
import numpy as np
from typing import List, Optional
from copy import deepcopy
from tqdm import tqdm
from google import genai

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini Embedding implementation using the google-genai SDK."""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        
        # Default to a known gemini embedding model if generic name passed
        if "gemini" in self.embedding_model_name and "/" not in self.embedding_model_name:
             self.embedding_model_name = "models/text-embedding-004"

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)

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
        # Preprocessing
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model_name,
                contents=texts,
            )
            # result.embeddings is a list of ContentEmbedding objects
            embeddings = [e.values for e in result.embeddings]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error in Gemini embedding: {e}")
            raise e

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        batch_size = params.pop("batch_size", 16)
        batch_size = min(batch_size, 100)  # Gemini API limit

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
                     raise e
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        # Normalization
        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            results = results / (norms + 1e-10)

        return results
