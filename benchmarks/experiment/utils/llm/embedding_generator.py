"""Embedding 生成工具

提供统一的 embedding 生成接口，支持 OpenAI-compatible embedding 服务。
"""

import time

import openai

EXTENDED_EMBEDDING_DIMENSIONS = {
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-small-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-zh-v1.5": 384,
}


class _OpenAIEmbedder:
    """基于 OpenAI 兼容接口的 Embedding 实现"""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "dummy")

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def get_dim(self) -> int:
        return EXTENDED_EMBEDDING_DIMENSIONS.get(self.model, 384)


def apply_embedding_model(
    name: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """创建 embedding 模型实例"""
    normalized_base_url = (base_url or "http://127.0.0.1:8890/v1").rstrip("/")
    if not normalized_base_url.endswith("/v1"):
        normalized_base_url = normalized_base_url + "/v1"
    return _OpenAIEmbedder(
        model=model or "BAAI/bge-m3",
        base_url=normalized_base_url,
        api_key=api_key or "dummy",
    )


class EmbeddingGenerator:
    """Embedding 生成器类"""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = "BAAI/bge-m3",
        api_key: str = "dummy",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if base_url:
            self.embedding_model = apply_embedding_model(
                name="openai",
                model=model_name,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self.embedding_model = None

    @classmethod
    def from_config(cls, config) -> "EmbeddingGenerator":
        base_url = config.get("runtime.embedding_base_url")
        model_name = config.get("runtime.embedding_model", "BAAI/bge-m3")
        max_retries = config.get("runtime.embedding_max_retries", 3)
        retry_delay = config.get("runtime.embedding_retry_delay", 1.0)

        return cls(
            base_url=base_url,
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def embed(self, text: str) -> list[float] | None:
        if self.embedding_model is None:
            return None

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.embedding_model.embed(text)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(
                        f"[EmbeddingGenerator] Retry {attempt + 1}/{self.max_retries} after error: {e}"
                    )
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        if self.embedding_model is None:
            return None

        if not texts:
            return []

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.embedding_model.embed_batch(texts)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    print(
                        f"[EmbeddingGenerator] Batch retry {attempt + 1}/{self.max_retries} after error: {e}"
                    )
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Batch embedding failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def is_available(self) -> bool:
        return self.embedding_model is not None
