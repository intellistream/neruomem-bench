"""EmbeddingAction - 查询向量化

将查询文本标记为需要生成 embedding，实际的 embedding 生成由 PreRetrieval 主类统一完成。
这是 PreRetrieval 阶段最常用的基础 Action。
"""

from .base import BasePreRetrievalAction, PreRetrievalInput, PreRetrievalOutput


class EmbeddingAction(BasePreRetrievalAction):
    """查询向量化Action"""

    def _init_action(self) -> None:
        self.embedding_dim: int | None = self.config.get("embedding_dim")

    def execute(self, input_data: PreRetrievalInput) -> PreRetrievalOutput:
        question = input_data.question
        return PreRetrievalOutput(
            query=question,
            query_embedding=None,
            metadata={
                "original_query": question,
                "needs_embedding": True,
            },
            retrieve_mode="passive",
        )
