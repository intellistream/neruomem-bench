"""NoneAction - 透传查询，不做任何处理

适用于不需要查询预处理的记忆体系统。
"""

from .base import BasePreRetrievalAction, PreRetrievalInput, PreRetrievalOutput


class NoneAction(BasePreRetrievalAction):
    """透传Action - 不做任何处理"""

    def _init_action(self) -> None:
        pass

    def execute(self, input_data: PreRetrievalInput) -> PreRetrievalOutput:
        question = input_data.question
        return PreRetrievalOutput(
            query=question,
            query_embedding=None,
            metadata={"original_query": question},
            retrieve_mode="passive",
        )
