"""记忆评估模块 - 负责使用 LLM 对所有可见问题进行问答评估"""

import time

from sage.foundation import MapFunction

from benchmarks.experiment.utils import LLMGenerator, process_logger


class MemoryEvaluation(MapFunction):
    """记忆评估算子"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.get("runtime.dataset", "locomo")
        self.question_answer_prompt = self.config.get(
            "runtime.prompt_template",
            """Based on the above context, answer the following question concisely using exact words from the context whenever possible. If the information is not mentioned in the conversation, respond with "Not mentioned in the conversation".

Question: {question}
Answer:""",
        )
        self.generator = LLMGenerator.from_config(config)

    def execute(self, data):
        start_time = time.perf_counter()
        if not data:
            return None

        question = data.get("question")
        history_text = data.get("history_text", "")
        question_metadata = data.get("question_metadata", {})

        if not question:
            data["answer"] = None
            return data

        full_prompt = history_text
        if full_prompt:
            full_prompt += "\n\n"
        question_prompt = self.question_answer_prompt.replace("{question}", question)
        full_prompt += question_prompt

        llm_start = time.perf_counter()
        answer_text = self.generator.generate(full_prompt)
        llm_elapsed = (time.perf_counter() - llm_start) * 1000

        data["answer"] = answer_text
        data["question_metadata"] = question_metadata

        question_idx = data.get("question_index", 0)
        process_logger.log_qa(
            question_idx=question_idx,
            question=question,
            answer=answer_text,
            context=history_text,
            metadata=question_metadata,
            full_prompt=full_prompt,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        data.setdefault("stage_timings", {})["memory_evaluation_ms"] = elapsed_ms

        print(
            f"  [MemoryEvaluation] LLM: {llm_elapsed:.2f}ms | 总耗时: {elapsed_ms:.2f}ms",
            flush=True,
        )

        return data
