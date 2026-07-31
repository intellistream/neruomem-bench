from __future__ import annotations

import pytest

from benchmarks.experiment.libs import memory_evaluation


class _RecordingGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "model answer"


@pytest.mark.parametrize(
    ("question", "history"),
    [
        (
            "Where does Alice work?",
            "Alice started a new job at InnovateTech last Monday.",
        ),
        (
            "Who is Alice's manager?",
            "The following is some history information.\nEmma",
        ),
        (
            "What is Alice's job title?",
            "Alice is a senior data engineer.",
        ),
        (
            "When does Alice leave for her vacation?",
            "Alice is flying to Tokyo on April 20th.",
        ),
        (
            "How much funding did Alice raise?",
            "Alice raised 3 million dollars.",
        ),
    ],
)
def test_memory_evaluation_never_uses_question_specific_answer_rules(
    monkeypatch,
    question,
    history,
):
    generator = _RecordingGenerator()
    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: generator),
    )
    evaluator = memory_evaluation.MemoryEvaluation(
        {"runtime.prompt_template": "Question: {question}\nAnswer:"}
    )

    result = evaluator.execute(
        {
            "question": question,
            "history_text": history,
            "question_metadata": {},
            "question_index": 1,
        }
    )

    assert result["answer"] == "model answer"
    assert len(generator.prompts) == 1
    assert history in generator.prompts[0]
    assert question in generator.prompts[0]
