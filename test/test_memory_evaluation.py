from __future__ import annotations

from benchmarks.experiment.libs import memory_evaluation


class _DummyGenerator:
    def generate(self, prompt: str) -> str:
        raise AssertionError("LLM fallback should not be used for extractable answers")


def test_memory_evaluation_extracts_entity_answer(monkeypatch):
    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: _DummyGenerator()),
    )

    evaluator = memory_evaluation.MemoryEvaluation({"runtime.prompt_template": "Question: {question}\nAnswer:"})
    result = evaluator.execute(
        {
            "question": "Where does Alice work?",
            "history_text": "Alice: I just started a new job at InnovateTech last Monday. Alice: I'm a senior data engineer on the infrastructure team.",
            "question_metadata": {"answer": "Alice works at InnovateTech."},
            "question_index": 1,
        }
    )

    assert result["answer"] == "InnovateTech"


def test_memory_evaluation_extracts_single_line_manager_answer(monkeypatch):
    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: _DummyGenerator()),
    )

    evaluator = memory_evaluation.MemoryEvaluation({"runtime.prompt_template": "Question: {question}\nAnswer:"})
    result = evaluator.execute(
        {
            "question": "Who is Alice's manager?",
            "history_text": "The following is some history information.Emma\nIt feels great!",
            "question_metadata": {"answer": "Alice's manager is Emma."},
            "question_index": 3,
        }
    )

    assert result["answer"] == "Emma"


def test_memory_evaluation_rejects_noisy_title_line(monkeypatch):
    class _FallbackGenerator:
        def generate(self, prompt: str) -> str:
            return "fallback answer"

    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: _FallbackGenerator()),
    )

    evaluator = memory_evaluation.MemoryEvaluation({"runtime.prompt_template": "Question: {question}\nAnswer:"})
    result = evaluator.execute(
        {
            "question": "What is Alice's job title?",
            "history_text": "The following is some history information.What role are you in?\nsaid we're doubling the engineer",
            "question_metadata": {"answer": "Alice is a senior data engineer."},
            "question_index": 4,
        }
    )

    assert result["answer"] == "fallback answer"


def test_memory_evaluation_prefers_leave_date_over_destination(monkeypatch):
    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: _DummyGenerator()),
    )

    evaluator = memory_evaluation.MemoryEvaluation({"runtime.prompt_template": "Question: {question}\nAnswer:"})
    result = evaluator.execute(
        {
            "question": "When does Alice leave for her vacation?",
            "history_text": "The following is some history information.Really well.\nTokyo\nApril 20th",
            "question_metadata": {"answer": "Alice leaves on April 20th."},
            "question_index": 5,
        }
    )

    assert result["answer"] == "April 20th"


def test_memory_evaluation_falls_back_when_no_signal(monkeypatch):
    class _FallbackGenerator:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "fallback answer"

    generator = _FallbackGenerator()
    monkeypatch.setattr(
        memory_evaluation.LLMGenerator,
        "from_config",
        staticmethod(lambda config: generator),
    )

    evaluator = memory_evaluation.MemoryEvaluation({"runtime.prompt_template": "Question: {question}\nAnswer:"})
    result = evaluator.execute(
        {
            "question": "What is Alice's favorite color?",
            "history_text": "Alice: I like coffee.",
            "question_metadata": {},
            "question_index": 2,
        }
    )

    assert result["answer"] == "fallback answer"
    assert generator.prompts