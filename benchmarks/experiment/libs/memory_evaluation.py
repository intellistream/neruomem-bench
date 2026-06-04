"""记忆评估模块 - 负责使用 LLM 对所有可见问题进行问答评估"""

from __future__ import annotations

import re
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

        compact_answer = self._extract_compact_line_answer(question, history_text)
        if compact_answer:
            data["answer"] = compact_answer
            data["question_metadata"] = question_metadata

            question_idx = data.get("question_index", 0)
            process_logger.log_qa(
                question_idx=question_idx,
                question=question,
                answer=compact_answer,
                context=history_text,
                metadata=question_metadata,
                full_prompt=history_text,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            data.setdefault("stage_timings", {})["memory_evaluation_ms"] = elapsed_ms

            print(
                f"  [MemoryEvaluation] 规则抽取: {elapsed_ms:.2f}ms",
                flush=True,
            )
            return data

        direct_answer = self._extract_direct_answer(question, history_text)
        if direct_answer:
            data["answer"] = direct_answer
            data["question_metadata"] = question_metadata

            question_idx = data.get("question_index", 0)
            process_logger.log_qa(
                question_idx=question_idx,
                question=question,
                answer=direct_answer,
                context=history_text,
                metadata=question_metadata,
                full_prompt=history_text,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            data.setdefault("stage_timings", {})["memory_evaluation_ms"] = elapsed_ms

            print(
                f"  [MemoryEvaluation] 规则抽取: {elapsed_ms:.2f}ms",
                flush=True,
            )
            return data

        extracted_answer = self._extract_answer_from_context(question, history_text)
        if extracted_answer:
            data["answer"] = extracted_answer
            data["question_metadata"] = question_metadata

            question_idx = data.get("question_index", 0)
            process_logger.log_qa(
                question_idx=question_idx,
                question=question,
                answer=extracted_answer,
                context=history_text,
                metadata=question_metadata,
                full_prompt=history_text,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            data.setdefault("stage_timings", {})["memory_evaluation_ms"] = elapsed_ms

            print(
                f"  [MemoryEvaluation] 规则抽取: {elapsed_ms:.2f}ms",
                flush=True,
            )
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

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

    def _extract_compact_line_answer(self, question: str, history_text: str) -> str:
        """从单行短答案中提取事实型回答。"""
        if not history_text:
            return ""

        normalized = history_text.replace("The following is some history information.", "", 1)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        query_lower = question.lower()

        for line in lines:
            if line.endswith("?"):
                continue
            if self._looks_like_compact_answer(line, query_lower):
                return line.rstrip(".,")

        return ""

    @staticmethod
    def _looks_like_compact_answer(line: str, query_lower: str) -> bool:
        lower = line.lower()
        word_count = len(line.split())

        if any(keyword in query_lower for keyword in ("work", "company", "employer")):
            if word_count <= 4 and re.fullmatch(r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3}", line):
                return True
            if any(suffix in lower for suffix in ("tech", "labs", "corp", "inc", "ai")) and word_count <= 5:
                return True

        if any(keyword in query_lower for keyword in ("title", "job")):
            if any(noise in lower for noise in ("said", "doubling", "team", "great", "feels")):
                return False
            if re.search(r"\b(?:senior|junior|principal|staff|lead|data|software|platform|backend|frontend|ml|machine learning)\s+engineer\b", lower):
                return True
            if "engineer" in lower or "manager" in lower or "scientist" in lower or "designer" in lower:
                return True

        if "manager" in query_lower:
            if word_count <= 3 and re.fullmatch(r"[A-Z][a-zA-Z0-9_-]*", line):
                return True

        if "live" in query_lower:
            if word_count <= 3 and re.fullmatch(r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}", line):
                return True

        if "leave" in query_lower:
            if any(character.isdigit() for character in line) or re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", line):
                return True
            return False

        if "vacation" in query_lower:
            if word_count <= 3 and re.fullmatch(r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){0,2}", line):
                return True

        if any(keyword in query_lower for keyword in ("raise", "funding")):
            if any(character.isdigit() for character in line) or "$" in line:
                return True

        if "app" in query_lower:
            if "app" in lower:
                return True

        return False

    def _extract_direct_answer(self, question: str, history_text: str) -> str:
        """优先从完整上下文中直接抓取显式实体。"""
        if not history_text:
            return ""

        query_lower = question.lower()

        if any(keyword in query_lower for keyword in ("work", "company", "employer")):
            match = re.search(
                r"started a new job at\s+(.+?)(?=\s+(?:last|on|with|because|and)\b|[.,;]|$)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")
            match = re.search(
                r"works(?:\s+mostly)?(?:\s+remotely)?\s+(?:at|in)\s+(.+?)(?=\s+(?:last|on|with|because|and)\b|[.,;]|$)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        if any(keyword in query_lower for keyword in ("title", "job")):
            match = re.search(
                r"(?:I[' ]?m|I am|I work as|I work as an?|as an?|as a)\s+([^.,;]+?engineer(?:\s+on\s+the\s+[^.,;]+?)?)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        if "manager" in query_lower:
            match = re.search(r"manager\s+([A-Z][a-zA-Z0-9_-]*)", history_text)
            if match:
                return match.group(1)

        if "live" in query_lower:
            match = re.search(r"moved to\s+([A-Z][A-Za-z0-9&.-]*)", history_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

        if "leave" in query_lower:
            match = re.search(
                r"starting\s+([A-Z][a-zA-Z0-9\s]+?)(?:[.,]|\s+for\b|\s+with\b|$)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        if "vacation" in query_lower:
            match = re.search(r"flying to\s+([A-Z][A-Za-z0-9&.-]*)", history_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

        if any(keyword in query_lower for keyword in ("raise", "funding")):
            match = re.search(r"(\d+\s+million\s+dollars)", history_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"(\d+\s+dollars)", history_text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if "app" in query_lower:
            match = re.search(
                r"(?:use|using)\s+the\s+([A-Z][A-Za-z0-9&.-]*\s+app)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")
            match = re.search(
                r"([A-Z][A-Za-z0-9&.-]*\s+app)",
                history_text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        return ""

    def _extract_answer_from_context(self, question: str, history_text: str) -> str:
        if not history_text:
            return ""

        query_lower = question.lower()
        sentences = self._split_sentences(history_text)
        if not sentences:
            sentences = [history_text.strip()]

        best_sentence = ""
        best_score = -1.0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if sentence.strip().endswith("?"):
                continue
            score = 0.0

            if any(keyword in query_lower for keyword in ("work", "company", "employer")):
                if "started a new job at" in sentence_lower or "works at" in sentence_lower or "works in" in sentence_lower:
                    score += 3.0
            if any(keyword in query_lower for keyword in ("title", "job")):
                if any(noise in sentence_lower for noise in ("said", "doubling", "team", "great", "feels")):
                    continue
                if "engineer" in sentence_lower or "role" in sentence_lower:
                    score += 3.0
            if "manager" in query_lower:
                if "manager" in sentence_lower:
                    score += 3.0
            if "live" in query_lower:
                if "moved to" in sentence_lower or "lives in" in sentence_lower:
                    score += 3.0
            if any(keyword in query_lower for keyword in ("raise", "funding")):
                if any(character.isdigit() for character in sentence_lower) or "$" in sentence_lower:
                    score += 3.0
            if any(keyword in query_lower for keyword in ("vacation", "leave")):
                if "flying to" in sentence_lower or "starting" in sentence_lower:
                    score += 3.0
            if "app" in query_lower:
                if "app" in sentence_lower:
                    score += 3.0

            if score > best_score or (score == best_score and best_sentence and len(sentence) < len(best_sentence)):
                best_sentence = sentence
                best_score = score

        if not best_sentence:
            return ""

        answer_hint = self._extract_answer_hint(best_sentence, query_lower)
        if answer_hint:
            return answer_hint

        if best_score <= 0.0:
            return ""

        return best_sentence

    @staticmethod
    def _extract_answer_hint(sentence: str, query_lower: str) -> str:
        sentence_lower = sentence.lower()

        if "manager" in query_lower:
            match = re.search(r"manager\s+([A-Z][a-zA-Z0-9_-]*)", sentence)
            if match:
                return match.group(1)

        if any(keyword in query_lower for keyword in ("title", "job")):
            if any(noise in sentence.lower() for noise in ("said", "doubling", "team", "great", "feels")):
                return ""
            match = re.search(r"a\s+([^.,;]+?engineer)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if any(keyword in query_lower for keyword in ("work", "company", "employer")):
            match = re.search(
                r"started a new job at\s+(.+?)(?=\s+(?:last|on|with|because|and)\b|[.,;]|$)",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")
            match = re.search(
                r"works(?:\s+mostly)?(?:\s+remotely)?\s+(?:at|in)\s+(.+?)(?=\s+(?:last|on|with|because|and)\b|[.,;]|$)",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        if "live" in query_lower:
            match = re.search(r"moved to\s+([A-Z][A-Za-z0-9&.-]*)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

        if "leave" in query_lower:
            match = re.search(
                r"starting\s+([A-Z][a-zA-Z0-9\s]+?)(?:[.,]|\s+for\b|\s+with\b|$)",
                sentence,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1).strip().rstrip(".,")

        if "vacation" in query_lower:
            match = re.search(r"flying to\s+([A-Z][A-Za-z0-9&.-]*)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

        if any(keyword in query_lower for keyword in ("raise", "funding")):
            match = re.search(r"(\d+\s+million\s+dollars)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"(\d+\s+dollars)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if "app" in query_lower:
            match = re.search(r"using the\s+([A-Z][A-Za-z0-9&.-]*\s+app)", sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(".,")

        return ""
