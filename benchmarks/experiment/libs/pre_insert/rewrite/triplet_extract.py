"""TripleExtractAction - 三元组提取策略

从对话中提取三元组（subject-predicate-object），并重构为适合检索的自然语言形式。
属于 Extract 类别，适用于 TiM、HippoRAG 等基于知识图谱的记忆体。

TiM 论文核心思路:
    - 只存储三元组，不保留原始对话（keep_original=false）
    - 使用 LSH 哈希索引进行近似文本检索
    - 结合语义巩固（post_insert）对冲突记忆去重合并

使用示例:
    pre_insert:
      action: "extract.triple"
      extraction_method: "llm"      # llm / simple
      max_triplets: 10
      keep_original: false
      triple_extraction_prompt: |   # 详见 tim_locomo_pipeline.yaml
        ...
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from ..base import BasePreInsertAction, PreInsertInput, PreInsertOutput


class TripleExtractAction(BasePreInsertAction):
    """三元组提取 Action

    处理流程:
    1. 格式化对话为纯文本
    2. 通过 LLM 或启发式规则提取 (subject, predicate, object) 三元组
    3. 将每条三元组重构为自然语言字符串并作为独立记忆条目返回
    """

    def _init_action(self) -> None:
        self.extraction_method: str = self.config.get("extraction_method", "simple")
        self.max_triplets: int = self.config.get("max_triplets", 10)
        self.reconstruct_template: str = self.config.get(
            "reconstruct_template", "{subject} {predicate} {object}"
        )
        self.keep_original: bool = self.config.get("keep_original", True)
        self.triple_extraction_prompt: str = self.config.get(
            "triple_extraction_prompt", ""
        )
        self.llm_generator = None

    # ── External setters called by operator.py ────────────────────────────────

    def set_llm_generator(self, llm_generator: Any) -> None:
        self.llm_generator = llm_generator

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute(self, input_data: PreInsertInput) -> PreInsertOutput:
        dialogs = input_data.data.get("dialogs", [])
        text = self._format_dialogue(dialogs)

        has_llm = self.llm_generator is not None
        has_prompt = bool(self.triple_extraction_prompt)
        print(
            f"[TripleExtract] method={self.extraction_method} "
            f"has_llm={has_llm} has_prompt={has_prompt} "
            f"max_triplets={self.max_triplets}"
        )
        if self.extraction_method != "llm" and has_llm and has_prompt:
            print(
                "[TripleExtract] Tip: set extraction_method='llm' to enable LLM extraction"
            )

        triplets = self._extract_triplets(text)
        reconstructed_texts = [self._reconstruct_triplet(t) for t in triplets]

        entries: list[dict[str, Any]] = []

        if self.keep_original:
            original_entry: dict[str, Any] = {
                "text": text,
                "metadata": {
                    "action": "extract.triple",
                    "type": "original",
                    "triplet_count": len(triplets),
                },
            }
            original_entry = self._set_default_fields(original_entry)
            original_entry["insert_method"] = "triple_extract_original"
            entries.append(original_entry)

        for i, (triplet, reconstructed) in enumerate(
            zip(triplets, reconstructed_texts)
        ):
            entry: dict[str, Any] = {
                "text": reconstructed,
                "triplet": triplet,
                "reconstructed_text": reconstructed,
                "metadata": {
                    "action": "extract.triple",
                    "type": "triplet",
                    "triplet_index": i,
                    "subject": triplet["subject"],
                    "predicate": triplet["predicate"],
                    "object": triplet["object"],
                },
                "insert_params": {
                    "entities": [triplet["subject"], triplet["object"]],
                    "relations": [
                        (triplet["subject"], triplet["predicate"], triplet["object"])
                    ],
                },
            }
            entry = self._set_default_fields(entry)
            entry["insert_method"] = "triple_extract_triplet"
            entries.append(entry)

        return PreInsertOutput(
            memory_entries=entries,
            metadata={
                "triplet_count": len(triplets),
                "extraction_method": self.extraction_method,
            },
        )

    # ── Triplet extraction ────────────────────────────────────────────────────

    def _extract_triplets(self, text: str) -> list[dict[str, str]]:
        if self.extraction_method == "llm":
            return self._extract_by_llm(text)
        return self._extract_simple(text)

    def _extract_simple(self, text: str) -> list[dict[str, str]]:
        """Heuristic SVO extraction — used as fallback when LLM is unavailable."""
        sentences = re.split(r"[。.!?！？]+", text)
        triplets: list[dict[str, str]] = []

        patterns = [
            r"(\w+)\s+(is|are|was|were|has|have)\s+(.+)",
            r"(\w+)\s+(做|说|认为|喜欢|讨厌)\s+(.+)",
        ]

        for sentence in sentences[: self.max_triplets]:
            sentence = sentence.strip()
            if not sentence:
                continue

            matched = False
            for pattern in patterns:
                m = re.search(pattern, sentence, re.IGNORECASE)
                if m:
                    triplets.append(
                        {
                            "subject": m.group(1).strip(),
                            "predicate": m.group(2).strip(),
                            "object": m.group(3).strip(),
                        }
                    )
                    matched = True
                    break

            if not matched and len(sentence.split()) >= 3:
                words = sentence.split()
                triplets.append(
                    {
                        "subject": words[0],
                        "predicate": words[1],
                        "object": " ".join(words[2:]),
                    }
                )

        return triplets[: self.max_triplets]

    def _extract_by_llm(self, text: str) -> list[dict[str, str]]:
        """LLM-based triple extraction with graceful fallback."""
        if not self.llm_generator:
            print("[TripleExtract] No LLM generator, falling back to simple extraction")
            return self._extract_simple(text)

        if not self.triple_extraction_prompt:
            print(
                "[TripleExtract] No triple_extraction_prompt configured, "
                "falling back to simple extraction"
            )
            return self._extract_simple(text)

        try:
            prompt = self.triple_extraction_prompt.format(dialogue=text)
            response = self.llm_generator.generate(prompt)
            triplets = self._parse_llm_response(response)

            with contextlib.suppress(Exception):
                model = getattr(self.llm_generator, "model_name", "-")
                print(
                    f"[TripleExtract] LLM called model={model} "
                    f"triplets_parsed={len(triplets)}"
                )

            if not triplets:
                print(
                    "[TripleExtract] LLM returned no parseable triples, "
                    "falling back to simple extraction"
                )
                return self._extract_simple(text)[: self.max_triplets]

            return triplets[: self.max_triplets]

        except Exception as e:
            with contextlib.suppress(Exception):
                model = getattr(self.llm_generator, "model_name", "-")
                print(
                    f"[TripleExtract] LLM call failed model={model} "
                    f"error={type(e).__name__}: {e}, fallback to simple"
                )
            return self._extract_simple(text)

    def _parse_llm_response(self, response: str) -> list[dict[str, str]]:
        """Parse LLM output of the form: (Subject, Predicate, Object)"""
        if not response or response.strip().lower() == "none":
            return []

        triplets: list[dict[str, str]] = []

        # Match patterns like (Alice, loves, rainy days) or Alice, loves, rainy days
        patterns = [
            r"\(([^,()]+),\s*([^,()]+),\s*([^,()]+)\)",  # (A, B, C)
            r"^([^,\n]+),\s*([^,\n]+),\s*([^,\n]+)$",    # A, B, C
        ]

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.lower() == "none":
                continue

            for pattern in patterns:
                m = re.search(pattern, line)
                if m:
                    subj = m.group(1).strip().strip("()")
                    pred = m.group(2).strip().strip("()")
                    obj = m.group(3).strip().strip("()")
                    if subj and pred and obj:
                        triplets.append(
                            {"subject": subj, "predicate": pred, "object": obj}
                        )
                    break

        return triplets

    def _reconstruct_triplet(self, triplet: dict[str, str]) -> str:
        return self.reconstruct_template.format(
            subject=triplet.get("subject", ""),
            predicate=triplet.get("predicate", ""),
            object=triplet.get("object", ""),
        )
