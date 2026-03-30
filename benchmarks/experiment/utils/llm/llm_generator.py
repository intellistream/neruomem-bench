"""LLM 生成工具

提供直接调用 OpenAI-compatible API 的功能，内置响应解析能力：
- generate(): 基础文本生成
- generate_json(): 生成并解析 JSON 响应
- generate_triples(): 生成并解析三元组（HippoRAG 风格）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class LLMGenerator:
    """简单的 LLM 生成器类"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: int | None = None,
        **kwargs: Any,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.extra_params = kwargs

    @classmethod
    def from_config(cls, config, prefix: str = "runtime") -> LLMGenerator:
        api_key = config.get(f"{prefix}.api_key")
        base_url = config.get(f"{prefix}.base_url")
        model_name = config.get(f"{prefix}.model_name")
        max_tokens = config.get(f"{prefix}.max_tokens", 512)
        temperature = config.get(f"{prefix}.temperature", 0.7)
        seed = config.get(f"{prefix}.seed")

        if not all([api_key, base_url, model_name]):
            raise ValueError("缺少必需的 LLM 配置: api_key, base_url, model_name")

        extra_params = {}
        extra_param_names = [
            "enable_thinking",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]

        for param_name in extra_param_names:
            value = config.get(f"{prefix}.{param_name}")
            if value is not None:
                extra_params[param_name] = value

        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            **extra_params,
        )

    def generate(self, prompt: str, **override_params: Any) -> str:
        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.seed is not None:
            request_params["seed"] = self.seed

        request_params.update(self.extra_params)
        request_params.update(override_params)

        response = self.client.chat.completions.create(**request_params)

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""

        return ""

    def generate_json(
        self, prompt: str, default: Any = None, **override_params: Any
    ) -> dict | list | Any:
        override_params.pop("default", None)
        response = self.generate(prompt, **override_params)
        return self._parse_json(response, default)

    def generate_triples(
        self, prompt: str, **override_params: Any
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        response = self.generate(prompt, **override_params)
        triples = self._parse_triples(response)
        refactors = self._refactor_triples(triples)
        return triples, refactors

    def _parse_json(self, response: str, default: Any = None) -> Any:
        if default is None:
            default = {}

        if not response:
            return default

        try:
            response_cleaned = response.strip()

            if not response_cleaned.startswith(("{", "[")):
                start_brace = response_cleaned.find("{")
                start_bracket = response_cleaned.find("[")

                if start_brace == -1 and start_bracket == -1:
                    return default

                if start_brace == -1:
                    start_idx = start_bracket
                elif start_bracket == -1:
                    start_idx = start_brace
                else:
                    start_idx = min(start_brace, start_bracket)

                response_cleaned = response_cleaned[start_idx:]

            if response_cleaned.startswith("{"):
                end_idx = response_cleaned.rfind("}") + 1
            else:
                end_idx = response_cleaned.rfind("]") + 1

            if end_idx > 0:
                response_cleaned = response_cleaned[:end_idx]

            try:
                return json.loads(response_cleaned)
            except json.JSONDecodeError:
                import ast

                try:
                    return ast.literal_eval(response_cleaned)
                except (ValueError, SyntaxError):
                    response_fixed = response_cleaned.replace("'", '"')
                    return json.loads(response_fixed)

        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            print(f"[WARNING] JSON parsing error: {e}")
            return default

    def _parse_triples(self, triples_text: str) -> list[tuple[str, str, str]]:
        triples = []

        if triples_text.strip().lower() == "none":
            return triples

        lines = triples_text.strip().split("\n")
        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            line = re.sub(r"^\(\d+\)\s*", "", line)
            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^\d+\)\s*", "", line)
            line = line.strip()

            if line.startswith("(") and line.endswith(")"):
                content = line[1:-1]
                parts = [part.strip() for part in content.split(",")]
                if len(parts) == 3:
                    triples.append((parts[0], parts[1], parts[2]))

        return triples

    def _refactor_triples(self, triples: list[tuple[str, str, str]]) -> list[str]:
        return [f"{s} {p} {o}" for s, p, o in triples]

    def deduplicate_triples(
        self,
        triples: list[tuple[str, str, str]],
        refactors: list[str],
    ) -> tuple[list[tuple[str, str, str]], list[str]]:
        seen: set[str] = set()
        unique_triples = []
        unique_refactors = []

        for triple, refactor in zip(triples, refactors):
            if refactor not in seen:
                seen.add(refactor)
                unique_triples.append(triple)
                unique_refactors.append(refactor)

        return unique_triples, unique_refactors
