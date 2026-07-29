from __future__ import annotations

from benchmarks.experiment.utils import RuntimeConfig


def test_runtime_config_expands_nested_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROMEM_ROOT", "/workspace/neuromem")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
runtime:
  dataset: mock
services:
  adapters:
    - repo_path: "${NEUROMEM_ROOT}/third_party/streamfp"
""".strip(),
        encoding="utf-8",
    )

    config = RuntimeConfig.load(str(config_path))

    assert config.get("services.adapters")[0]["repo_path"] == (
        "/workspace/neuromem/third_party/streamfp"
    )
