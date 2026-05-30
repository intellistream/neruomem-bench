from __future__ import annotations

import textwrap

from benchmarks.experiment.utils.config.config_loader import RuntimeConfig


def test_runtime_config_preserves_strategy_adapters(tmp_path):
    config_path = tmp_path / "adapter_config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            runtime:
              dataset: "mock_locomo"
            services:
              services_type: "partitional.fifo_queue"
              fifo_queue:
                strategy_adapters:
                  - name: "streamfp_selector"
                    enabled: true
                    repo_path: "/home/shuhao/streamfp"
                    threshold: 0.35
                  - name: "flowrag_retriever"
                    enabled: false
                    repo_path: "/home/shuhao/FlowRAG"
                    index_dir: "/tmp/flowrag-index"
                    index_name: "toy"
            """
        ),
        encoding="utf-8",
    )

    config = RuntimeConfig.load(str(config_path), task_id="mock-01")
    adapters = config.get("services.fifo_queue.strategy_adapters")

    assert isinstance(adapters, list)
    assert [adapter["name"] for adapter in adapters] == [
        "streamfp_selector",
        "flowrag_retriever",
    ]
    assert adapters[0]["enabled"] is True
    assert adapters[1]["index_name"] == "toy"