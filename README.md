![NeuroMem Logo](docs/assets/neuromem.png)

<h3 align="center">
Benchmark suite for NeuroMem memory systems
</h3>

<p align="center">
| <a href="https://github.com/intellistream/NeuroMem"><b>NeuroMem Core</b></a> | <a href="https://github.com/intellistream/SAGE"><b>SAGE</b></a> | <a href="https://arxiv.org/abs/2602.13967"><b>Paper (arXiv)</b></a> |
</p>

**neuromem-bench** is the standalone benchmark companion for [NeuroMem](https://github.com/intellistream/NeuroMem). It provides an extensible pipeline architecture to evaluate memory systems under long-dialogue scenarios, supporting both native NeuroMem/TiM operators and black-box adapters (e.g., mem0).

## Publication

This benchmark suite accompanies the following paper:

- Ruicheng Zhang et al. "Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs." ICML 2026. See also the [arXiv version](https://arxiv.org/abs/2602.13967).

## Owner & Contact

- Repository Owner: [@KimmoZAG](https://github.com/KimmoZAG) (RuiCheng / 张睿诚)
- Maintainer Contact: Please open an issue in this repo and mention `@KimmoZAG`.

---

## Getting Started

```bash
# Clone
git clone https://github.com/intellistream/NeuroMem-Bench.git
cd neuromem-bench

# Install (editable mode)
pip install -e .

# For black-box baselines (e.g., mem0)
pip install -e ".[mem0]"

# Or use the quickstart script
./quickstart.sh
```

### Prerequisites — Model Services

Most benchmarks require an LLM and an embedding service. Start them with the provided scripts:

```bash
bash scripts/deploy/deploy_llama31_8b.sh   # LLM   → port 18000
bash scripts/deploy/deploy_bge_m3.sh       # Embed → port 18001
```

Stop them with the corresponding `stop_*.sh` scripts.

### Run a benchmark

**Option A — Native pipeline** (NeuroMem / TiM operators, full pre/post stage control):

```bash
# Use the TiM-on-LoCoMo config as a starting point
cp benchmarks/experiment/config/tim_locomo_pipeline.yaml my_config.yaml
# Edit my_config.yaml (dataset, task_id, endpoints, etc.)

python -m benchmarks.experiment.memory_test_pipeline \
    --config my_config.yaml --task_id conv-26

# Or use the convenience script
bash scripts/run_tim_locomo.sh --task_id conv-26
# Multiple tasks: --tasks "conv-26 conv-27"
```

Online continual neural memory now has a ready-to-run LoCoMo template too:

```bash
python -m benchmarks.experiment.memory_test_pipeline \
  --config benchmarks/experiment/config/online_continual_memory_locomo_pipeline.yaml \
  --task_id conv-mini-01

# Or use the convenience script
bash scripts/run_online_continual_locomo.sh --task_id conv-mini-01
```

This config uses the built-in local `locomo` adapter. By default it reads
`benchmarks/experiment/data/locomo/mini_locomo.json`; set `LOCOMO_DATA_FILE`
to point at another LoCoMo-format JSON file if needed.

For a full LoCoMo JSON file, use the production-oriented template:

```bash
bash scripts/run_online_continual_locomo_full.sh \
  --data_file /absolute/path/to/locomo_full.json \
  --task_id conv-26
```

This runner exports `LOCOMO_DATA_FILE` for the built-in local `locomo` adapter,
so you do not need to edit the YAML just to switch datasets.

If you just need a public LoCoMo file quickly, the repo now includes a helper:

```bash
bash scripts/download_locomo10.sh
```

It downloads the public `snap-research/locomo` release file into
`benchmarks/experiment/data/locomo/locomo10_official.json` and converts it into
the local adapter format at
`benchmarks/experiment/data/locomo/locomo10_local.json`.

**Option B — Black-box pipeline** (wrap any external memory system, e.g., mem0):

```bash
# Requires Qdrant: docker run -p 6333:6333 qdrant/qdrant
python -m benchmarks.simple_experiment.simple_pipeline \
    --config benchmarks/simple_experiment/config/mem0_locomo.yaml --task_id conv-26

# Or use the convenience script
bash scripts/run_mem0_locomo.sh --task_id conv-26
```

Outputs are written to `.sage/output/benchmarks/benchmark_memory/<dataset>/<memory_name>/<task_id>_<ts>/`.

### Run ATC-style system benchmarks

The repository also includes a lightweight runtime benchmark driver for the
ATC-facing systems matrix. It measures three views directly at the memory
service boundary:

- concurrency scaling: throughput, tail latency, and Jain fairness
- retained-state footprint: process RSS, Python heap peak, and backend storage stats
- observability overhead: telemetry disabled vs enabled under the same workload

Use the dedicated online continual config as a starting point:

```bash
python -m benchmarks.evaluation.system_runtime_bench \
  --config benchmarks/experiment/config/online_continual_memory_system_eval.yaml \
  --workers 1 2 4 8 \
  --initial-records 128 \
  --operations-per-worker 64 \
  --insert-every 4 \
  --retrieval-top-k 3 \
  --footprint-checkpoints 64,128,256,512 \
  --telemetry-limit-enabled 100 \
  --output .sage/output/benchmarks/system_runtime/online_continual_memory_atc.json
```

This command writes a JSON report with `concurrency`, `footprint`, and
`observability` sections. The driver can also be pointed at other service YAMLs
as long as they define `services.services_type` and the corresponding service
config block.

### Enable External Strategy Adapters

The native pipeline can now enable repo-backed external adapters directly from YAML via `services.<service_name>.strategy_adapters`.

- `streamfp_selector`: runs before insert and can score or skip low-value entries.
- `flowrag_retriever`: runs after retrieval and can merge external FlowRAG results into the final context set.

Use [benchmarks/experiment/config/fifo_external_adapters.yaml](benchmarks/experiment/config/fifo_external_adapters.yaml) as the starting point. The main fields are:

```yaml
services:
  services_type: "partitional.fifo_queue"
  fifo_queue:
    strategy_adapters:
      - name: "streamfp_selector"
        enabled: true
        repo_path: "/home/shuhao/streamfp"
        threshold: 0.35
      - name: "flowrag_retriever"
        enabled: true
        repo_path: "/home/shuhao/FlowRAG"
        index_dir: "/absolute/path/to/index"
        index_name: "toy"
        external_top_k: 5
```

Notes:

- Enable one adapter first when bringing up a new environment; it is easier to isolate dependency or path problems.
- Repo-backed real runs require the same local dependencies used in the integration tests: `faiss-cpu`, `geomloss`, and `loguru`.
- `streamfp_selector` is useful for insert-time gating, while `flowrag_retriever` is useful for retrieval-time augmentation. They can be enabled independently.

### Run tests

```bash
# Installation validation (Sage pipeline demos)
python test/installation_validation/sage_pipeline.py
python test/installation_validation/pipeline_as_service.py

# Installation validation for online continual memory
PYTHONPATH=/home/shuhao/neuromem:$PYTHONPATH \
python test/installation_validation/online_continual_memory_pipeline.py

# Offline validation with the real local LoCoMo adapter/data but mocked model services
PYTHONPATH=/home/shuhao/neuromem:$PYTHONPATH \
python test/installation_validation/online_continual_locomo_pipeline.py

# Component-level benchmark validation with real FlowRAG/streamfp adapters on mock data
bash scripts/run_fifo_external_adapters_mock.sh --task_id mock-01

# Offline mock benchmark tests
python -m pytest test/benchmark/ -v

# Config-level validation for external strategy adapter YAML
python -m pytest test/benchmark/test_strategy_adapter_config.py -v
```

## Architecture

Two pipeline modes share the same evaluation layer:

```
Native pipeline (memory_test_pipeline.py):
  MemorySource → PipelineCaller → MemorySink
                      │
       ┌──────────────┴──────────────────────────────────────────┐
       ▼ Insert                                                   ▼ Test
  PreInsert → MemoryInsert → PostInsert    PreRetrieval → MemoryRetrieval → PostRetrieval → MemoryEvaluation

Black-box pipeline (simple_pipeline.py):
  MemorySource → SimplePipelineCaller → SimpleMemorySink
                        │
         ┌──────────────┴──────────────────────────┐
         ▼ Insert                                   ▼ Test
    SimpleMemoryAdd                   SimpleMemorySearch → MemoryEvaluation
```

The four extensible stages in the native pipeline (**PreInsert** / **PostInsert** / **PreRetrieval** / **PostRetrieval**) use a strategy pattern — swap operators via config without changing pipeline code.

## Customization

- **Custom Operator** — Subclass the stage base class and register it in the `Registry`.
- **Custom Dataset** — Implement `BaseDataLoader` and register it with `DataLoaderFactory`.
- **Custom Memory Service (native)** — Implement `BaseIndex` / `BaseMemoryService` and register via the decorator pattern.
- **Custom Black-box Adapter** — Implement the `SimpleAdapterFactory` interface.

See [Developer Guide](docs/DEVELOPER_GUIDE.md) for details.

## Project Structure

```
benchmarks/
  experiment/                          # Native pipeline
    memory_test_pipeline.py            # Entry point
    pipeline_service.py                # Pipeline-as-Service bridge
    config/                            # YAML config templates
    libs/
      {memory_source,insert,retrieval,evaluation,sink}.py
      pipeline_caller.py
      pre_insert/ post_insert/         # Extensible stages
      pre_retrieval/ post_retrieval/
      datastructure/                   # BaseIndex, LSH, service registry
    utils/
      config/ dataloader/ helpers/ llm/ ui/
  simple_experiment/                   # Black-box adapter pipeline
    simple_pipeline.py                 # Entry point
    config/                            # mem0_locomo.yaml, etc.
    adapters/                          # SimpleAdapterFactory + adapters
    libs/                              # SimpleMemoryAdd/Search/Sink/Caller
test/
  installation_validation/             # Sage feature demos
  benchmark/                           # Offline mock tests
scripts/
  deploy/                              # Model service lifecycle (vLLM)
  run_tim_locomo.sh                    # TiM on LoCoMo
  run_mem0_locomo.sh                   # mem0 on LoCoMo
```

## Dependencies

- Python >= 3.11
- `isage-neuromem[full]` — Sage runtime + NeuromemServiceFactory
- `openai` — LLM / Embedding API client
- `pyyaml` — Config file parsing
- `datasketch` — LSH data structures
- `mem0ai`, `qdrant-client` *(optional)* — Black-box mem0 baseline

## Part of SAGE Ecosystem

neuromem-bench is a component of the [SAGE](https://github.com/intellistream/SAGE) (Structured AI Graph Engine) project by IntelliStream Team.

## License

Apache-2.0 License — see LICENSE file for details.

## Citation

If you use neuromem-bench in your research, please cite:

```bibtex
@misc{zhang2026neuromemgranulardecompositionstreaming,
  title={Neuromem: A Granular Decomposition of the Streaming Lifecycle in External Memory for LLMs},
  author={Ruicheng Zhang and Xinyi Li and Tianyi Xu and Shuhao Zhang and Xiaofei Liao and Hai Jin},
  year={2026},
  eprint={2602.13967},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.13967}
}
```
