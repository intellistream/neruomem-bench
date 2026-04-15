![NeuroMem Logo](docs/assets/neuromem.png)

<h3 align="center">
Benchmark suite for NeuroMem memory systems
</h3>

<p align="center">
| <a href="https://github.com/intellistream/NeuroMem"><b>NeuroMem Core</b></a> | <a href="https://github.com/intellistream/SAGE"><b>SAGE</b></a> | <a href="https://arxiv.org/abs/2602.13967"><b>Paper (arXiv)</b></a> |
</p>

**neuromem-bench** is the standalone benchmark companion for [NeuroMem](https://github.com/intellistream/NeuroMem). It provides an extensible pipeline architecture to evaluate memory systems under long-dialogue scenarios, supporting both native NeuroMem/TiM operators and black-box adapters (e.g., mem0).

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

**Option B — Black-box pipeline** (wrap any external memory system, e.g., mem0):

```bash
# Requires Qdrant: docker run -p 6333:6333 qdrant/qdrant
python -m benchmarks.simple_experiment.simple_pipeline \
    --config benchmarks/simple_experiment/config/mem0_locomo.yaml --task_id conv-26

# Or use the convenience script
bash scripts/run_mem0_locomo.sh --task_id conv-26
```

Outputs are written to `.sage/output/benchmarks/benchmark_memory/<dataset>/<memory_name>/<task_id>_<ts>/`.

### Run tests

```bash
# Installation validation (Sage pipeline demos)
python test/installation_validation/sage_pipeline.py
python test/installation_validation/pipeline_as_service.py

# Offline mock benchmark tests
python -m pytest test/benchmark/ -v
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
