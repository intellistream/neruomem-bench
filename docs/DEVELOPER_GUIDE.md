# NeuroMem Benchmark Developer Guide

本指南面向希望基于 `neuromem-bench` 框架开发自定义记忆实验的研究者和工程师。

## 目录

- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [自定义 Operator 开发](#自定义-operator-开发)
- [自定义数据集适配器](#自定义数据集适配器)
- [配置说明](#配置说明)
- [运行实验](#运行实验)

---

## 架构概览

neuromem-bench 采用三层 Pipeline 架构：

```
主 Pipeline:     MemorySource → PipelineCaller → MemorySink
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                                         ▼
插入 Pipeline: PreInsert → MemoryInsert → PostInsert   测试 Pipeline: PreRetrieval → MemoryRetrieval → PostRetrieval → MemoryEvaluation
```

**四个可扩展阶段**（策略模式）：

| 阶段 | 位置 | 用途 | 基类 |
|------|------|------|------|
| PreInsert | 插入前 | 数据清洗、特征提取、Embedding | `PreInsertBase` |
| PostInsert | 插入后 | 蒸馏、反思、遗忘、链接演化 | `PostInsertBase` |
| PreRetrieval | 检索前 | 查询优化、Embedding、查询分解 | `PreRetrievalBase` |
| PostRetrieval | 检索后 | 重排、过滤、合并、格式化 | `PostRetrievalBase` |

每个阶段包含：
- `base.py` - 抽象基类，定义接口
- `none_action.py` - 默认直通实现
- `registry.py` - Action 注册表
- `operator.py` - Operator 主体（从注册表选择 Action）

---

## 快速开始

```bash
# 1. 安装
pip install -e .

# 2. 配置
cp benchmarks/experiment/config/example.yaml my_config.yaml
# 编辑 my_config.yaml

# 3. 运行
python -m benchmarks.experiment.memory_test_pipeline --config my_config.yaml --task_id sample-01
```

---

## 自定义 Operator 开发

### 示例：实现 PreInsert 关键词提取 Action

```python
# my_actions/keyword_extract.py

from benchmarks.experiment.libs.pre_insert.base import PreInsertBase


class KeywordExtractAction(PreInsertBase):
    """从对话中提取关键词并附加到 metadata"""

    def __init__(self, config):
        self.keywords_model = config.get("operators.pre_insert.keywords_model", "simple")

    def process(self, data: dict) -> dict:
        dialogs = data.get("dialogs", [])
        memory_entries = []

        for dialog in dialogs:
            text = dialog.get("text", "")
            keywords = self._extract_keywords(text)

            entry = {
                "text": text,
                "metadata": {
                    "speaker": dialog.get("speaker"),
                    "keywords": keywords,
                },
            }
            memory_entries.append(entry)

        data["memory_entries"] = memory_entries
        return data

    def _extract_keywords(self, text: str) -> list[str]:
        # 实现关键词提取逻辑
        words = text.split()
        return [w for w in words if len(w) > 5][:10]
```

### 注册自定义 Action

```python
# 在你的入口脚本或 __init__.py 中
from benchmarks.experiment.libs.pre_insert.registry import PreInsertRegistry
from my_actions.keyword_extract import KeywordExtractAction

# 注册
PreInsertRegistry.register("keyword_extract", KeywordExtractAction)
```

### 配置使用

```yaml
operators:
  pre_insert:
    action: "keyword_extract"
    keywords_model: "simple"
```

### 其他阶段的开发方式完全相同

- `PostInsert`: 继承 `PostInsertBase`，注册到 `PostInsertRegistry`
- `PreRetrieval`: 继承 `PreRetrievalBase`，注册到 `PreRetrievalRegistry`
- `PostRetrieval`: 继承 `PostRetrievalBase`，注册到 `PostRetrievalRegistry`

---

## 自定义数据集适配器

### 实现 BaseDataLoader

```python
# my_dataset/adapter.py

from benchmarks.experiment.utils.dataloader.base import BaseDataLoader


class MyDatasetAdapter(BaseDataLoader):
    """自定义数据集适配器"""

    def __init__(self):
        # 加载你的数据集
        self._data = self._load_data()

    @property
    def dataset_name(self) -> str:
        return "my_dataset"

    def get_dialog(self, task_id, session_x, dialog_y):
        """返回一轮对话（1-2条消息）"""
        session = self._data[task_id]["sessions"][session_x]
        messages = session["messages"]

        result = []
        for i in range(dialog_y, min(dialog_y + 2, len(messages))):
            result.append({
                "speaker": messages[i]["speaker"],
                "text": messages[i]["text"],
            })
        return result

    def get_evaluation(self, task_id, session_x, dialog_y):
        """返回当前可见的测试问题"""
        questions = self._data[task_id]["questions"]
        visible = [q for q in questions if q["visible_after"] <= (session_x, dialog_y)]
        return visible

    def sessions(self, task_id):
        """返回 [(session_id, max_dialog_idx), ...]"""
        sessions_data = self._data[task_id]["sessions"]
        return [(sid, len(s["messages"]) - 1) for sid, s in sessions_data.items()]

    def question_count(self, task_id):
        return len(self._data[task_id]["questions"])

    def dialog_count(self, task_id):
        return sum(
            (len(s["messages"]) + 1) // 2
            for s in self._data[task_id]["sessions"].values()
        )

    def message_count(self, task_id):
        return sum(
            len(s["messages"])
            for s in self._data[task_id]["sessions"].values()
        )

    def statistics(self, task_id):
        return {
            "dataset": self.dataset_name,
            "task_id": task_id,
            "sessions": len(self._data[task_id]["sessions"]),
            "questions": self.question_count(task_id),
        }

    def _load_data(self):
        # 加载你的数据集文件
        ...
```

### 注册数据集

```python
from benchmarks.experiment.utils.dataloader import DataLoaderFactory
from my_dataset.adapter import MyDatasetAdapter

DataLoaderFactory.register("my_dataset", MyDatasetAdapter)
```

---

## 自定义记忆数据结构

`libs/datastructure/` 提供记忆系统的数据结构抽象，三层架构：

```
Layer 1 索引层:   BaseIndex ──→ IndexFactory.register("my_index", MyIndex)
Layer 2 容器层:   SimpleCollection（原始数据 + 多索引管理）
Layer 3 服务层:   BaseMemoryService ──→ @MemoryServiceRegistry.register("my_service")
```

### 示例：实现自定义索引

```python
# my_index.py
from benchmarks.experiment.libs.datastructure import BaseIndex, IndexFactory

class MyVectorIndex(BaseIndex):
    """自定义向量索引"""

    def __init__(self, config):
        super().__init__(config)
        self.dim = self.config.get("dim", 768)
        self._vectors = {}

    def add(self, data_id, text="", metadata=None, **kwargs):
        vector = kwargs.get("vector")
        if vector is not None:
            self._vectors[data_id] = vector

    def remove(self, data_id):
        self._vectors.pop(data_id, None)

    def query(self, query, **params):
        top_k = params.get("top_k", 5)
        # 实现你的向量检索逻辑
        return list(self._vectors.keys())[:top_k]

    def contains(self, data_id):
        return data_id in self._vectors

    def size(self):
        return len(self._vectors)

    def save(self, save_dir):
        pass  # 实现持久化

    def load(self, load_dir):
        pass  # 实现加载

    def clear(self):
        self._vectors.clear()

# 注册
IndexFactory.register("my_vector", MyVectorIndex)
```

### 示例：实现自定义记忆服务

```python
# my_service.py
from benchmarks.experiment.libs.datastructure import BaseMemoryService, MemoryServiceRegistry

@MemoryServiceRegistry.register("my_memory")
class MyMemoryService(BaseMemoryService):

    def _setup_indexes(self):
        self.collection.add_index("main", "my_vector", {"dim": 768})

    def insert(self, text, metadata=None, **kwargs):
        return self.collection.insert(text, metadata)

    def retrieve(self, query, top_k=5, threshold=None, **kwargs):
        return self.collection.retrieve("main", query, top_k=top_k)
```

### 使用内置 LSH 示例

```python
from benchmarks.experiment.libs.datastructure import SimpleCollection, MemoryServiceRegistry

# 创建 Collection 和 LSH 服务
collection = SimpleCollection("demo")
service = MemoryServiceRegistry.create("lsh_hash", collection, {
    "n_gram": 3,
    "num_perm": 128,
    "threshold": 0.5,
})

# 写入
service.insert("今天天气很好，阳光明媚", {"speaker": "user"})
service.insert("今天天气不错，万里无云", {"speaker": "user"})
service.insert("我喜欢吃苹果", {"speaker": "user"})

# 检索
results = service.retrieve("天气怎么样")
for r in results:
    print(f"[{r['score']:.2f}] {r['text']}")

# 去重检测
duplicates = service.find_duplicates("今天天气挺好的")
```

---

## 配置说明

参见 `benchmarks/experiment/config/example.yaml`，核心配置项：

| 配置项 | 说明 |
|--------|------|
| `runtime.dataset` | 数据集名称（必需） |
| `runtime.api_key` | LLM API Key |
| `runtime.base_url` | LLM 服务地址 |
| `runtime.model_name` | LLM 模型名 |
| `runtime.test_segments` | 测试分段数 |
| `services.services_type` | 记忆服务类型 |
| `operators.pre_insert.action` | PreInsert Action 名称 |
| `operators.post_insert.action` | PostInsert Action 名称 |
| `operators.pre_retrieval.action` | PreRetrieval Action 名称 |
| `operators.post_retrieval.action` | PostRetrieval Action 名称 |

---

## 运行实验

```bash
# 基本运行
python -m benchmarks.experiment.memory_test_pipeline \
    --config benchmarks/experiment/config/example.yaml \
    --task_id sample-01

# 输出文件
# .sage/benchmarks/benchmark_memory/{dataset}/{memory_name}/{task_id}_{timestamp}.json
# .sage/output/benchmarks/benchmark_memory/{dataset}/{memory_name}/{task_id}_{timestamp}/
#   ├── memory_service.log   # 增删查操作日志
#   └── memory_qa.log        # 问答详细日志
```
