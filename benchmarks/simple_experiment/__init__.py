"""simple_experiment — 黑盒记忆体基准测试

在 Sage LocalEnvironment 框架下运行，将外部记忆体（mem0 等）视为黑盒，
直接调用 add / search 接口，不拆解内部阶段。

与 benchmarks/experiment/ 的区别：
- 插入子 pipeline：PreInsert + MemoryInsert + PostInsert  →  SimpleMemoryAdd（单算子）
- 测试子 pipeline：PreRetrieval + MemoryRetrieval + PostRetrieval  →  SimpleMemorySearch
- 其余组件（MemorySource / MemorySink / MemoryEvaluation / PipelineBridge）全部复用
"""

__all__: list[str] = []
