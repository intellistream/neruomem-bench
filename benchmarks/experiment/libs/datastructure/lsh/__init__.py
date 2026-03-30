"""LSH 记忆数据结构示例

基于 MinHash + LSH 的文本近似匹配索引和服务。

包含：
- LSHIndex: BaseIndex 实现，基于 datasketch MinHash + MinHashLSH
- LSHHashService: BaseMemoryService 实现，封装完整的 LSH 记忆服务

依赖：
    pip install datasketch
"""

from .lsh_index import LSHIndex
from .lsh_service import LSHHashService

__all__ = ["LSHIndex", "LSHHashService"]
