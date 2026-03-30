"""datastructure 包 - 记忆数据结构库

提供记忆系统的核心数据结构抽象和示例实现。

三层架构：
    Layer 1 - 索引层 (BaseIndex / IndexFactory)
        定义索引的增删查改接口，通过工厂模式注册和创建。
    Layer 2 - 容器层 (SimpleCollection)
        管理原始数据 + 多索引，提供统一的数据操作接口。
    Layer 3 - 服务层 (BaseMemoryService / MemoryServiceRegistry)
        封装业务逻辑，组合 Collection + Index 提供完整的记忆服务。

内置示例：
    - lsh/ : 基于 MinHash + LSH 的文本近似匹配（需 datasketch）

扩展方式：
    1. 实现 BaseIndex → IndexFactory.register()
    2. 实现 BaseMemoryService → @MemoryServiceRegistry.register()
    3. 在配置中指定你的索引/服务类型即可使用
"""

# Layer 1: 索引层
from benchmarks.experiment.libs.datastructure.base_index import BaseIndex
from benchmarks.experiment.libs.datastructure.index_factory import IndexFactory

# Layer 2: 容器层
from benchmarks.experiment.libs.datastructure.collection import SimpleCollection

# Layer 3: 服务层
from benchmarks.experiment.libs.datastructure.base_service import BaseMemoryService
from benchmarks.experiment.libs.datastructure.service_registry import MemoryServiceRegistry

# 注册内置 LSH 索引
from benchmarks.experiment.libs.datastructure.lsh.lsh_index import LSHIndex

IndexFactory.register("lsh", LSHIndex)

# 导入 LSH 服务（触发 @register 装饰器自动注册）
import benchmarks.experiment.libs.datastructure.lsh.lsh_service  # noqa: F401

__all__ = [
    # 索引层
    "BaseIndex",
    "IndexFactory",
    # 容器层
    "SimpleCollection",
    # 服务层
    "BaseMemoryService",
    "MemoryServiceRegistry",
    # 内置实现
    "LSHIndex",
]
