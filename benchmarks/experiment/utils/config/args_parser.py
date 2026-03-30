"""命令行参数解析器"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroMem 记忆实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python memory_test_pipeline.py --config config/example.yaml --task_id conv-26
  python memory_test_pipeline.py --config config/example.yaml
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="配置文件路径 (必需)",
    )
    parser.add_argument(
        "--task_id",
        "-t",
        type=str,
        default=None,
        help="任务ID，覆盖配置文件中的设置 (可选)",
    )
    return parser.parse_args()
