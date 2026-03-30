from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录（包含 pyproject.toml 的目录）"""
    current_path = Path(__file__).resolve()

    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("未找到项目根目录")
