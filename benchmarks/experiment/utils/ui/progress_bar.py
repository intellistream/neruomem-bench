"""进度条工具"""


class ProgressBar:
    """简单的进度条实现

    示例:
        progress = ProgressBar(total=100, desc="处理中")
        for i in range(100):
            progress.update(1)
        progress.close()
    """

    def __init__(self, total: int, desc: str = "进度", bar_length: int = 40):
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0

    def update(self, n: int = 1):
        self.current += n
        self._print()

    def _print(self):
        if self.total == 0:
            return

        progress = self.current / self.total
        filled_length = int(self.bar_length * progress)
        bar = "█" * filled_length + "░" * (self.bar_length - filled_length)
        percent = progress * 100

        print(
            f"\r⏳ {self.desc}: |{bar}| {self.current}/{self.total} ({percent:.1f}%)",
            end="",
            flush=True,
        )

    def close(self):
        print()
