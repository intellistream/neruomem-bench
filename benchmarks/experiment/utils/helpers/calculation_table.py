"""计算相关的工具函数"""


def calculate_test_thresholds(total_questions, segments):
    """计算测试阈值数组

    将总问题数均匀分成 segments 段，返回每段的结束位置作为测试触发点。

    Examples:
        >>> calculate_test_thresholds(100, 10)
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        >>> calculate_test_thresholds(0, 10)
        []
    """
    if total_questions == 0:
        return []

    segments = max(1, segments)

    thresholds = []
    for i in range(1, segments + 1):
        threshold = round(total_questions * i / segments)
        if not thresholds or threshold > thresholds[-1]:
            thresholds.append(threshold)

    return thresholds
