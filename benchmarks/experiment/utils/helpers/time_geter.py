from datetime import UTC, datetime


def get_time_filename():
    """返回格式为 'YYMMDD' 的时间字符串"""
    now = datetime.now(UTC)
    return now.strftime("%y%m%d")


def get_runtime_timestamp():
    """返回格式为 'HHMM' 的时间字符串，用于文件名"""
    now = datetime.now(UTC)
    return now.strftime("%H%M")
