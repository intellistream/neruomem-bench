"""SimplePipelineCaller — 黑盒测试 Pipeline 的主编排算子

相较于 PipelineCaller（四段式 experiment），本类剔除了四个策略阶段的 timing 追踪，
保留了核心调度逻辑不变：阈值驱动测试触发、分段 timing 累积、进度条展示。

调用两个子 pipeline（通过 PipelineBridge）：
    memory_add_service   → SimpleMemoryAdd
    memory_search_service → SimpleMemorySearch → MemoryEvaluation

返回给 MemorySink 的数据包结构与 PipelineCaller 完全兼容。
"""

from __future__ import annotations

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    DataLoaderFactory,
    ProgressBar,
    calculate_test_thresholds,
)


class SimplePipelineCaller(MapFunction):
    """黑盒测试主编排算子 — 简化版 PipelineCaller"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dataset: str = config.get("dataset")
        self.task_id: str = config.get("task_id")

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.adapter_name: str = services_type.split(".")[-1]

        self.service_timeout: float = config.get("runtime.service_timeout", 300.0)

        self.loader = DataLoaderFactory.create(self.dataset)
        self.total_questions: int = self.loader.question_count(self.task_id)
        self.last_tested_count: int = 0

        test_segments: int = config.get("runtime.test_segments", 10)
        self.test_thresholds: list[int] = calculate_test_thresholds(
            self.total_questions, test_segments
        )
        self.next_threshold_idx: int = 0
        self.total_dialogs_inserted: int = 0

        # 累积插入 timing（每条 add_ms）
        self.accumulated_add_timings: list[float] = []
        self.sent_add_timing_count: int = 0

        self.progress_bar: ProgressBar | None = None
        self.verbose: bool = config.get("runtime.memory_test_verbose", True)
        self.insert_verbose: bool = config.get("runtime.memory_insert_verbose", False)

    def execute(self, data: dict) -> dict | None:
        if not data:
            return None

        task_id = data.get("task_id")
        session_id = data.get("session_id")
        dialog_id = data.get("dialog_id")
        dialogs = data.get("dialogs", [])
        dialog_len = data.get("dialog_len", 0)
        packet_idx = data.get("packet_idx", 0)
        total_packets = data.get("total_packets", 0)
        is_session_end = data.get("is_session_end", False)
        is_last_packet = (packet_idx + 1) >= total_packets

        # 进度条初始化
        if self.progress_bar is None:
            self.progress_bar = ProgressBar(total=total_packets, desc="处理对话")
        self.progress_bar.update(1)
        if not self.insert_verbose:
            print()

        if self.insert_verbose:
            print(f"\n{'=' * 60}")
            print(
                f"\033[92m[Memory Source]\033[0m "
                f"(Packet {packet_idx + 1}/{total_packets})"
            )
            prefix = ">> "
            print(f"{prefix}Session: {session_id}, Dialog {dialog_id}")
            for i, turn in enumerate(dialogs):
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                print(f"{prefix}   Dialog {dialog_id + i} ({speaker}): {text}")
            print(f"{'=' * 60}")

        # ── Phase 1: Memory Add ────────────────────────────────────────────
        add_data = {
            "task_id": task_id,
            "session_id": session_id,
            "dialog_id": dialog_id,
            "dialogs": dialogs,
            "packet_idx": packet_idx,
            "total_packets": total_packets,
            "is_session_end": is_session_end,
        }

        add_result = None
        try:
            add_result = self.call_service(
                "memory_add_service",
                add_data,
                method="process",
                timeout=self.service_timeout,
            )
        except TimeoutError as e:
            print(f"[WARNING SimplePipelineCaller] memory_add_service 超时: {e}")
            if is_last_packet:
                if self.progress_bar:
                    self.progress_bar.close()
                return {
                    "dataset": self.dataset,
                    "task_id": task_id,
                    "completed": True,
                    "warning": f"最后一包 memory_add 超时: {e}",
                }
            raise

        # 累积 add_ms timing
        if add_result and "stage_timings" in add_result:
            add_ms = add_result["stage_timings"].get("add_ms")
            if add_ms is not None:
                self.accumulated_add_timings.append(float(add_ms))

        self.total_dialogs_inserted += dialog_len

        # ── Phase 2: 阈值检查 + Memory Search + Evaluation ────────────────
        current_questions = self.loader.get_evaluation(
            task_id,
            session_x=session_id,
            dialog_y=dialog_id + dialog_len - 1,
        )
        current_count = len(current_questions)

        should_test = False
        next_threshold = None
        if self.next_threshold_idx < len(self.test_thresholds):
            next_threshold = self.test_thresholds[self.next_threshold_idx]
            if current_count >= next_threshold:
                should_test = True

        if not should_test:
            if self.verbose:
                threshold_info = (
                    f"下一阈值: {next_threshold} 问题" if next_threshold else "无更多阈值"
                )
                print(
                    f"  [阈值检查] 当前: {current_count}/{self.total_questions} 问题 | "
                    f"已测: {self.last_tested_count} | {threshold_info}"
                )

            if is_last_packet:
                if self.progress_bar:
                    self.progress_bar.close()

                # 发送剩余 timing 数据
                remaining = self.accumulated_add_timings[self.sent_add_timing_count :]
                if remaining:
                    return {
                        "dataset": self.dataset,
                        "task_id": task_id,
                        "completed": True,
                        "stage_timings": {
                            "insert": {
                                "add_ms": remaining,
                            },
                            "test": [],
                        },
                    }
                return {
                    "dataset": self.dataset,
                    "task_id": task_id,
                    "completed": True,
                }

            if self.verbose:
                print(f"{'=' * 60}")
            return None

        # ── 阈值触发：遍历当前所有可见问题 ───────────────────────────────
        if self.verbose:
            print(f"{'+' * 60}", flush=True)
            print("【QA】：问题驱动测试触发", flush=True)
            print(f">> 当前可见问题数：{current_count}/{self.total_questions}", flush=True)
            print(f">> 已测试问题数：{self.last_tested_count}", flush=True)
            print(
                f">> 触发阈值：{next_threshold}"
                f"（第 {self.next_threshold_idx + 1}/{len(self.test_thresholds)} 个阈值）",
                flush=True,
            )
            print(f">> 测试范围：问题 1 到 {current_count}", flush=True)

        # 获取记忆体统计（best-effort）
        memory_stats = None
        try:
            memory_stats = self.call_service(
                self.adapter_name,
                method="get_stats",
                timeout=self.service_timeout,
            )
        except Exception as e:
            if self.verbose:
                print(f">> 警告：获取记忆体统计失败: {e}")

        test_answers: list[dict] = []
        search_ms_list: list[float] = []

        for q_idx, qa in enumerate(current_questions):
            question = qa["question"]
            test_data = {
                "task_id": task_id,
                "session_id": session_id,
                "dialog_id": dialog_id,
                "dialogs": dialogs,
                "question": question,
                "question_idx": q_idx + 1,
                "question_metadata": qa,
            }

            result = self.call_service(
                "memory_search_service",
                test_data,
                method="process",
                timeout=self.service_timeout,
            )

            if "answer" in result:
                if "stage_timings" in result:
                    ms = result["stage_timings"].get("search_ms")
                    if ms is not None:
                        search_ms_list.append(float(ms))

                test_answers.append(
                    {
                        "question_index": q_idx + 1,
                        "question": question,
                        "predicted_answer": result["answer"],
                        "metadata": result.get("question_metadata", qa),
                    }
                )

                if self.verbose:
                    print(f">> Question {q_idx + 1}：{question}", flush=True)
                    print(f">> Answer：{result['answer']}", flush=True)

        avg_search_ms = (
            sum(search_ms_list) / len(search_ms_list) if search_ms_list else 0.0
        )

        # 增量 add_ms（只发送本段新增部分）
        incremental_add = self.accumulated_add_timings[self.sent_add_timing_count :]
        self.sent_add_timing_count = len(self.accumulated_add_timings)

        test_result: dict = {
            "dataset": self.dataset,
            "task_id": task_id,
            "question_range": {"start": 1, "end": current_count},
            "dialogs_inserted": self.total_dialogs_inserted,
            "answers": test_answers,
            "completed": is_last_packet,
            "stage_timings": {
                "insert": {"add_ms": incremental_add},
                "test": {"search_ms": avg_search_ms},
                "memory_stats": memory_stats,
            },
        }

        self.last_tested_count = current_count
        self.next_threshold_idx += 1

        if self.verbose:
            print(f"{'+' * 60}")
        if is_last_packet and self.progress_bar:
            self.progress_bar.close()

        return test_result
