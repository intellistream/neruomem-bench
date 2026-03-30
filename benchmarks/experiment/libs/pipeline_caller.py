"""主 Pipeline 的核心处理算子

协调记忆存储和记忆测试两个子 Pipeline，实现问题驱动的测试策略。
"""

from sage.foundation import MapFunction

from benchmarks.experiment.utils import (
    DataLoaderFactory,
    ProgressBar,
    calculate_test_thresholds,
)


class PipelineCaller(MapFunction):
    """主 Pipeline 的核心 Map 算子

    负责协调记忆存储和记忆测试两个子 Pipeline，实现问题驱动的测试策略。
    """

    def __init__(self, config):
        super().__init__()
        self.dataset = config.get("dataset")
        self.task_id = config.get("task_id")

        services_type = config.get("services.services_type")
        if not services_type:
            raise ValueError("Missing required config: services.services_type")
        self.memory_service_name = services_type.split(".")[-1]

        self.service_timeout = config.get("runtime.service_timeout", 300.0)

        self.loader = DataLoaderFactory.create(self.dataset)

        self.progress_bar = None

        self.total_questions = self.loader.question_count(self.task_id)
        self.last_tested_count = 0

        test_segments = config.get("runtime.test_segments", 10)
        self.test_thresholds = calculate_test_thresholds(self.total_questions, test_segments)
        self.test_based_on_facts = False

        self.next_threshold_idx = 0

        self.total_dialogs_inserted = 0

        self.accumulated_insert_timings = {
            "pre_insert_ms": [],
            "memory_insert_ms": [],
            "post_insert_ms": [],
        }

        self.sent_insert_timing_count = 0

        self.memory_insert_verbose = config.get("runtime.memory_insert_verbose", False)
        self.memory_test_verbose = config.get("runtime.memory_test_verbose", True)

    def execute(self, data):
        if not data:
            return None

        task_id = data.get("task_id")
        session_id = data.get("session_id")
        dialog_id = data.get("dialog_id")
        dialogs = data.get("dialogs", [])
        dialog_len = data.get("dialog_len", 0)
        packet_idx = data.get("packet_idx", 0)
        total_packets = data.get("total_packets", 0)

        if self.progress_bar is None:
            self.progress_bar = ProgressBar(total=total_packets, desc="处理对话")
        self.progress_bar.update(1)
        if not self.memory_insert_verbose:
            print()

        if self.memory_insert_verbose:
            print(f"\n{'=' * 60}")
            print(f"\033[92m[Memory Source]\033[0m (Packet {packet_idx + 1}/{total_packets})")
            prefix = ">> "
            session_info = f"{prefix}Session: {session_id}, Dialog {dialog_id}"
            if len(dialogs) == 2:
                session_info += f" - {dialog_id + 1}"
            print(session_info)
            for i, dialog in enumerate(dialogs):
                speaker = dialog.get("speaker", "Unknown")
                text = dialog.get("text", "")
                print(f"{prefix}   Dialog {dialog_id + i} ({speaker}): {text}")
            print(f"{'=' * 60}")

        # Phase 1: Memory Insert
        is_session_end = data.get("is_session_end", False)

        insert_data = {
            "task_id": task_id,
            "session_id": session_id,
            "dialog_id": dialog_id,
            "dialogs": dialogs,
            "packet_idx": packet_idx,
            "total_packets": total_packets,
            "is_session_end": is_session_end,
        }

        is_last_packet = packet_idx + 1 >= total_packets

        insert_result = None
        try:
            insert_result = self.call_service(
                "memory_insert_service",
                insert_data,
                method="process",
                timeout=self.service_timeout,
            )
        except TimeoutError as e:
            print(f"[WARNING PipelineCaller] memory_insert_service 超时: {e}")
            if is_last_packet:
                if self.progress_bar:
                    self.progress_bar.close()
                return {
                    "dataset": self.dataset,
                    "task_id": task_id,
                    "completed": True,
                    "warning": f"最后一包 memory_insert 超时: {e}",
                }
            raise

        if insert_result and "stage_timings" in insert_result:
            batch_timings = insert_result["stage_timings"]
            for key in ["pre_insert_ms", "memory_insert_ms", "post_insert_ms"]:
                if key in batch_timings:
                    timing_list = batch_timings[key]
                    if isinstance(timing_list, list):
                        self.accumulated_insert_timings[key].extend(timing_list)

        self.total_dialogs_inserted += dialog_len

        # Phase 2: Memory Test (question-driven)
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

            if self.test_based_on_facts:
                if self.total_dialogs_inserted >= next_threshold:
                    should_test = True
            else:
                if current_count >= next_threshold:
                    should_test = True

        if not should_test:
            if self.memory_test_verbose:
                if self.test_based_on_facts:
                    threshold_info = (
                        f"下一阈值: {next_threshold} facts" if next_threshold else "无更多阈值"
                    )
                    print(
                        f"  [阈值检查] 当前: {current_count}问题 | 已插: {self.total_dialogs_inserted} facts | 已测: {self.last_tested_count} | {threshold_info}"
                    )
                else:
                    threshold_info = (
                        f"下一阈值: {next_threshold} 问题" if next_threshold else "无更多阈值"
                    )
                    print(
                        f"  [阈值检查] 当前: {current_count}/{self.total_questions} 问题 | 已测: {self.last_tested_count} | {threshold_info}"
                    )

            if is_last_packet:
                if self.memory_test_verbose:
                    print(">> 最后一个数据包，发送剩余 timing 数据")
                    print(f"{'=' * 60}")

                if self.progress_bar:
                    self.progress_bar.close()

                current_count = len(self.accumulated_insert_timings["pre_insert_ms"])
                remaining_count = current_count - self.sent_insert_timing_count

                if remaining_count > 0:
                    incremental_insert_timings = {
                        key: values[self.sent_insert_timing_count :]
                        for key, values in self.accumulated_insert_timings.items()
                    }

                    return {
                        "dataset": self.dataset,
                        "task_id": task_id,
                        "completed": True,
                        "stage_timings": {
                            "insert": incremental_insert_timings,
                            "test": [],
                        },
                    }
                return {
                    "dataset": self.dataset,
                    "task_id": task_id,
                    "completed": True,
                }

            if self.memory_test_verbose:
                print(f"{'=' * 60}")
            return None

        # Threshold reached - trigger test
        if self.memory_test_verbose:
            print(f"{'+' * 60}", flush=True)
            if self.test_based_on_facts:
                print("【QA】：Facts数量驱动测试触发", flush=True)
                print(f">> 已插入facts数：{self.total_dialogs_inserted}", flush=True)
                print(f">> 当前可见问题数：{current_count}/{self.total_questions}", flush=True)
                print(f">> 已测试问题数：{self.last_tested_count}", flush=True)
                print(
                    f">> 触发阈值：{next_threshold} facts（第 {self.next_threshold_idx + 1}/{len(self.test_thresholds)} 个阈值）",
                    flush=True,
                )
            else:
                print("【QA】：问题驱动测试触发", flush=True)
                print(f">> 当前可见问题数：{current_count}/{self.total_questions}", flush=True)
                print(f">> 已测试问题数：{self.last_tested_count}", flush=True)
                print(
                    f">> 触发阈值：{next_threshold}（第 {self.next_threshold_idx + 1}/{len(self.test_thresholds)} 个阈值）",
                    flush=True,
                )
            print(f">> 测试范围：问题 1 到 {current_count}", flush=True)

        memory_stats = None
        try:
            stats_result = self.call_service(
                self.memory_service_name,
                method="get_stats",
                timeout=self.service_timeout,
            )
            if stats_result:
                memory_stats = stats_result
        except Exception as e:
            if self.memory_test_verbose:
                print(f">> 警告：获取记忆体统计失败: {e}")

        test_answers = []

        test_timing_accumulator = {
            "pre_retrieval_ms": [],
            "memory_retrieval_ms": [],
            "post_retrieval_ms": [],
        }

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
                "memory_test_service",
                test_data,
                method="process",
                timeout=self.service_timeout,
            )

            if "answer" in result:
                if "stage_timings" in result:
                    stage_timings = result["stage_timings"]
                    for key in ["pre_retrieval_ms", "memory_retrieval_ms", "post_retrieval_ms"]:
                        if key in stage_timings:
                            test_timing_accumulator[key].append(stage_timings[key])

                answer_record = {
                    "question_index": q_idx + 1,
                    "question": question,
                    "predicted_answer": result["answer"],
                    "metadata": result.get("question_metadata", qa),
                }
                test_answers.append(answer_record)

                if self.memory_test_verbose:
                    print(f">> Question {q_idx + 1}：{question}", flush=True)
                    print(f">> Answer：{result['answer']}", flush=True)

        avg_test_timing = {}
        for key, values in test_timing_accumulator.items():
            if values:
                avg_test_timing[key] = sum(values) / len(values)
            else:
                avg_test_timing[key] = 0.0

        test_result = {
            "dataset": self.dataset,
            "task_id": task_id,
            "question_range": {
                "start": 1,
                "end": current_count,
            },
            "dialogs_inserted": self.total_dialogs_inserted,
            "answers": test_answers,
            "completed": is_last_packet,
        }

        current_count = len(self.accumulated_insert_timings["pre_insert_ms"])

        incremental_insert_timings = {
            key: values[self.sent_insert_timing_count :]
            for key, values in self.accumulated_insert_timings.items()
        }

        test_result["stage_timings"] = {
            "insert": incremental_insert_timings,
            "test": avg_test_timing,
            "memory_stats": memory_stats,
        }

        self.sent_insert_timing_count = current_count

        self.last_tested_count = current_count
        self.next_threshold_idx += 1

        if self.memory_test_verbose:
            print(f"{'+' * 60}")

        if is_last_packet and self.progress_bar:
            self.progress_bar.close()

        return test_result
