"""MemorySink - 收集测试结果并保存"""

import json
import os

from sage.foundation import SinkFunction

from benchmarks.experiment.utils import (
    DataLoaderFactory,
    get_project_root,
    get_runtime_timestamp,
)


class MemorySink(SinkFunction):
    """收集测试结果并保存为 JSON 格式的 Sink"""

    def __init__(self, config):
        self.dataset = config.get("dataset")
        self.task_id = config.get("task_id")
        self.test_segments = config.get("runtime.test_segments", 10)
        self.memory_name = config.get("runtime.memory_name", "default")

        project_root = get_project_root()
        self.output_dir = os.path.join(
            project_root,
            f".sage/benchmarks/benchmark_memory/{self.dataset}/{self.memory_name}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        runtime_stamp = get_runtime_timestamp()
        self.output_file = os.path.join(self.output_dir, f"{self.task_id}_{runtime_stamp}.json")
        print(f"💾 输出文件: {self.output_file}")

        self.test_results = []
        self.all_insert_timings = []
        self.all_test_timings = []
        self.all_memory_stats = []
        self.loader = DataLoaderFactory.create(self.dataset)

    def execute(self, data):
        if not data:
            return

        if "answers" in data or "stage_timings" in data:
            if "answers" in data:
                test_result = {
                    "test_index": len(self.test_results) + 1,
                    "question_range": data.get("question_range"),
                    "dialogs_inserted_count": data.get("dialogs_inserted"),
                    "answers": data.get("answers", []),
                }
                self.test_results.append(test_result)

            if "stage_timings" in data:
                stage_timings = data["stage_timings"]
                if "insert" in stage_timings:
                    insert_timing = stage_timings["insert"]
                    if insert_timing:
                        first_key = next(iter(insert_timing.keys()))
                        if isinstance(insert_timing[first_key], list):
                            list_len = len(insert_timing[first_key])
                            for i in range(list_len):
                                single_timing = {k: v[i] for k, v in insert_timing.items()}
                                self.all_insert_timings.append(single_timing)
                        else:
                            self.all_insert_timings.append(insert_timing)

                if "test" in stage_timings:
                    test_timing = stage_timings["test"]
                    if test_timing:
                        self.all_test_timings.append(test_timing)

            if "stage_timings" in data and "memory_stats" in data["stage_timings"]:
                memory_stats = data["stage_timings"]["memory_stats"]
                if memory_stats:
                    self.all_memory_stats.append(memory_stats)

        if data.get("completed", False):
            self._save_results(data)

    def _save_results(self, data):
        dataset = data.get("dataset", self.dataset)
        task_id = data.get("task_id", self.task_id)
        dataset_stats = self.loader.statistics(task_id)

        timing_summary = {
            "insert_timings": self._format_insert_timings(),
            "retrieval_timings": self._format_retrieval_timings(),
        }

        memory_snapshots = self._format_memory_snapshots()

        output_data = {
            "experiment_info": {"dataset": dataset, "task_id": task_id},
            "dataset_statistics": dataset_stats,
            "test_summary": {
                "total_tests": len(self.test_results),
                "test_segments": self.test_segments,
                "test_threshold": f"1/{self.test_segments} of total questions",
            },
            "test_results": self._format_test_results(self.test_results),
            "timing_summary": timing_summary,
            "memory_snapshots": memory_snapshots,
        }

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 测试结果已保存至: {self.output_file}")
        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")

    def _format_test_results(self, test_results):
        formatted_results = []
        for test in test_results:
            formatted_test = {
                "test_index": test.get("test_index"),
                "question_range": test.get("question_range"),
                "dialogs_inserted_count": test.get("dialogs_inserted_count"),
                "questions": [],
            }
            for answer in test.get("answers", []):
                metadata = answer.get("metadata", {})
                question_data = {
                    "question_index": answer.get("question_index"),
                    "question_text": answer.get("question"),
                    "predicted_answer": answer.get("predicted_answer"),
                }
                if "answer" in metadata:
                    question_data["reference_answer"] = metadata["answer"]
                if "evidence" in metadata:
                    question_data["evidence"] = metadata["evidence"]
                if "category" in metadata:
                    question_data["category"] = metadata["category"]
                if "error" in answer:
                    question_data["error"] = answer["error"]
                formatted_test["questions"].append(question_data)
            formatted_results.append(formatted_test)
        return formatted_results

    def _format_insert_timings(self) -> dict:
        if not self.all_insert_timings:
            return {"summary": {}, "details": []}
        summary = {}
        for stage_name in ["pre_insert_ms", "memory_insert_ms", "post_insert_ms"]:
            values = [timing.get(stage_name, 0) for timing in self.all_insert_timings]
            if values:
                summary[stage_name] = {
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "count": len(values),
                }
        return {"summary": summary, "details": self.all_insert_timings}

    def _format_retrieval_timings(self) -> dict:
        if not self.all_test_timings:
            return {"summary": {}, "details": []}
        summary = {}
        for stage_name in ["pre_retrieval_ms", "memory_retrieval_ms", "post_retrieval_ms"]:
            values = [timing.get(stage_name, 0) for timing in self.all_test_timings]
            if values:
                summary[stage_name] = {
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "count": len(values),
                }
        details = []
        for idx, timing in enumerate(self.all_test_timings, start=1):
            detail = {"test_index": idx}
            detail.update(timing)
            details.append(detail)
        return {"summary": summary, "details": details}

    def _format_memory_snapshots(self) -> list[dict]:
        if not self.all_memory_stats:
            return []
        snapshots = []
        for idx, stats in enumerate(self.all_memory_stats, start=1):
            snapshot = {"test_index": idx}
            snapshot.update(stats)
            snapshots.append(snapshot)
        return snapshots
