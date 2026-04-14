"""
Mini-Experiment F1 评测脚本

职责:
- 读取 .sage/benchmarks/benchmark_memory/ 下的实验结果 JSON
- 使用 token-level F1 对每个 (predicted_answer, reference_answer) 计分
- 按 dataset × structure × dimension × strategy 汇总
- 输出 Markdown 表格和 CSV 文件

用法:
    python evaluate_f1.py                         # 评测全部
    python evaluate_f1.py --dataset longmemeval   # 仅 longmemeval
    python evaluate_f1.py --csv results.csv       # 导出 CSV
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 复用项目内已有的 F1 计算逻辑
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # neuromem-bench/
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluation.analysis.utils.indicators import (  # noqa: E402
    exact_match,
    f1_score,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_ROOT = PROJECT_ROOT / ".sage" / "benchmarks" / "benchmark_memory"

# 从 memory_name 解析维度信息的正则:  MiniD2_lsh_hash_enrich / MiniD3_property_graph_crud
NAME_RE = re.compile(
    r"^Mini(?P<dim>D[23])_(?P<structure>property_graph|queue_segment|lsh_hash)_(?P<strategy>\w+)$"
)


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------


def _parse_reference(ref) -> str:
    """将 reference_answer 统一为字符串（CR 数据集的 reference 可能是 list 格式的字符串）"""
    if isinstance(ref, list):
        return ", ".join(str(r) for r in ref)
    s = str(ref).strip()
    # CR 数据某些 reference 存为 "['Belgium']" 字符串
    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, list):
                return ", ".join(str(i) for i in items)
        except (ValueError, SyntaxError):
            pass
    return s


def evaluate_single_file(json_path: Path) -> dict | None:
    """评测单个结果 JSON, 返回汇总 dict 或 None（解析失败时）"""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 解析 memory_name
    memory_name = json_path.parent.name
    m = NAME_RE.match(memory_name)
    if not m:
        return None

    dim = m.group("dim")
    structure = m.group("structure")
    strategy = m.group("strategy")
    dataset = data.get("experiment_info", {}).get("dataset", json_path.parent.parent.name)
    task_id = data.get("experiment_info", {}).get("task_id", "")

    # 收集所有 question 的 F1 / EM
    f1_scores: list[float] = []
    em_scores: list[float] = []
    per_round: dict[int, list[float]] = {}

    for tr in data.get("test_results", []):
        test_idx = tr.get("test_index", 0)
        round_f1s: list[float] = []
        for q in tr.get("questions", []):
            pred = str(q.get("predicted_answer", ""))
            ref = _parse_reference(q.get("reference_answer", ""))
            f1 = f1_score(pred, ref)
            em = exact_match(pred, ref)
            f1_scores.append(f1)
            em_scores.append(em)
            round_f1s.append(f1)
        if round_f1s:
            per_round[test_idx] = round_f1s

    if not f1_scores:
        return None

    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_em = sum(em_scores) / len(em_scores)

    # 每轮平均 F1
    round_avg = {k: sum(v) / len(v) for k, v in sorted(per_round.items())}

    return {
        "dataset": dataset,
        "dimension": dim,
        "structure": structure,
        "strategy": strategy,
        "task_id": task_id,
        "num_questions": len(f1_scores),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "round_f1": round_avg,
        "file": str(json_path.relative_to(PROJECT_ROOT)),
    }


def collect_results(dataset_filter: str | None = None) -> list[dict]:
    """遍历 RESULTS_ROOT 下所有结果 JSON 并评测"""
    results = []
    seen_keys: set[tuple] = set()  # 去重（同一实验有冒烟测试重复）

    if not RESULTS_ROOT.exists():
        print(f"结果目录不存在: {RESULTS_ROOT}", file=sys.stderr)
        return results

    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue

        for mem_dir in sorted(dataset_dir.iterdir()):
            if not mem_dir.is_dir():
                continue
            # 找最新的 JSON（按文件名排序取最后一个）
            jsons = sorted(mem_dir.glob("*.json"))
            if not jsons:
                continue
            latest = jsons[-1]  # 最新的结果
            r = evaluate_single_file(latest)
            if r is None:
                continue
            key = (r["dataset"], r["dimension"], r["structure"], r["strategy"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# 输出格式
# ---------------------------------------------------------------------------


def _sort_key(r: dict) -> tuple:
    """排序: dataset → dimension → structure → strategy"""
    struct_order = {"property_graph": 0, "queue_segment": 1, "lsh_hash": 2}
    return (
        r["dataset"],
        r["dimension"],
        struct_order.get(r["structure"], 9),
        r["strategy"],
    )


def print_markdown_table(results: list[dict]) -> None:
    """输出 Markdown 汇总表"""
    results = sorted(results, key=_sort_key)

    # --- 分 dataset 输出 ---
    datasets = sorted({r["dataset"] for r in results})
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        dims = sorted({r["dimension"] for r in ds_results})

        for dim in dims:
            dim_results = [r for r in ds_results if r["dimension"] == dim]
            dim_label = "D2 (PreInsert)" if dim == "D2" else "D3 (PostInsert)"
            print(f"\n### {ds} — {dim_label}\n")
            print(f"| {'Structure':<16} | {'Strategy':<12} | {'#Q':>4} | {'F1':>6} | {'EM':>6} |")
            print(f"|{'-' * 18}|{'-' * 14}|{'-' * 6}|{'-' * 8}|{'-' * 8}|")
            for r in dim_results:
                print(
                    f"| {r['structure']:<16} | {r['strategy']:<12} | {r['num_questions']:>4} "
                    f"| {r['avg_f1']:>5.1%} | {r['avg_em']:>5.1%} |"
                )

    # --- 全局 pivot: structure × strategy (仅 F1) ---
    print("\n### Pivot: Avg F1 (structure × strategy)\n")
    strategies_d2 = ["none", "enrich", "rewrite"]
    strategies_d3 = ["none", "crud", "forgetting"]
    structures = ["property_graph", "queue_segment", "lsh_hash"]

    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        for dim in ["D2", "D3"]:
            dim_results = [r for r in ds_results if r["dimension"] == dim]
            if not dim_results:
                continue
            strategies = strategies_d2 if dim == "D2" else strategies_d3
            dim_label = "D2 (PreInsert)" if dim == "D2" else "D3 (PostInsert)"
            print(f"\n**{ds} — {dim_label}**\n")
            header = f"| {'Structure':<16} |" + "|".join(f" {s:>12} " for s in strategies) + "|"
            sep = f"|{'-' * 18}|" + "|".join("-" * 14 for _ in strategies) + "|"
            print(header)
            print(sep)
            for struct in structures:
                row = f"| {struct:<16} |"
                for strat in strategies:
                    match = [
                        r
                        for r in dim_results
                        if r["structure"] == struct and r["strategy"] == strat
                    ]
                    if match:
                        row += f" {match[0]['avg_f1']:>11.1%} |"
                    else:
                        row += f" {'—':>12} |"
                print(row)


def write_csv(results: list[dict], path: Path) -> None:
    """导出 CSV"""
    results = sorted(results, key=_sort_key)
    fieldnames = [
        "dataset",
        "dimension",
        "structure",
        "strategy",
        "num_questions",
        "avg_f1",
        "avg_em",
        "file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            row["avg_f1"] = f"{r['avg_f1']:.4f}"
            row["avg_em"] = f"{r['avg_em']:.4f}"
            writer.writerow(row)
    print(f"\nCSV 已保存: {path}")


def print_round_progression(results: list[dict]) -> None:
    """输出多轮 (per-round) F1 变化表"""
    datasets = sorted({r["dataset"] for r in results})
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        dims = sorted({r["dimension"] for r in ds_results})

        for dim in dims:
            dim_results = sorted(
                [r for r in ds_results if r["dimension"] == dim],
                key=_sort_key,
            )
            if not dim_results:
                continue

            dim_label = "D2 (PreInsert)" if dim == "D2" else "D3 (PostInsert)"
            print(f"\n### {ds} — {dim_label} — Per-Round F1\n")

            # Collect all round IDs across all experiments
            all_rounds = set()
            for r in dim_results:
                all_rounds.update(r.get("round_f1", {}).keys())
            rounds = sorted(all_rounds)

            if not rounds:
                print("No per-round data available.\n")
                continue

            # Header: Structure | Strategy | R1 | R2 | ...
            r_headers = [f"R{i + 1}" for i in range(len(rounds))]
            header = (
                f"| {'Structure':<16} | {'Strategy':<12} |"
                + "|".join(f" {h:>6} " for h in r_headers)
                + "|"
            )
            sep = f"|{'-' * 18}|{'-' * 14}|" + "|".join("-" * 8 for _ in rounds) + "|"
            print(header)
            print(sep)

            for r in dim_results:
                row = f"| {r['structure']:<16} | {r['strategy']:<12} |"
                rf = r.get("round_f1", {})
                for rd in rounds:
                    val = rf.get(rd)
                    if val is not None:
                        row += f" {val:>5.1%} |"
                    else:
                        row += f" {'—':>6} |"
                print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Mini-Experiment F1 评测")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="仅评测指定数据集 (longmemeval / conflict_resolution)",
    )
    parser.add_argument("--csv", type=str, default=None, help="导出 CSV 文件路径")
    parser.add_argument("--json", type=str, default=None, help="导出 JSON 文件路径")
    args = parser.parse_args()

    results = collect_results(args.dataset)
    if not results:
        print("未找到任何实验结果", file=sys.stderr)
        sys.exit(1)

    print(f"共评测 {len(results)} 组实验\n")
    print_markdown_table(results)

    print("\n\n## Per-Round F1 Progression\n")
    print_round_progression(results)

    if args.csv:
        write_csv(results, Path(args.csv))

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON 已保存: {out_path}")


if __name__ == "__main__":
    main()
