import argparse
import math
import os
import random
from collections import Counter
from statistics import mean, median

from utils import read_xlsx_to_list, write_list_to_xlsx
from videocut_multithread import (
    DATACUTS_ROOT_DEFAULT,
    DATA_PATH_DEFAULT,
    TIME_TAG_KEY,
    build_paths,
    find_missing_tasks,
    repair_missing_cuts,
    split_time_range,
)


QUESTION_TYPE_KEY = "问题类型"
QUESTION_KEY = "Question_EN"
ANSWER_KEY = "答案_EN"
CHOICE_KEYS = {choice: f"选项{choice}" for choice in "ABCD"}


def get_questions(data_path: str = DATA_PATH_DEFAULT):
    return read_xlsx_to_list(data_path)


def is_multiple_choice(item: dict) -> bool:
    question_type = str(item.get(QUESTION_TYPE_KEY, "") or "").strip()
    if question_type == "选择题":
        return True
    return any(str(item.get(key, "") or "").strip() for key in CHOICE_KEYS.values())


def count_choice_answers(data_path: str = DATA_PATH_DEFAULT, data: list[dict] | None = None) -> dict[str, int]:
    counter = Counter({choice: 0 for choice in "ABCD"})
    rows = data if data is not None else get_questions(data_path)
    for item in rows:
        if not is_multiple_choice(item):
            continue
        answer = str(item.get(ANSWER_KEY, "") or "").strip().upper()
        if answer in counter:
            counter[answer] += 1
    return dict(counter)


def count_clip_duration_distribution(
    data_path: str = DATA_PATH_DEFAULT,
    data: list[dict] | None = None,
    bucket_edges: list[float] | None = None,
) -> dict:
    rows = data if data is not None else get_questions(data_path)
    bucket_edges = bucket_edges or [5, 10, 20, 30, 60, 120]
    durations: list[float] = []
    invalid_rows: list[int] = []

    for row_number, item in enumerate(rows, start=2):
        time_tag = item.get(TIME_TAG_KEY)
        if not time_tag:
            continue
        try:
            start_seconds, end_seconds = split_time_range(str(time_tag))
        except ValueError:
            invalid_rows.append(row_number)
            continue
        durations.append(end_seconds - start_seconds)

    bucket_counts: dict[str, int] = {}
    lower = 0.0
    for upper in bucket_edges:
        label = f"{int(lower)}-{int(upper)}s"
        bucket_counts[label] = sum(lower <= duration < upper for duration in durations)
        lower = upper
    bucket_counts[f">={int(lower)}s"] = sum(duration >= lower for duration in durations)

    if not durations:
        return {
            "count": 0,
            "invalid_rows": invalid_rows,
            "buckets": bucket_counts,
        }

    sorted_durations = sorted(durations)
    percentile_90 = sorted_durations[min(len(sorted_durations) - 1, math.ceil(len(sorted_durations) * 0.9) - 1)]
    return {
        "count": len(durations),
        "invalid_rows": invalid_rows,
        "summary": {
            "min_seconds": round(sorted_durations[0], 2),
            "max_seconds": round(sorted_durations[-1], 2),
            "mean_seconds": round(mean(durations), 2),
            "median_seconds": round(median(durations), 2),
            "p90_seconds": round(percentile_90, 2),
        },
        "buckets": bucket_counts,
    }


def filter_questions_by_duration(
    data_path: str = DATA_PATH_DEFAULT,
    output_path: str | None = None,
    min_seconds: float = 3.0,
    max_seconds: float = 120.0,
) -> dict:
    data = get_questions(data_path)
    kept_rows: list[dict] = []
    filtered_short_rows: list[int] = []
    filtered_long_rows: list[int] = []
    invalid_rows: list[int] = []

    for row_number, item in enumerate(data, start=2):
        time_tag = item.get(TIME_TAG_KEY)
        if not time_tag:
            invalid_rows.append(row_number)
            continue
        try:
            start_seconds, end_seconds = split_time_range(str(time_tag))
        except ValueError:
            invalid_rows.append(row_number)
            continue

        duration = end_seconds - start_seconds
        if duration < min_seconds:
            filtered_short_rows.append(row_number)
            continue
        if duration > max_seconds:
            filtered_long_rows.append(row_number)
            continue
        kept_rows.append(item)

    output_path = output_path or f"{os.path.splitext(data_path)[0]}_{int(min_seconds)}s_to_{int(max_seconds)}s.xlsx"
    write_list_to_xlsx(kept_rows, output_path)
    return {
        "output_path": output_path,
        "original_count": len(data),
        "kept_count": len(kept_rows),
        "removed_short_count": len(filtered_short_rows),
        "removed_long_count": len(filtered_long_rows),
        "invalid_count": len(invalid_rows),
        "removed_total": len(filtered_short_rows) + len(filtered_long_rows) + len(invalid_rows),
        "short_rows": filtered_short_rows,
        "long_rows": filtered_long_rows,
        "invalid_rows": invalid_rows,
    }


def choose_target_counts(total: int, rng: random.Random) -> dict[str, int]:
    target = {choice: total // 4 for choice in "ABCD"}
    for choice in rng.sample(list("ABCD"), total % 4):
        target[choice] += 1
    return target


def weighted_pick(weights: dict[str, int], rng: random.Random) -> str:
    total = sum(max(weight, 0) for weight in weights.values())
    if total <= 0:
        return rng.choice(list(weights.keys()))

    threshold = rng.uniform(0, total)
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += max(weight, 0)
        if cumulative >= threshold:
            return key
    return next(iter(weights))


def rebalance_choice_answers(
    data_path: str = DATA_PATH_DEFAULT,
    output_path: str | None = None,
    seed: int = 42,
) -> tuple[str, dict[str, int], dict[str, int]]:
    rng = random.Random(seed)
    data = get_questions(data_path)
    before_counts = count_choice_answers(data=data)

    mc_indices = [
        index
        for index, item in enumerate(data)
        if is_multiple_choice(item) and str(item.get(ANSWER_KEY, "") or "").strip().upper() in CHOICE_KEYS
    ]
    target_counts = choose_target_counts(len(mc_indices), rng)
    assigned_counts = Counter({choice: 0 for choice in "ABCD"})

    shuffled_indices = mc_indices[:]
    rng.shuffle(shuffled_indices)

    for index in shuffled_indices:
        item = data[index]
        correct_choice = str(item.get(ANSWER_KEY, "") or "").strip().upper()
        options = {choice: item.get(key) for choice, key in CHOICE_KEYS.items()}
        correct_value = options[correct_choice]
        distractors = [(choice, value) for choice, value in options.items() if choice != correct_choice]
        rng.shuffle(distractors)

        deficits = {
            choice: target_counts[choice] - assigned_counts[choice]
            for choice in "ABCD"
        }
        new_answer = weighted_pick(deficits, rng)
        assigned_counts[new_answer] += 1

        remaining_slots = [choice for choice in "ABCD" if choice != new_answer]
        rng.shuffle(remaining_slots)

        item[CHOICE_KEYS[new_answer]] = correct_value
        for slot, (_, distractor_value) in zip(remaining_slots, distractors):
            item[CHOICE_KEYS[slot]] = distractor_value
        item[ANSWER_KEY] = new_answer

    output_path = output_path or os.path.splitext(data_path)[0] + "_choice_rebalanced.xlsx"
    write_list_to_xlsx(data, output_path)
    after_counts = count_choice_answers(data= data)
    return output_path, before_counts, after_counts


def get_question(index: int, data: list[dict], datacuts_root: str = DATACUTS_ROOT_DEFAULT):
    item = data[index]
    video_path = build_paths(item, datacuts_root=datacuts_root)[3]

    missing = 0
    if not video_path or not os.path.exists(video_path):
        print(f"File doesn't exist -> {index}")
        print(f"video path: {video_path}")
        missing = 1

    question = {
        "Question": item.get(QUESTION_KEY, ""),
        "answer": item.get(ANSWER_KEY, ""),
        "video_path": video_path,
    }
    if is_multiple_choice(item):
        question["Choices"] = [item.get(CHOICE_KEYS[choice], "") for choice in "ABCD"]
    return question, missing


def get_question_by_id(
    question_id: int,
    data: list[dict] | None = None,
    data_path: str = DATA_PATH_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
):
    rows = data if data is not None else get_questions(data_path)
    for index, item in enumerate(rows):
        item_id = item.get("ID")
        try:
            normalized_id = int(item_id)
        except (TypeError, ValueError):
            continue
        if normalized_id == question_id:
            question, missing = get_question(index, rows, datacuts_root)
            return {
                "index": index,
                "id": normalized_id,
                "missing": missing,
                "question": question,
            }
    raise ValueError(f"Question ID {question_id} not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查缺失切片、补切，或处理选择题分布")
    parser.add_argument("--data-path", default=DATA_PATH_DEFAULT, help="Excel 路径")
    parser.add_argument("--datacuts-root", default=DATACUTS_ROOT_DEFAULT, help="切片输出根目录")
    parser.add_argument("--videos-root", default="videos", help="原视频根目录")
    parser.add_argument("--repair", action="store_true", help="对缺失切片执行补切")
    parser.add_argument("--count-choices", action="store_true", help="统计选择题答案的 A/B/C/D 分布")
    parser.add_argument("--count-durations", action="store_true", help="统计视频切片时长分布")
    parser.add_argument("--filter-durations", action="store_true", help="按时长范围筛选题目并保存新数据集")
    parser.add_argument("--rebalance-choices", action="store_true", help="随机重排选择题选项并尽量拉平 A/B/C/D 分布")
    parser.add_argument("--question-id", type=int, default=None, help="按数据集里的 ID 字段读取单题")
    parser.add_argument("--output-path", default=None, help="重排后 Excel 输出路径")
    parser.add_argument("--min-seconds", type=float, default=3.0, help="筛选保留的最小时长")
    parser.add_argument("--max-seconds", type=float, default=120.0, help="筛选保留的最大时长")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-workers", type=int, default=16, help="补切线程数")
    parser.add_argument("--limit", type=int, default=None, help="限制检查或补切数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.count_choices:
        print(count_choice_answers(args.data_path))
    elif args.question_id is not None:
        print(
            get_question_by_id(
                question_id=args.question_id,
                data_path=args.data_path,
                datacuts_root=args.datacuts_root,
            )
        )
    elif args.count_durations:
        print(count_clip_duration_distribution(args.data_path))
    elif args.filter_durations:
        print(
            filter_questions_by_duration(
                data_path=args.data_path,
                output_path=args.output_path,
                min_seconds=args.min_seconds,
                max_seconds=args.max_seconds,
            )
        )
    elif args.rebalance_choices:
        output_path, before_counts, after_counts = rebalance_choice_answers(
            data_path=args.data_path,
            output_path=args.output_path,
            seed=args.seed,
        )
        print(f"Saved rebalanced dataset to: {output_path}")
        print(f"Before: {before_counts}")
        print(f"After:  {after_counts}")
    elif args.repair:
        repair_missing_cuts(
            data_path=args.data_path,
            videos_root=args.videos_root,
            datacuts_root=args.datacuts_root,
            max_workers=args.max_workers,
            limit=args.limit,
        )
    else:
        missing_tasks, invalid_errors = find_missing_tasks(
            data_path=args.data_path,
            videos_root=args.videos_root,
            datacuts_root=args.datacuts_root,
            limit=args.limit,
        )
        print(f"Missing {len(missing_tasks)} video cuts.")
        if invalid_errors:
            print(f"Invalid rows: {len(invalid_errors)}")
