import argparse
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from utils import read_xlsx_to_list


DATA_PATH_DEFAULT = "question_datasets/VidNum1_4K_options_en_category_en.xlsx"
VIDEOS_ROOT_DEFAULT = "videos"
DATACUTS_ROOT_DEFAULT = "datacuts"
OSS_PATH_KEY = "OSS_path"
TIME_TAG_KEY = "Timestamp"
QUESTION_KEY = "question"


@dataclass
class CutTask:
    item: dict
    row_number: int
    start_seconds: float
    end_seconds: float
    time_tag: str
    video_path: str
    save_path: str


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_time_tag(value) -> str:
    text = normalize_text(value)
    return re.sub(r"\s+", "", text)
def build_safe_time_suffix(time_tag: str) -> str:
    normalized = normalize_time_tag(time_tag)
    return normalized.replace(":", "").replace("-", "_")


def parse_clock_to_seconds(time_part: str) -> float:
    cleaned = normalize_time_tag(time_part)
    cleaned = re.sub(r":{2,}", ":", cleaned)
    parts = [part for part in cleaned.split(":") if part != ""]

    if len(parts) not in (2, 3):
        raise ValueError(f"鏃堕棿鏍煎紡鏃犳硶瑙ｆ瀽: {time_part}")

    numbers = [float(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds

    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def split_time_range(time_tag: str) -> tuple[float, float]:
    normalized = normalize_time_tag(time_tag)
    if "-" not in normalized:
        raise ValueError(f"鏃堕棿鎴虫牸寮忛敊璇? {time_tag}")

    start_text, end_text = normalized.split("-", 1)
    start_seconds = parse_clock_to_seconds(start_text)
    end_seconds = parse_clock_to_seconds(end_text)

    if end_seconds <= start_seconds:
        raise ValueError(f"缁撴潫鏃堕棿蹇呴』澶т簬寮€濮嬫椂闂? {time_tag}")

    return start_seconds, end_seconds


def normalize_oss_path(oss_path: str) -> str:
    return normalize_text(oss_path).replace("\\", "/").lstrip("/")


def build_paths(
    item: dict,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
) -> tuple[str, str, str, str]:
    # New schema only: one video per question, stored at videos/QID_{id}.mp4
    _ = datacuts_root
    direct_video = normalize_text(item.get("Video_Path", ""))
    if not direct_video:
        item_id = normalize_text(item.get("ID", ""))
        if item_id:
            direct_video = f"QID_{item_id}.mp4"

    if not direct_video:
        raise ValueError("Missing video mapping: expected Video_Path or ID")

    direct_video = direct_video.replace("\\", "/")
    video_rel = Path(*direct_video.lstrip("/").split("/"))
    if not video_rel.suffix:
        video_rel = video_rel.with_suffix(".mp4")

    video_path = str(Path(videos_root) / video_rel)
    return str(video_rel), "", video_path, video_path
def prepare_task(
    item: dict,
    row_number: int,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
) -> CutTask:
    _, time_tag, video_path, save_path = build_paths(item, videos_root, datacuts_root)
    start_seconds, end_seconds = split_time_range(time_tag)
    return CutTask(
        item=item,
        row_number=row_number,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        time_tag=time_tag,
        video_path=video_path,
        save_path=save_path,
    )


def load_cut_tasks(
    data_path: str = DATA_PATH_DEFAULT,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
    only_missing: bool = False,
    limit: int | None = None,
) -> tuple[list[CutTask], list[str]]:
    data = read_xlsx_to_list(data_path)

    tasks: list[CutTask] = []
    errors: list[str] = []
    seen_save_paths: set[str] = set()

    for row_number, item in enumerate(data, start=2):
        if not item.get(OSS_PATH_KEY) or not item.get(TIME_TAG_KEY):
            continue

        try:
            task = prepare_task(item, row_number, videos_root, datacuts_root)
        except ValueError as exc:
            question = normalize_text(item.get(QUESTION_KEY) or item.get("question") or item.get("Question_EN")) or "鏈煡棰樼洰"
            errors.append(f"Row {row_number} | Q: {question} | 閿欒: {exc}")
            continue

        if only_missing and os.path.exists(task.save_path):
            continue

        normalized_save_path = os.path.normcase(os.path.abspath(task.save_path))
        if normalized_save_path in seen_save_paths:
            continue
        seen_save_paths.add(normalized_save_path)

        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break

    return tasks, errors


def find_missing_tasks(
    data_path: str = DATA_PATH_DEFAULT,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
    limit: int | None = None,
) -> tuple[list[CutTask], list[str]]:
    return load_cut_tasks(
        data_path=data_path,
        videos_root=videos_root,
        datacuts_root=datacuts_root,
        only_missing=True,
        limit=limit,
    )


def resolve_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ModuleNotFoundError:
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            return ffmpeg_exe
        raise ModuleNotFoundError("Missing ffmpeg. Install imageio_ffmpeg or add ffmpeg to PATH.")


def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    gpu_id: int | None,
) -> list[str]:
    ffmpeg_exe = resolve_ffmpeg_executable()
    duration = end_time - start_time

    command = [ffmpeg_exe, "-y"]
    if gpu_id is not None:
        command.extend(["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)])

    command.extend(
        [
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-threads",
            "2",
        ]
    )

    if gpu_id is not None:
        command.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "2M", "-gpu", str(gpu_id)])
    else:
        command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])

    command.extend(["-c:a", "copy", str(output_path)])
    return command


def run_ffmpeg_cut(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    gpu_id: int | None,
    max_retries: int = 3,
) -> tuple[str, str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = f"{output_path}.tmp.mp4"
    command = build_ffmpeg_command(input_path, temp_output_path, start_time, end_time, gpu_id)
    duration = max(1.0, end_time - start_time)
    timeout_seconds = min(7200, max(300, int(duration * 2 + 120)))

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
            if os.path.exists(temp_output_path):
                os.replace(temp_output_path, output_path)
            return "success", ""
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if attempt == max_retries:
                return "error", "FFmpeg 鎵ц瓒呮椂鍗℃"
        except subprocess.CalledProcessError as exc:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            error_msg = exc.stderr.decode("utf-8", errors="ignore")
            if attempt == max_retries:
                return "error", error_msg[-400:]
            time.sleep(1)
        except Exception as exc:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if attempt == max_retries:
                return "error", f"绯荤粺寮傚父: {exc}"
            time.sleep(1)

    return "error", "鏈煡閿欒"


def cut_video_accurate(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    gpu_id: int | None,
    max_retries: int = 3,
    allow_cpu_fallback: bool = True,
) -> tuple[str, str]:
    status, message = run_ffmpeg_cut(
        input_path=input_path,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        gpu_id=gpu_id,
        max_retries=max_retries,
    )
    if status == "success":
        return status, message

    if not allow_cpu_fallback or gpu_id is None:
        return status, message

    fallback_status, fallback_message = run_ffmpeg_cut(
        input_path=input_path,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        gpu_id=None,
        max_retries=1,
    )
    if fallback_status == "success":
        return "success", "GPU 澶辫触锛屽凡鍥為€€ CPU 鎴愬姛琛ュ垏"

    return "error", f"GPU 澶辫触: {message[-180:]} | CPU 鍥為€€澶辫触: {fallback_message[-180:]}"


def process_single_task(
    task: CutTask,
    task_index: int,
    allow_cpu_fallback: bool = True,
) -> tuple[str, str]:
    try:
        gpu_id = task_index % 2

        if os.path.exists(task.save_path):
            return "skipped", ""

        if not os.path.exists(task.video_path):
            return "error", f"鍘熻棰戠己澶? {task.video_path}"

        return cut_video_accurate(
            input_path=task.video_path,
            output_path=task.save_path,
            start_time=task.start_seconds,
            end_time=task.end_seconds,
            gpu_id=gpu_id,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    except Exception as exc:
        return "error", f"瑙ｆ瀽澶辫触: {exc}"


def print_missing_summary(tasks: list[CutTask], errors: list[str], sample_count: int = 10) -> None:
    print("=" * 40)
    print(f"缂哄け鍒囩墖鏁伴噺: {len(tasks)}")
    print(f"寮傚父璁板綍鏁伴噺: {len(errors)}")
    print("=" * 40)

    if tasks:
        print("缂哄け鍒囩墖绀轰緥:")
        for task in tasks[:sample_count]:
            print(f"  - Row {task.row_number}: {task.save_path}")

    if errors:
        print("寮傚父璁板綍绀轰緥:")
        for error in errors[:sample_count]:
            print(f"  - {error}")


def cut_and_save_multithread(
    data_path: str = DATA_PATH_DEFAULT,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
    only_missing: bool = False,
    max_workers: int = 16,
    limit: int | None = None,
    allow_cpu_fallback: bool = True,
) -> dict:
    mode_text = "缂哄け琛ュ垏" if only_missing else "鍏ㄩ噺鍒囩墖"
    print(f"[INFO] 姝ｅ湪璇诲彇 Excel 鏁版嵁锛屾ā寮? {mode_text}")

    tasks, precheck_errors = load_cut_tasks(
        data_path=data_path,
        videos_root=videos_root,
        datacuts_root=datacuts_root,
        only_missing=only_missing,
        limit=limit,
    )

    print("[LOG]")
    if precheck_errors:
        print("[LOG]")

    success_count = 0
    skipped_count = 0
    error_list = list(precheck_errors)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_single_task, task, index, allow_cpu_fallback): task
            for index, task in enumerate(tasks)
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="cut progress", unit="task"):
            task = future_to_task[future]
            status, message = future.result()

            if status == "success":
                success_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                question = normalize_text(task.item.get(QUESTION_KEY) or task.item.get("question") or task.item.get("Question_EN")) or "鏈煡棰樼洰"
                error_list.append(f"Row {task.row_number} | Q: {question} | 閿欒: {message}")

    print("\n" + "=" * 40)
    print("鎵归噺鍒囩墖浠诲姟缁撴潫")
    print("=" * 40)
    print("[LOG]")
    print("[LOG]")
    print("[LOG]")

    if error_list:
        print("閿欒璇︽儏鎽樿:")
        for error in error_list[:10]:
            print(f"  - {error}")

    return {
        "success_count": success_count,
        "skipped_count": skipped_count,
        "error_list": error_list,
        "task_count": len(tasks),
    }


def repair_missing_cuts(
    data_path: str = DATA_PATH_DEFAULT,
    videos_root: str = VIDEOS_ROOT_DEFAULT,
    datacuts_root: str = DATACUTS_ROOT_DEFAULT,
    max_workers: int = 16,
    limit: int | None = None,
    allow_cpu_fallback: bool = True,
) -> dict:
    return cut_and_save_multithread(
        data_path=data_path,
        videos_root=videos_root,
        datacuts_root=datacuts_root,
        only_missing=True,
        max_workers=max_workers,
        limit=limit,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video cut utility")
    parser.add_argument("--data-path", default=DATA_PATH_DEFAULT, help="Excel file path")
    parser.add_argument("--videos-root", default=VIDEOS_ROOT_DEFAULT, help="Video root directory")
    parser.add_argument("--datacuts-root", default=DATACUTS_ROOT_DEFAULT, help="Output directory for cuts")
    parser.add_argument("--check-only", action="store_true", help="Only check missing cuts")
    parser.add_argument("--only-missing", action="store_true", help="Process only missing outputs")
    parser.add_argument("--max-workers", type=int, default=16, help="Worker count")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--disable-cpu-fallback", action="store_true", help="Disable CPU fallback")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check_only:
        missing_tasks, invalid_errors = find_missing_tasks(
            data_path=args.data_path,
            videos_root=args.videos_root,
            datacuts_root=args.datacuts_root,
            limit=args.limit,
        )
        print_missing_summary(missing_tasks, invalid_errors)
    else:
        cut_and_save_multithread(
            data_path=args.data_path,
            videos_root=args.videos_root,
            datacuts_root=args.datacuts_root,
            only_missing=args.only_missing,
            max_workers=args.max_workers,
            limit=args.limit,
            allow_cpu_fallback=not args.disable_cpu_fallback,
        )


