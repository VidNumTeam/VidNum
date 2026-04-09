import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "analysis_outputs/level_aoe_stats"
DEFAULT_MODELS = [
    "InternVL2_5-8B",
    "InternVL3-8B",
    "InternVL3_5-8B",
    "LLaVA-NeXT-7B",
    "Qwen2.5-VL-7B",
    "Qwen3-VL-8B",
    "InternVL3-14B",
    "InternVL3-38B",
    "InternVL3-78B",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model accuracy by level and A/O/E.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="comma-separated model names")
    parser.add_argument("--include-videollama3", action="store_true", help="include VideoLLaMA3-7B if provided in --models")
    return parser.parse_args()


def normalize_choice(x):
    if pd.isna(x):
        return None
    s = str(x)
    tag = re.search(r"<answer>(.*?)</answer>", s, flags=re.I | re.S)
    if tag:
        s = tag.group(1)
    sec = re.search(
        r"(?:^|\n)\s*#{2,6}\s*Answer\s*:?\s*(.*?)(?=\n\s*#{2,6}\s*\w+|\Z)",
        s,
        flags=re.I | re.S,
    )
    if sec:
        s = sec.group(1)
    hits = re.findall(r"\b([A-D])\b", s.upper())
    return hits[-1] if hits else None


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_gt_col(df: pd.DataFrame):
    return "correct_answer" if "correct_answer" in df.columns else None


def normalize_level(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"([123])", s)
    return int(m.group(1)) if m else None


def evaluate_file(path: Path):
    df = pd.read_excel(path)
    if "Predicted_Answer" not in df.columns:
        return None, "missing Predicted_Answer"
    gt_col = infer_gt_col(df)
    if gt_col is None:
        return None, "missing correct_answer"
    level_col = find_col(df, ["Level"])
    if level_col is None:
        return None, "missing Level"
    aoe_col = find_col(df, ["Counting_Type"])
    if aoe_col is None:
        return None, "missing Counting_Type"

    work = df[[gt_col, "Predicted_Answer", level_col, aoe_col]].copy()
    work["gt"] = work[gt_col].map(normalize_choice)
    work["pred"] = work["Predicted_Answer"].map(normalize_choice)
    work["level"] = work[level_col].map(normalize_level)
    work["aoe"] = work[aoe_col].astype(str).str.strip().str.lower()
    work = work[work["aoe"].isin(["action", "object", "event"])]
    work = work.dropna(subset=["gt", "pred", "level"]).copy()
    if len(work) == 0:
        return None, "no evaluable rows"

    work["ok"] = work["gt"] == work["pred"]

    overall = {
        "n": int(len(work)),
        "correct": int(work["ok"].sum()),
        "acc": float(work["ok"].mean()),
    }

    by_level = (
        work.groupby("level")["ok"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n", "sum": "correct", "mean": "acc"})
        .reset_index()
    )
    by_level["level"] = by_level["level"].astype(int)

    by_level_aoe = (
        work.groupby(["level", "aoe"])["ok"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n", "sum": "correct", "mean": "acc"})
        .reset_index()
    )
    by_level_aoe["level"] = by_level_aoe["level"].astype(int)

    return {
        "overall": overall,
        "by_level": by_level,
        "by_level_aoe": by_level_aoe,
    }, None


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.include_videollama3:
        models = [m for m in models if m.lower() != "videollama3-7b"]

    overall_rows = []
    by_level_rows = []
    by_level_aoe_rows = []
    skipped_rows = []

    for model in models:
        for mode in ["CoT", "NoCoT"]:
            file_path = results_dir / f"{model}_{mode}.xlsx"
            if not file_path.exists():
                skipped_rows.append({"model": model, "mode": mode, "file": str(file_path), "reason": "file_not_found"})
                continue
            res, err = evaluate_file(file_path)
            if err:
                skipped_rows.append({"model": model, "mode": mode, "file": str(file_path), "reason": err})
                continue

            overall_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    **res["overall"],
                }
            )

            lvl = res["by_level"].copy()
            lvl["model"] = model
            lvl["mode"] = mode
            by_level_rows.append(lvl)

            lvl_aoe = res["by_level_aoe"].copy()
            lvl_aoe["model"] = model
            lvl_aoe["mode"] = mode
            by_level_aoe_rows.append(lvl_aoe)

    overall_df = pd.DataFrame(overall_rows)
    by_level_df = pd.concat(by_level_rows, ignore_index=True) if by_level_rows else pd.DataFrame()
    by_level_aoe_df = pd.concat(by_level_aoe_rows, ignore_index=True) if by_level_aoe_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows)

    overall_df.to_csv(output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    by_level_df.to_csv(output_dir / "accuracy_by_level.csv", index=False, encoding="utf-8-sig")
    by_level_aoe_df.to_csv(output_dir / "accuracy_by_level_aoe.csv", index=False, encoding="utf-8-sig")
    skipped_df.to_csv(output_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")

    print(f"[Done] saved to: {output_dir}")
    if not overall_df.empty:
        show = overall_df.copy()
        show["acc"] = (show["acc"] * 100).round(2)
        show = show[["model", "mode", "correct", "acc"]]
        print("\nOverall accuracy (%):")
        print(show.sort_values(["model", "mode"]).to_string(index=False))
    if not by_level_df.empty:
        show_level = by_level_df.copy()
        show_level["acc"] = (show_level["acc"] * 100).round(2)
        show_level = show_level[["model", "mode", "level", "correct", "acc"]]
        print("\nAccuracy by level (%):")
        print(show_level.sort_values(["model", "mode", "level"]).to_string(index=False))
    if not by_level_aoe_df.empty:
        show_level_aoe = by_level_aoe_df.copy()
        show_level_aoe["acc"] = (show_level_aoe["acc"] * 100).round(2)
        show_level_aoe = show_level_aoe[["model", "mode", "level", "aoe", "correct", "acc"]]
        print("\nAccuracy by level and A/O/E (%):")
        print(show_level_aoe.sort_values(["model", "mode", "level", "aoe"]).to_string(index=False))
    if not skipped_df.empty:
        print("\nSkipped:")
        print(skipped_df.to_string(index=False))


if __name__ == "__main__":
    main()


