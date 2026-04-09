import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot model accuracy vs shot-transition complexity.")
    parser.add_argument("--results-dir", default="results/final_results", help="Directory containing model result xlsx files")
    parser.add_argument("--output-dir", default="analysis_outputs/shotcut_acc_by_model", help="Output directory")
    return parser.parse_args()


def extract_choice(value) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    m = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    cands = re.findall(r"\b([A-D])\b", text)
    if cands:
        return cands[-1]
    return None


def parse_duration_seconds(value) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace("—", "-").replace("–", "-")
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    start = parse_clock(left)
    end = parse_clock(right)
    if start is None or end is None or end <= start:
        return None
    return end - start


def parse_clock(value: str) -> float | None:
    parts = [p for p in str(value).strip().split(":") if p != ""]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    return None


def safe_model_name(path: Path) -> str:
    return path.stem


def compute_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    duration_col = "时间戳" if "时间戳" in df.columns else None
    gt_col = "答案_EN" if "答案_EN" in df.columns else ("答案" if "答案" in df.columns else None)
    pred_col = "Predicted_Answer" if "Predicted_Answer" in df.columns else None
    shot_col = "meta_shot_transition_count" if "meta_shot_transition_count" in df.columns else None

    if not all([duration_col, gt_col, pred_col, shot_col]):
        raise RuntimeError("Missing required columns among 时间戳/答案_EN(or答案)/Predicted_Answer/meta_shot_transition_count")

    out = pd.DataFrame()
    out["duration_sec"] = df[duration_col].map(parse_duration_seconds)
    out["shot_count"] = pd.to_numeric(df[shot_col], errors="coerce")
    out["gt_choice"] = df[gt_col].map(extract_choice)
    out["pred_choice"] = df[pred_col].map(extract_choice)
    out["is_correct"] = (out["gt_choice"].notna()) & (out["pred_choice"] == out["gt_choice"])
    out["is_eval"] = out["gt_choice"].notna() & out["shot_count"].notna()
    out["shot_per_sec"] = out["shot_count"] / out["duration_sec"]
    out["is_eval_rel"] = out["is_eval"] & out["duration_sec"].notna() & (out["duration_sec"] > 0) & out["shot_per_sec"].notna()
    return out


def aggregate_absolute(mf: pd.DataFrame) -> pd.DataFrame:
    d = mf[mf["is_eval"]].copy()
    d["bin"] = d["shot_count"].astype(int).astype(str)
    g = d.groupby("bin", as_index=False).agg(all_n=("is_correct", "size"), acc_n=("is_correct", "sum"))
    g["acc_rate"] = g["acc_n"] / g["all_n"]
    g["bin_order"] = g["bin"].astype(int)
    g = g.sort_values("bin_order").drop(columns=["bin_order"])
    return g


def aggregate_relative(mf: pd.DataFrame) -> pd.DataFrame:
    d = mf[mf["is_eval_rel"]].copy()
    bins = [-1e-9, 0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1e9]
    labels = [
        "0",
        "(0,0.05]",
        "(0.05,0.10]",
        "(0.10,0.20]",
        "(0.20,0.40]",
        "(0.40,0.80]",
        "(0.80,1.20]",
        ">1.20",
    ]
    d["bin"] = pd.cut(d["shot_per_sec"], bins=bins, labels=labels, include_lowest=True, right=True)
    g = d.groupby("bin", as_index=False, observed=True).agg(all_n=("is_correct", "size"), acc_n=("is_correct", "sum"))
    g = g[g["all_n"] > 0].copy()
    g["bin"] = g["bin"].astype(str)
    g["acc_rate"] = g["acc_n"] / g["all_n"]
    return g


def calc_ylim_top(rates: np.ndarray) -> float:
    if rates.size == 0:
        return 0.35
    max_rate = float(np.nanmax(rates))
    # Keep enough headroom for 2-line labels above bars.
    return min(1.35, max(0.35, max_rate + 0.28))


def annotate_bars(ax, rates: np.ndarray, acc_n: np.ndarray, all_n: np.ndarray, y_top: float):
    pad = max(0.015, y_top * 0.025)
    for i, (r, a, n) in enumerate(zip(rates, acc_n, all_n)):
        y = min(float(r) + pad, y_top - 0.01)
        ax.text(i, y, f"{r*100:.1f}%\n{int(a)}/{int(n)}", ha="center", va="bottom", fontsize=8, clip_on=False)


def plot_model(model_name: str, abs_df: pd.DataFrame, rel_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6), dpi=200, constrained_layout=True)

    # Absolute
    x1 = abs_df["bin"].tolist()
    y1 = abs_df["acc_rate"].to_numpy(dtype=float)
    a1 = abs_df["acc_n"].to_numpy(dtype=float)
    n1 = abs_df["all_n"].to_numpy(dtype=float)
    axes[0].bar(x1, y1, color="#4C78A8", edgecolor="none")
    axes[0].set_title(f"{model_name}\nAccuracy vs Absolute Shot Transition Count", fontsize=12, weight="bold")
    axes[0].set_xlabel("meta_shot_transition_count")
    axes[0].set_ylabel("Accuracy")
    y1_top = calc_ylim_top(y1)
    axes[0].set_ylim(0, y1_top)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].tick_params(axis="x", rotation=55, labelsize=8)
    annotate_bars(axes[0], y1, a1, n1, y1_top)

    # Relative
    x2 = rel_df["bin"].tolist()
    y2 = rel_df["acc_rate"].to_numpy(dtype=float)
    a2 = rel_df["acc_n"].to_numpy(dtype=float)
    n2 = rel_df["all_n"].to_numpy(dtype=float)
    axes[1].bar(x2, y2, color="#F58518", edgecolor="none")
    axes[1].set_title(f"{model_name}\nAccuracy vs Relative Shot Transition Rate", fontsize=12, weight="bold")
    axes[1].set_xlabel("meta_shot_transition_count / duration")
    axes[1].set_ylabel("Accuracy")
    y2_top = calc_ylim_top(y2)
    axes[1].set_ylim(0, y2_top)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    annotate_bars(axes[1], y2, a2, n2, y2_top)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("*.xlsx"))
    if not files:
        raise RuntimeError(f"No xlsx files found under {results_dir}")

    rows = []
    for p in files:
        model_name = safe_model_name(p)
        df = pd.read_excel(p)
        mf = compute_model_frame(df)
        abs_df = aggregate_absolute(mf)
        rel_df = aggregate_relative(mf)

        plot_path = out_dir / f"{model_name}_shotcut_acc.png"
        plot_model(model_name, abs_df, rel_df, plot_path)

        for _, r in abs_df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "metric_type": "absolute",
                    "bin": r["bin"],
                    "acc_n": int(r["acc_n"]),
                    "all_n": int(r["all_n"]),
                    "acc_rate": float(r["acc_rate"]),
                }
            )
        for _, r in rel_df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "metric_type": "relative",
                    "bin": r["bin"],
                    "acc_n": int(r["acc_n"]),
                    "all_n": int(r["all_n"]),
                    "acc_rate": float(r["acc_rate"]),
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "shotcut_acc_summary.csv"
    summary_json = out_dir / "shotcut_acc_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(
            {
                "results_dir": str(results_dir),
                "model_count": int(summary_df["model"].nunique()) if not summary_df.empty else 0,
                "row_count": int(len(summary_df)),
                "summary_csv": str(summary_csv),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] Per-model plots saved under: {out_dir}")
    print(f"[Done] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
