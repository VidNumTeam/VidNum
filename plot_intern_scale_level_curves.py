import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ["InternVL3-8B", "InternVL3-14B", "InternVL3-38B", "InternVL3-78B"]
MODEL_X = [8, 14, 38, 78]
LEVELS = ["1", "2", "3"]
LEVEL_COLOR = {"1": "#6BAED6", "2": "#74C476", "3": "#E34A33"}


def parse_args():
    p = argparse.ArgumentParser(description="Plot level-wise scale curves for InternVL3 models.")
    p.add_argument("--results-dir", default="results", help="Directory with InternVL3 result xlsx files")
    p.add_argument("--output-dir", default="analysis_outputs/intern_scale_plots", help="Output directory")
    p.add_argument("--dpi", type=int, default=420, help="Figure dpi")
    p.add_argument("--transparent", action="store_true", help="Save transparent background")
    return p.parse_args()


def extract_choice(value) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return None

    m = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"(?:^|\n)\s*#{2,6}\s*ANSWER\s*:?\s*(.*?)(?=\n\s*#{2,6}\s*\w+|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        sec = m.group(1)
        cands = re.findall(r"\b([A-D])\b", sec)
        if cands:
            return cands[-1].upper()

    cands = re.findall(r"\b([A-D])\b", text)
    if cands:
        return cands[-1].upper()
    return None


def infer_gt_col(df: pd.DataFrame) -> str:
    for c in ["答案_EN", "答案", "Answer"]:
        if c in df.columns:
            return c

    cands = [c for c in df.columns if str(c).endswith("_EN")]
    if cands:
        scored = [(c, df[c].map(extract_choice).notna().sum()) for c in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    scored = [(c, df[c].map(extract_choice).notna().sum()) for c in df.columns]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def infer_level_col(df: pd.DataFrame) -> str:
    for c in ["LLM_Level_Final", "Level", "Hier_Level", "Hierarchy_Level"]:
        if c in df.columns:
            return c
    raise RuntimeError("No level column found.")


def build_mode_df(results_dir: Path, mode: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        path = results_dir / f"{model}_{mode}.xlsx"
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue

        df = pd.read_excel(path)
        if "Predicted_Answer" not in df.columns:
            print(f"[skip] missing Predicted_Answer: {path}")
            continue

        gt_col = infer_gt_col(df)
        level_col = infer_level_col(df)

        gt = df[gt_col].map(extract_choice)
        pred = df["Predicted_Answer"].map(extract_choice)
        ok = (gt.notna()) & (pred == gt)
        level = (
            df[level_col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.extract(r"([123])", expand=False)
            .fillna("UNKNOWN")
        )

        d = pd.DataFrame({"level": level, "ok": ok, "eval": gt.notna()})
        d = d[d["eval"]]
        for lv in LEVELS:
            g = d[d["level"] == lv]
            rows.append(
                {
                    "model": model,
                    "scale_b": int(model.split("-")[-1].replace("B", "")),
                    "mode": mode,
                    "level": lv,
                    "n": int(len(g)),
                    "acc": float(g["ok"].mean()) if len(g) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def draw_curve(df: pd.DataFrame, title: str, out_png: Path, out_pdf: Path, dpi: int, transparent: bool):
    fig, ax = plt.subplots(figsize=(3.55, 2.7), dpi=dpi)

    for lv in LEVELS:
        s = df[df["level"] == lv].sort_values("scale_b")
        x = s["scale_b"].tolist()
        y = (s["acc"] * 100.0).tolist()
        nvals = s["n"].tolist()
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.2, color=LEVEL_COLOR[lv], label=f"Level {lv}")
        for xi, yi, n in zip(x, y, nvals):
            if pd.notna(yi):
                ax.text(xi, yi + 0.8, f"{yi:.1f}", fontsize=6.8, ha="center", va="bottom", color=LEVEL_COLOR[lv])

    ax.set_xticks(MODEL_X)
    ax.set_xticklabels([str(x) for x in MODEL_X], fontsize=8)
    ax.set_xlabel("Model Scale (B)", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title, fontsize=10, pad=6)
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.7)
    ax.set_ylim(20, 65)
    ax.legend(loc="lower right", frameon=False, fontsize=7.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cot_df = build_mode_df(results_dir, "CoT")
    nocot_df = build_mode_df(results_dir, "NoCoT")
    cot_df.to_csv(out_dir / "internvl3_level_curve_cot.csv", index=False, encoding="utf-8-sig")
    nocot_df.to_csv(out_dir / "internvl3_level_curve_nocot.csv", index=False, encoding="utf-8-sig")

    if not cot_df.empty:
        draw_curve(
            cot_df,
            "InternVL3 Scale Curve by Level (CoT)",
            out_png=out_dir / "internvl3_level_curve_cot.png",
            out_pdf=out_dir / "internvl3_level_curve_cot.pdf",
            dpi=args.dpi,
            transparent=args.transparent,
        )
    if not nocot_df.empty:
        draw_curve(
            nocot_df,
            "InternVL3 Scale Curve by Level (NoCoT)",
            out_png=out_dir / "internvl3_level_curve_nocot.png",
            out_pdf=out_dir / "internvl3_level_curve_nocot.pdf",
            dpi=args.dpi,
            transparent=args.transparent,
        )
    print(f"[Done] saved to: {out_dir}")


if __name__ == "__main__":
    main()

