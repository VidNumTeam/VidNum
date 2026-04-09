"""
Combined CoT-vs-NoCoT figure by level:
- left panel per level: stacked proportion (both_correct / cot_helps / cot_hurts / both_wrong)
- right panel per level: overall CoT gain (pp)
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FLIP_COLORS = {
    "both_correct": "#0072B2",
    "cot_helps": "#009E73",
    "cot_hurts": "#D55E00",
    "both_wrong": "#E0E0E0",
}
FLIP_LABELS = {
    "both_correct": "Both correct",
    "cot_helps": "CoT helps",
    "cot_hurts": "CoT hurts",
    "both_wrong": "Both wrong",
}
STACK_ORDER = ["both_correct", "cot_helps", "cot_hurts", "both_wrong"]

COL_POS = "#56B4E9"
COL_NEG = "#CC79A7"
COL_ZERO = "#AAAAAA"

LEVELS = [1, 2, 3]
DEFAULT_EXCL = ["videollama3"]
GEMINI_KEY = "gemini"


def norm_choice(x) -> Optional[str]:
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


def infer_gt_col(df: pd.DataFrame) -> str:
    for c in ["答案_EN", "答案", "Answer"]:
        if c in df.columns:
            return c
    cands = [c for c in df.columns if str(c).endswith("_EN")]
    if cands:
        scored = [(c, df[c].map(norm_choice).notna().sum()) for c in cands]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    scored = [(c, df[c].map(norm_choice).notna().sum()) for c in df.columns]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def infer_level_col(df: pd.DataFrame) -> str:
    for n in ("LLM_Level_Final", "Level", "Hier_Level", "Hierarchy_Level"):
        if n in df.columns:
            return n
    raise RuntimeError("No level column found.")


def normalize_level(x) -> Optional[int]:
    m = re.search(r"([123])", str(x).strip()) if not pd.isna(x) else None
    return int(m.group(1)) if m else None


def collect_pairs(results_dir: Path, exclude: List[str]):
    pairs = []
    for cot in sorted(results_dir.glob("*_CoT.xlsx")):
        model = cot.name[: -len("_CoT.xlsx")]
        if any(s and s in model.lower() for s in exclude):
            continue
        nocot = results_dir / f"{model}_NoCoT.xlsx"
        if nocot.exists():
            pairs.append((model, cot, nocot))
    return pairs


def shorten(name: str) -> str:
    subs = [
        ("InternVL2_5", "IVL2.5"),
        ("InternVL3_5", "IVL3.5"),
        ("InternVL3", "IVL3"),
        ("InternVL2", "IVL2"),
        ("Qwen2.5-VL", "Qwen2.5-VL"),
        ("Qwen3-VL", "Qwen3-VL"),
        ("LLaVA-NeXT", "LLaVA-NeXT"),
        ("gemini-3-flash-preview", "Gemini-3-Flash"),
        ("gemini-3.1-pro-preview", "Gemini-3.1-Pro"),
    ]
    for old, new in subs:
        name = name.replace(old, new)
    return name


def compute_all(cot_path: Path, nocot_path: Path) -> Dict[int, dict]:
    cot_df = pd.read_excel(cot_path)
    nocot_df = pd.read_excel(nocot_path)
    gt_col = infer_gt_col(cot_df)
    level_col = infer_level_col(cot_df)

    base = cot_df[["ID", "Predicted_Answer", gt_col, level_col]].copy().rename(columns={"Predicted_Answer": "pred_cot"})
    side = nocot_df[["ID", "Predicted_Answer"]].copy().rename(columns={"Predicted_Answer": "pred_nocot"})
    for df in (base, side):
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    merged = (
        base.dropna(subset=["ID"]).drop_duplicates("ID").merge(
            side.dropna(subset=["ID"]).drop_duplicates("ID"), on="ID", how="inner"
        )
    )
    merged["gt"] = merged[gt_col].map(norm_choice)
    merged["cot"] = merged["pred_cot"].map(norm_choice)
    merged["nocot"] = merged["pred_nocot"].map(norm_choice)
    merged["level"] = merged[level_col].map(normalize_level)
    merged = merged.dropna(subset=["gt", "cot", "nocot", "level"]).copy()
    merged["level"] = merged["level"].astype(int)
    merged["cot_ok"] = merged["cot"] == merged["gt"]
    merged["nocot_ok"] = merged["nocot"] == merged["gt"]

    out = {}
    for lv in LEVELS:
        sub = merged[merged["level"] == lv]
        n = len(sub)
        if n == 0:
            out[lv] = {k: float("nan") for k in STACK_ORDER + ["gain_pp", "n"]}
            continue
        bc = (sub["cot_ok"] & sub["nocot_ok"]).sum()
        ch = (sub["cot_ok"] & ~sub["nocot_ok"]).sum()
        cr = (~sub["cot_ok"] & sub["nocot_ok"]).sum()
        bw = (~sub["cot_ok"] & ~sub["nocot_ok"]).sum()
        gain = (sub["cot_ok"].mean() - sub["nocot_ok"].mean()) * 100
        out[lv] = dict(
            both_correct=bc / n,
            cot_helps=ch / n,
            cot_hurts=cr / n,
            both_wrong=bw / n,
            gain_pp=gain,
            n=n,
        )
    return out


def build_order(all_data: dict, ref_level: int = 2) -> List[str]:
    def net(m):
        v = all_data[m][ref_level].get("gain_pp", -999)
        return v if not np.isnan(v) else -999

    gemini = sorted([m for m in all_data if GEMINI_KEY in m.lower()], key=net)
    others = sorted([m for m in all_data if GEMINI_KEY not in m.lower()], key=net)
    return gemini + others


def plot(all_data: dict, output_dir: Path, dpi: int) -> None:
    model_order = build_order(all_data)
    short_names = [shorten(m) for m in model_order]
    n_models = len(model_order)
    y_pos = np.arange(n_models)

    gain_vals = [all_data[m][lv]["gain_pp"] for m in all_data for lv in LEVELS if not np.isnan(all_data[m][lv]["gain_pp"])]
    gain_xlim = max(8.0, float(np.ceil(max(abs(min(gain_vals, default=0)), abs(max(gain_vals, default=0))) / 2) * 2))

    bar_h = 0.55
    fig_w = 17.0
    fig_h = max(5.0, n_models * 0.46 + 2.2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1,
        11,
        width_ratios=[4, 0.45, 1, 0.6, 4, 0.45, 1, 0.6, 4, 0.45, 1],
        wspace=0.05,
        left=0.10,
        right=0.98,
        top=0.91,
        bottom=0.12,
    )
    col_map = {0: (0, 2), 1: (4, 6), 2: (8, 10)}
    ax_flip0 = fig.add_subplot(gs[0, 0])
    ax_pairs = [(ax_flip0, fig.add_subplot(gs[0, 2], sharey=ax_flip0))]
    for li in [1, 2]:
        fc, gc = col_map[li]
        af = fig.add_subplot(gs[0, fc], sharey=ax_flip0)
        ag = fig.add_subplot(gs[0, gc], sharey=ax_flip0)
        ax_pairs.append((af, ag))

    n_gemini = sum(1 for m in model_order if GEMINI_KEY in m.lower())

    for li, (ax_f, ax_g) in enumerate(ax_pairs):
        lv = LEVELS[li]
        lefts = np.zeros(n_models)
        for cat in STACK_ORDER:
            vals = np.array([(all_data[m][lv].get(cat, 0) or 0) * 100 for m in model_order])
            ax_f.barh(y_pos, vals, left=lefts, height=bar_h, color=FLIP_COLORS[cat], linewidth=0, zorder=2)
            lefts += vals

        ax_f.axvline(50, color="white", linewidth=0.8, zorder=3, alpha=0.85)
        ax_f.axhline(n_gemini - 0.5, color="#999", linewidth=0.55, linestyle="--", zorder=4)
        ax_f.set_xlim(0, 100)
        ax_f.set_ylim(-0.5, n_models - 0.5)
        ax_f.set_xlabel("Proportion (%)", labelpad=3)
        ax_f.set_xticks([0, 25, 50, 75, 100])
        ax_f.xaxis.grid(True, linewidth=0.3, color="#CCCCCC", zorder=0)
        ax_f.set_axisbelow(True)

        if li == 0:
            ax_f.set_yticks(y_pos)
            ax_f.set_yticklabels(short_names)
        else:
            ax_f.tick_params(axis="y", left=False, labelleft=False)
            ax_f.spines["left"].set_visible(False)

        gains = np.array([all_data[m][lv].get("gain_pp", 0) or 0 for m in model_order])
        colors = [COL_POS if g > 0 else (COL_NEG if g < 0 else COL_ZERO) for g in gains]
        ax_g.barh(y_pos, gains, height=bar_h, color=colors, linewidth=0, zorder=2)
        for i, g in enumerate(gains):
            if np.isnan(g):
                continue
            offset = 0.25 if g >= 0 else -0.25
            ha = "left" if g >= 0 else "right"
            ax_g.text(g + offset, i, f"{g:+.1f}", va="center", ha=ha, fontsize=8, color="#333333")

        ax_g.axvline(0, color="black", linewidth=0.9, zorder=4)
        ax_g.axhline(n_gemini - 0.5, color="#999", linewidth=0.55, linestyle="--", zorder=4)
        ax_g.axvspan(0, gain_xlim, alpha=0.05, color=COL_POS, zorder=0)
        ax_g.axvspan(-gain_xlim, 0, alpha=0.05, color=COL_NEG, zorder=0)
        ax_g.set_xlim(-gain_xlim, gain_xlim)
        ax_g.set_xlabel("Gain (pp)", labelpad=3)
        ax_g.set_xticks(np.linspace(-gain_xlim, gain_xlim, 5))
        ax_g.xaxis.grid(True, linewidth=0.3, color="#DDDDDD", zorder=0)
        ax_g.set_axisbelow(True)
        ax_g.tick_params(axis="y", left=False, labelleft=False)
        ax_g.spines["left"].set_visible(False)

        fig.text(
            0.5 * (ax_f.get_position().x0 + ax_g.get_position().x1),
            ax_f.get_position().y1 + 0.025,
            f"Level {lv}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            transform=fig.transFigure,
        )

    flip_handles = [mpatches.Patch(facecolor=FLIP_COLORS[k], label=FLIP_LABELS[k], linewidth=0) for k in STACK_ORDER]
    gain_handles = [
        mpatches.Patch(facecolor=COL_POS, label="Overall CoT gain > 0", linewidth=0),
        mpatches.Patch(facecolor=COL_NEG, label="Overall CoT gain < 0", linewidth=0),
    ]
    all_handles = flip_handles + [mpatches.Patch(visible=False)] + gain_handles
    fig.legend(handles=all_handles, loc="lower center", ncol=7, fontsize=13)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"cot_combined.{ext}", dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"[saved] cot_combined.{ext}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="analysis_outputs/cot_combined")
    p.add_argument("--exclude-model-substrings", default=",".join(DEFAULT_EXCL))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    exclude = [s.strip().lower() for s in args.exclude_model_substrings.split(",") if s.strip()]
    pairs = collect_pairs(Path(args.results_dir), exclude)
    if not pairs:
        raise RuntimeError(f"No CoT/NoCoT pairs found in: {args.results_dir}")

    all_data = {}
    for model, cot_path, nocot_path in pairs:
        all_data[model] = compute_all(cot_path, nocot_path)
        for lv in LEVELS:
            d = all_data[model][lv]
            print(
                f"[{model}] L{lv}  gain={d['gain_pp']:+.2f}pp  "
                f"helps={d['cot_helps']*100:.1f}%  hurts={d['cot_hurts']*100:.1f}%  n={int(d['n'])}"
            )

    plot(all_data, Path(args.output_dir), dpi=args.dpi)


if __name__ == "__main__":
    main()

