import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch


MAJOR_BASE_COLORS = {
    "Knowledge": "#79B4A9",
    "Life Record": "#E8B26A",
    "Sports Competition": "#8D8BD7",
    "Artistic Performance": "#E28D86",
    "Film & Television": "#7DA6D8",
    "Unknown": "#BDBDBD",
}

COUNTING_BASE = {
    # Journal-style, colorblind-safe trio (Wong palette):
    # Action: vermilion-orange, Event: steel blue, Object: forest teal.
    "Action": "#D55E00",
    "Event": "#2171B5",
    "Object": "#1A7A4A",
}

LEVEL_ORDER = ["1", "2", "3"]
COUNTING_ORDER = ["Action", "Event", "Object"]


MAJOR_MAP_ZH2EN = {
    "教育与知识": "Knowledge",
    "生活纪录片": "Life Record",
    "运动竞赛": "Sports Competition",
    "艺术表演": "Artistic Performance",
    "电视与电影": "Film & Television",
    "电视和电影": "Film & Television",
    "UNKNOWN": "Unknown",
    "Unknown": "Unknown",
}


MINOR_ALIASES = {
    "人文与历史": "人文和历史",
    "电影和TVshow": "电影和TV show",
    "电影与TV show": "电影和TV show",
    "电影和 Tv show": "电影和TV show",
    "电影和 TV show": "电影和TV show",
    "旅游": "旅行",
    "美食": "食物",
    # ── category merges ──────────────────────────────────────────────────
    "电影": "电影和TV show",   # Film & TV: Movie → Movie & TV Show
    "魔术秀": "舞台表演",       # Artistic: Magic Show → Stage Performance
    "棒球": "其他运动",         # Sports: Baseball → Other Sports
    "动物": "宠物和动物",       # Life Record: Animal → Pet & Animal
}


MINOR_MAP_ZH2EN = {
    "食物": "Food",
    "纪录片": "Documentary",
    "人文和历史": "Humanity & History",
    "生活提示": "Life Tip",
    "文学和艺术": "Literature & Art",
    "足球": "Football",
    "日常生活": "Daily Life",
    "航空": "Aviation",
    "其他运动": "Other Sports",
    "地理": "Geography",
    "生物和医学": "Biology & Medicine",
    "科技": "Technology",
    "杂技": "Acrobatics",
    "宠物和动物": "Pet & Animal",
    "时尚": "Fashion",
    "锻炼": "Exercise",
    "旅行": "Travel",
    "电影和TV show": "Movie & TV Show",
    "篮球": "Basketball",
    "手工": "Handicraft",
    "商业和金融": "Finance & Commerce",
    "新闻播报": "News Report",
    "综艺秀": "Variety Show",
    "杂技秀": "Acrobatics Show",
    "电子竞技": "Esports",
    "法律": "Law",
    "舞台表演": "Stage Performance",
    "棒球": "Baseball",
    "电影": "Movie",
    "魔术秀": "Magic Show",
    "动物": "Animal",
    "UNKNOWN": "Unknown",
    "Unknown": "Unknown",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Draw three separate paper-style overview panels for VidNum1.4K.")
    parser.add_argument("--input-xlsx", default="question_datasets/VidNum1_4K_options_en_category_en.xlsx", help="Input xlsx path")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/metadata_stats/paper_style_panels",
        help="Output directory for panel figures",
    )
    parser.add_argument("--dpi", type=int, default=360, help="Output DPI")
    return parser.parse_args()


def try_set_cjk_font():
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Source Han Sans SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


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


def norm_text(v: object) -> str:
    s = str(v or "").strip()
    return s if s else "UNKNOWN"


def map_major_to_en(major_zh: str) -> str:
    return MAJOR_MAP_ZH2EN.get(major_zh, major_zh)


def norm_minor_zh(minor_zh: str) -> str:
    return MINOR_ALIASES.get(minor_zh, minor_zh)


def map_minor_to_en(minor_zh: str) -> str:
    return MINOR_MAP_ZH2EN.get(minor_zh, minor_zh)


def adjust_color(hex_color: str, factor: float) -> str:
    rgb = np.array(mcolors.to_rgb(hex_color))
    if factor >= 1:
        out = 1 - (1 - rgb) / factor
    else:
        out = rgb * factor
    return mcolors.to_hex(np.clip(out, 0, 1))


def find_column(df: pd.DataFrame, preferred: str, fallback_index: int | None = None) -> str:
    if preferred in df.columns:
        return preferred
    if fallback_index is not None and 0 <= fallback_index < len(df.columns):
        return str(df.columns[fallback_index])
    raise RuntimeError(f"Cannot find column: {preferred}")


def find_first_existing_column(df: pd.DataFrame, candidates: List[str], fallback_index: int | None = None) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if fallback_index is not None and 0 <= fallback_index < len(df.columns):
        return str(df.columns[fallback_index])
    raise RuntimeError(f"Cannot find any candidate columns: {candidates}")


def build_topic_data(df: pd.DataFrame):
    # Support both translated schema (Category/Sub-Category) and legacy schema (视频大类/视频小类).
    major_col = find_first_existing_column(df, ["Category", "视频大类"], fallback_index=2)
    minor_col = find_first_existing_column(df, ["Sub-Category", "SubCategory", "视频小类"], fallback_index=3)

    _orig_major_en = df[major_col].map(norm_text).map(map_major_to_en)
    tmp = pd.DataFrame(
        {
            "major_en": _orig_major_en,
            "minor_en": df[minor_col].map(norm_text).map(norm_minor_zh).map(map_minor_to_en),
        }
    )

    # Merge NaN / Unknown minors in Sports Competition into Other Sports.
    # norm_text() returns the string "nan" for float NaN (since nan is truthy),
    # so we catch both "nan", "Unknown", and any unmapped residual strings.
    _sports_mask = _orig_major_en == "Sports Competition"
    _unknown_mask = tmp["minor_en"].str.lower().isin(["nan", "unknown", "unknow", ""])
    tmp.loc[_sports_mask & _unknown_mask, "minor_en"] = "Other Sports"

    # Keep hierarchy by assigning each minor to its major via majority vote.
    pair_counts = tmp.groupby(["major_en", "minor_en"]).size().reset_index(name="n")
    minor_owner = (
        pair_counts.sort_values(["minor_en", "n"], ascending=[True, False])
        .drop_duplicates("minor_en")
        .set_index("minor_en")["major_en"]
        .to_dict()
    )
    tmp["major_en"] = tmp["minor_en"].map(minor_owner)

    major_counts = tmp["major_en"].value_counts()
    major_sorted = major_counts.index.tolist()

    inner_labels: List[str] = []
    inner_sizes: List[int] = []
    inner_colors: List[str] = []
    outer_labels: List[str] = []
    outer_majors: List[str] = []
    outer_sizes: List[int] = []
    outer_colors: List[str] = []

    for major in major_sorted:
        base = MAJOR_BASE_COLORS.get(major, "#BDBDBD")
        major_df = tmp[tmp["major_en"] == major]
        minor_counts = major_df["minor_en"].value_counts()
        inner_labels.append(major)
        inner_sizes.append(int(minor_counts.sum()))
        inner_colors.append(base)

        n_minor = len(minor_counts)
        for j, (minor, cnt) in enumerate(minor_counts.items()):
            shade = 1.35 - 0.42 * (j / max(1, n_minor - 1))
            outer_labels.append(minor)
            outer_majors.append(major)
            outer_sizes.append(int(cnt))
            outer_colors.append(adjust_color(base, shade))

    unknown_minor_count = int((tmp["minor_en"] == "Unknown").sum())
    return {
        "inner_labels": inner_labels,
        "inner_sizes": inner_sizes,
        "inner_colors": inner_colors,
        "outer_labels": outer_labels,
        "outer_majors": outer_majors,
        "outer_sizes": outer_sizes,
        "outer_colors": outer_colors,
        "unknown_minor_count": unknown_minor_count,
        "major_col": major_col,
        "minor_col": minor_col,
    }


def wrap_label(label: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(label, width=width))


def pretty_wrap_topic_label(label: str) -> str:
    manual = {
        "Humanity & History": "Humanity\n& History",
        "Literature & Art": "Literature\n& Art",
        "Biology & Medicine": "Biology\n& Medicine",
        "Finance & Commerce": "Finance\n& Commerce",
        "Movie & TV Show": "Movie &\nTV Show",
        "Sports Competition": "Sports\nCompetition",
        "Artistic Performance": "Artistic\nPerformance",
        "Film & Television": "Film &\nTelevision",
        "Life Record": "Life\nRecord",
    }
    if label in manual:
        return manual[label]
    if len(label) > 15 and " " in label:
        return wrap_label(label, width=12)
    return label


def radial_upright_rotation(theta_deg: float) -> float:
    rot = theta_deg
    if 90.0 < rot < 270.0:
        rot -= 180.0
    return rot


def place_text_fit_annulus(
    ax,
    label: str,
    theta_deg: float,
    r_center: float,
    r_inner: float,
    r_outer: float,
    fontsize: float,
    color: str,
    weight: str = "normal",
    rotation_extra_deg: float = 0.0,
):
    rot = radial_upright_rotation(theta_deg) + rotation_extra_deg
    rad = math.radians(theta_deg)
    x = r_center * math.cos(rad)
    y = r_center * math.sin(rad)
    txt = ax.text(
        x,
        y,
        label,
        rotation=rot,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        weight=weight,
        linespacing=0.88,
    )

    # Shrink font until text is fully inside ring thickness.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    center_disp = ax.transData.transform((0.0, 0.0))
    rin_disp = np.linalg.norm(ax.transData.transform((r_inner, 0.0)) - center_disp) + 2.0
    rout_disp = np.linalg.norm(ax.transData.transform((r_outer, 0.0)) - center_disp) - 2.0

    for _ in range(16):
        bbox = txt.get_window_extent(renderer=renderer)
        corners = np.array(
            [
                [bbox.x0, bbox.y0],
                [bbox.x0, bbox.y1],
                [bbox.x1, bbox.y0],
                [bbox.x1, bbox.y1],
            ]
        )
        d = np.sqrt(((corners - center_disp) ** 2).sum(axis=1))
        if d.min() >= rin_disp and d.max() <= rout_disp:
            break
        fs = txt.get_fontsize()
        if fs <= 5.3:
            break
        txt.set_fontsize(fs - 0.35)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    return txt


def plot_topic_donut(topic_data, out_png: Path, out_pdf: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(9.6, 8.8), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_title("Video Topic Hierarchy", fontsize=17, weight="bold", pad=12)

    # Thicker outer ring + smaller center hole for more label space.
    outer_radius = 1.18
    outer_width = 0.42
    inner_radius = 0.74
    inner_width = 0.42
    hole_radius = 0.29

    wedges_outer, _ = ax.pie(
        topic_data["outer_sizes"],
        labels=None,
        radius=outer_radius,
        startangle=95,
        counterclock=False,
        colors=topic_data["outer_colors"],
        wedgeprops={
            "width": outer_width,
            "edgecolor": "#5B687A",
            "linewidth": 1.25,
            "joinstyle": "round",
        },
    )

    wedges_inner, _ = ax.pie(
        topic_data["inner_sizes"],
        labels=None,
        radius=inner_radius,
        startangle=95,
        counterclock=False,
        colors=topic_data["inner_colors"],
        wedgeprops={
            "width": inner_width,
            "edgecolor": "#4B5E72",
            "linewidth": 1.35,
            "joinstyle": "round",
        },
    )

    # Outer ring labels: radial orientation, upright on both halves, forced inside ring.
    r_text_outer = outer_radius - outer_width * 0.52
    outer_inner_bound = outer_radius - outer_width
    for wedge, label, major in zip(wedges_outer, topic_data["outer_labels"], topic_data["outer_majors"]):
        theta = (wedge.theta1 + wedge.theta2) * 0.5
        span = abs(wedge.theta2 - wedge.theta1)
        label_wrapped = pretty_wrap_topic_label(label)
        fs = 10.5 if span >= 12 else 9.0 if span >= 8 else 7.5 if span >= 5.5 else 6.2
        extra_rot = 180.0 if major != "Knowledge" else 0.0
        txt = place_text_fit_annulus(
            ax=ax,
            label=label_wrapped,
            theta_deg=theta,
            r_center=r_text_outer,
            r_inner=outer_inner_bound,
            r_outer=outer_radius,
            fontsize=fs,
            color="#1F2D3D",
            rotation_extra_deg=extra_rot,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.1, foreground="white", alpha=0.85)])

    # Inner ring major labels: radial and upright to avoid center crowding.
    r_text_inner = inner_radius - inner_width * 0.52
    inner_inner_bound = inner_radius - inner_width
    for wedge, label in zip(wedges_inner, topic_data["inner_labels"]):
        theta = (wedge.theta1 + wedge.theta2) * 0.5
        span = abs(wedge.theta2 - wedge.theta1)
        fs = 14.0 if span >= 55 else 12.5 if span >= 35 else 11.0 if span >= 22 else 9.0
        extra_rot = 180.0 if label != "Knowledge" else 0.0
        txt = place_text_fit_annulus(
            ax=ax,
            label=pretty_wrap_topic_label(label),
            theta_deg=theta,
            r_center=r_text_inner,
            r_inner=inner_inner_bound,
            r_outer=inner_radius,
            fontsize=fs,
            color="#1D2E40",
            weight="bold",
            rotation_extra_deg=extra_rot,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.78)])

    ax.add_artist(plt.Circle((0, 0), hole_radius, color=(1, 1, 1, 0), zorder=10, ec="#6F8196", lw=1.2))
    center_txt = ax.text(
        0,
        0,
        "VidNum\n1.4K",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color="#3A4A56",
    )
    center_txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)])
    ax.set_aspect("equal")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_duration_hist(durations: np.ndarray, out_png: Path, out_pdf: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(12.6, 3.9), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_title("Video Duration Distribution", fontsize=18, weight="bold", pad=10)
    if len(durations) == 0:
        ax.text(0.5, 0.5, "No Duration Data", transform=ax.transAxes, ha="center", va="center")
    else:
        max_sec = max(120, int(np.ceil(durations.max() / 10.0) * 10))
        bins = np.arange(0, max_sec + 10, 10)
        counts, edges = np.histogram(durations, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges) * 0.82

        colors = ["#4682B4"] * len(counts)
        bars = ax.bar(centers, counts, width=widths, color=colors, edgecolor="black", linewidth=1.1)
        ax.set_xlabel("Duration (seconds)", fontsize=14)
        ax.set_ylabel("Video Count", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.tick_params(axis="both", labelsize=14)

        ymax = max(1, counts.max())
        for b, v in zip(bars, counts):
            if v > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + ymax * 0.015,
                    f"{int(v)}",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    # set y lim
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05) # add 25% headroom for labels
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def level_shades(base: str):
    return {
        "1": adjust_color(base, 1.50),
        "2": adjust_color(base, 1.18),
        "3": adjust_color(base, 0.90),
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from matplotlib.patches import Patch
from pathlib import Path

def adjust_lightness(color, amount=1.2):
    """
    调整颜色亮度。
    amount > 1.0 为调亮，amount < 1.0 为调暗。
    """
    try:
        c = mc.to_rgb(color)
        # 将 RGB 转换为 HLS
        h, l, s = colorsys.rgb_to_hls(*c)
        # 提高亮度 (l)，并适当降低饱和度 (s) 让颜色更清爽
        l = max(0, min(1, l * amount))
        s = s * 0.9 
        return colorsys.hls_to_rgb(h, l, s)
    except Exception:
        return color

def plot_level_counting_bar(df: pd.DataFrame, out_png: Path, out_pdf: Path, dpi: int):
    level_col = find_first_existing_column(df, ["LLM_Level_Final", "Level", "Hier_Level", "Hierarchy_Level"])
    ct_col = find_first_existing_column(df, ["meta_counting_type"])

    def _norm_level(v: object) -> str:
        s = norm_text(v)
        m = re.search(r"([123])", s)
        return m.group(1) if m else s

    work = pd.DataFrame(
        {
            "level": df[level_col].map(_norm_level),
            "counting_type": df[ct_col].map(norm_text).str.lower(),
        }
    )
    work["counting_type"] = work["counting_type"].map(
        {
            "action": "Action",
            "event": "Event",
            "object": "Object",
        }
    )
    # 确保 COUNTING_ORDER 和 LEVEL_ORDER 已定义，例如 ["Action", "Event", "Object"]
    work = work[work["counting_type"].isin(COUNTING_ORDER)]
    work = work[work["level"].isin(LEVEL_ORDER)]

    pivot = (
        work.groupby(["counting_type", "level"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=COUNTING_ORDER, columns=LEVEL_ORDER, fill_value=0)
    )

    # 画布初始化
    fig, ax = plt.subplots(figsize=(12.6, 4.2), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_title("Level Distribution in Action / Event / Object", fontsize=18, weight="bold", pad=20)

    x = np.arange(len(COUNTING_ORDER))
    width = 0.22
    offsets = [-width, 0, width]

    ymax = max(1, int(pivot.values.max()))
    
    # 绘制各层级的柱子
    for i, lv in enumerate(LEVEL_ORDER):
        vals = pivot[lv].values.astype(int)
        
        # 1. 获取基底颜色并调亮 (amount=1.4 表示增加 40% 亮度)
        raw_colors = [level_shades(COUNTING_BASE[c])[lv] for c in COUNTING_ORDER]
        bright_colors = [adjust_lightness(c, amount=1.35) for c in raw_colors]
        
        bars = ax.bar(
            x + offsets[i], 
            vals, 
            width=width, 
            color=bright_colors, 
            edgecolor="#333333", 
            linewidth=0.8
        )
        
        # 标注数值
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + ymax * 0.015,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    weight="bold",
                    color="#444444"
                )

    # 轴标签设置
    ax.set_xticks(x)
    ax.set_xticklabels(COUNTING_ORDER, fontsize=14, weight="bold")
    ax.set_ylabel("Question Count", fontsize=14)
    ax.set_ylim(0, ymax * 1.25) # 顶部留白，防止数字顶到头
    
    # 细节优化：使用更淡的点状网格
    ax.grid(axis="y", linestyle=":", color="gray", alpha=0.45)
    ax.tick_params(axis="y", labelsize=13)
    
    # Legend: each Level entry shows three side-by-side patches for Action / Event / Object
    legend_handles = []
    for lv in LEVEL_ORDER:
        raw = [level_shades(COUNTING_BASE[c])[lv] for c in COUNTING_ORDER]
        bright = [adjust_lightness(c, amount=1.35) for c in raw]
        legend_handles.append(tuple(
            Patch(facecolor=bright[j], edgecolor="#333333", linewidth=0.8)
            for j in range(len(COUNTING_ORDER))
        ))

    ax.legend(
        handles=legend_handles,
        labels=["Level 1", "Level 2", "Level 3"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        frameon=False,
        fontsize=12,
        loc="upper right",
    )

    # 保存并关闭
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

def main():
    args = parse_args()
    try_set_cjk_font()

    df = pd.read_excel(args.input_xlsx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_col = find_first_existing_column(df, ["Timestamp", "时间戳"], fallback_index=4)
    parsed_duration = df[duration_col].map(parse_duration_seconds)
    duration_missing_mask = parsed_duration.isna()
    durations = parsed_duration.dropna().astype(float).to_numpy()
    duration_missing_ids = (
        df.loc[duration_missing_mask, "ID"].tolist() if "ID" in df.columns else []
    )

    topic_data = build_topic_data(df)

    topic_png = output_dir / "topic_hierarchy_donut_en.png"
    topic_pdf = output_dir / "topic_hierarchy_donut_en.pdf"
    dur_png = output_dir / "video_duration_distribution.png"
    dur_pdf = output_dir / "video_duration_distribution.pdf"
    bar_png = output_dir / "level_countingtype_distribution_bar.png"
    bar_pdf = output_dir / "level_countingtype_distribution_bar.pdf"

    plot_topic_donut(topic_data, topic_png, topic_pdf, dpi=args.dpi)
    plot_duration_hist(durations, dur_png, dur_pdf, dpi=args.dpi)
    plot_level_counting_bar(df, bar_png, bar_pdf, dpi=args.dpi)

    summary = {
        "input_xlsx": str(args.input_xlsx),
        "row_count": int(len(df)),
        "duration_col": duration_col,
        "duration_parsed_count": int(parsed_duration.notna().sum()),
        "duration_missing_count": int(duration_missing_mask.sum()),
        "duration_missing_ids": [int(x) for x in duration_missing_ids if pd.notna(x)],
        "topic_major_col": topic_data["major_col"],
        "topic_minor_col": topic_data["minor_col"],
        "unknown_minor_count": topic_data["unknown_minor_count"],
        "notes": [
            "Outer-ring labels are all shown in English.",
            "Minor aliases are merged (e.g., 旅游->旅行, 美食->食物, TVshow variants merged).",
        ],
        "outputs": {
            "topic_donut_png": str(topic_png),
            "topic_donut_pdf": str(topic_pdf),
            "duration_png": str(dur_png),
            "duration_pdf": str(dur_pdf),
            "level_counting_bar_png": str(bar_png),
            "level_counting_bar_pdf": str(bar_pdf),
        },
    }
    (output_dir / "overview_panels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[Done] Saved panels:")
    for k, v in summary["outputs"].items():
        print(f" - {k}: {v}")
    if summary["duration_missing_count"] > 0:
        print(
            f" - warning: duration parsed {summary['duration_parsed_count']}/{summary['row_count']}, "
            f"missing IDs: {summary['duration_missing_ids']}"
        )
    print(f" - summary: {output_dir / 'overview_panels_summary.json'}")


if __name__ == "__main__":
    main()
