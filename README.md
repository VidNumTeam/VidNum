# VidNum-1.4K: A Comprehensive Benchmark for Video-based Numerical Reasoning

**Project Page:** https://VidNumTeam.github.io  
**Paper:** *VidNum-1.4K: A Comprehensive Benchmark for Video-based Numerical Reasoning* (MM 2026, author draft)

![VidNum Teaser](paper/teaser.png)

VidNum-1.4K is a benchmark for evaluating **video-based numerical reasoning** in Vision-Language Models (VLMs).

## Current Project Format (New Naming)

This repository now uses a unified naming scheme for data and video mapping.

- Dataset file: `question_datasets/VidNum1_4K_options_en_category_en.xlsx`
- Question column: `question`
- Options columns: `option_A`, `option_B`, `option_C`, `option_D`
- Video files: stored in `videos/` as `QID_{ID}.mp4`
- Legacy OSS/timestamp mapping is no longer used in evaluation scripts.

## Data-Video Mapping Rule

Each row is mapped to one video file.

- Preferred: use `Video_Path` column when present.
- Fallback: if `Video_Path` is empty, scripts use `QID_{ID}.mp4`.
- Final resolved path: `videos/<Video_Path or QID_{ID}.mp4>`.

## Dataset Schema (Required)

Minimum required columns:

- `ID`
- `question`
- `option_A`
- `option_B`
- `option_C`
- `option_D`

Common optional columns:

- `Answer`
- `Video_Path`
- `Level`
- `Reasoning_Type`
- `Count_Scope`

## Repository Structure

```text
.
├── question_datasets/          # benchmark tables (xlsx/jsonl)
├── videos/                     # full videos, named QID_{id}.mp4
├── datacuts/                   # optional clipped videos
├── run_VLM_evaluation/         # all run_*.py evaluation scripts
├── templates/                  # prompt templates
├── results/                    # model outputs
├── analysis_outputs/           # plots and analysis artifacts
├── videocut_multithread.py     # video utility pipeline
├── utils.py                    # shared data/schema helpers
├── ana_level_by_or_er.py
├── analyze_shotcut_accuracy_by_model.py
├── plot_vidnum14k_overview_panels.py
├── plot_intern_scale_level_curves.py
└── plot_cot_combined.py
```

## Quick Start

### 1) Prepare data and videos

1. Place `VidNum1_4K_options_en_category_en.xlsx` under `question_datasets/`.
2. Put videos under `videos/` with names like `QID_1.mp4`, `QID_2.mp4`, ...

### 2) Run evaluation

Run scripts from project root.

```bash
# InternVL examples
python run_VLM_evaluation/run_internVL_noCoT.py
python run_VLM_evaluation/run_internVL3_8B.py

# Qwen examples
python run_VLM_evaluation/run_qwen2_5_vl.py
python run_VLM_evaluation/run_qwen3_vl_noCoT.py

# LLaVA-Next examples
python run_VLM_evaluation/run_llava_next.py
```

Gemini script supports CLI args:

```bash
python run_VLM_evaluation/run_gemini31_vl.py \
  --data question_datasets/VidNum1_4K_options_en_category_en.xlsx \
  --template templates/qa_prompt_EN_noCoT.md
```

### 3) Run analysis

```bash
python analyze_shotcut_accuracy_by_model.py
python ana_level_by_or_er.py
python plot_vidnum14k_overview_panels.py
python plot_intern_scale_level_curves.py
python plot_cot_combined.py
```

## Reproducibility Notes

- Keep prompt templates fixed when comparing models.
- Keep frame sampling policy fixed (1 FPS / 48-frame cap depending on script).
- For CoT vs NoCoT, change only prompting style.
- Use the same filtered evaluation subset for fair comparison.
