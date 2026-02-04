# Repository Guidelines

## Project Structure & Module Organization
- `configs/` holds model and training JSON configs.
- `models/` contains model, tokenizer, and inference code.
- `train/` implements datasets and training entrypoints.
- `tools/` provides dataset processing, pickling, and evaluation helpers.
- `scripts/` contains runnable training/eval shell scripts.
- `docs/` documents benchmark setup (CALVIN, LIBERO, SimplerEnv, ALOHA).
- `pretrain/` stores action tokenizer assets and related utilities.
- `reference/` vendors external code used for baselines/eval.
- `logs/` and `data_storage/` are used for outputs/cached artifacts.

## Build, Test, and Development Commands
- Environment setup:
  - `conda create -n emu_vla python=3.10`
  - `pip install -r requirements.txt`
- World model pretraining: `bash scripts/pretrain/train_video_1node.sh`.
- Benchmark training examples:
  - `bash scripts/simulator/calvin/train_calvin_abcd_video.sh`
  - `bash scripts/simulator/libero/train_libero_video.sh`
  - `bash scripts/simulator/simplerenv/train_simplerenv_bridge_video.sh`
- Evaluation scripts are under `scripts/` (see `docs/*.md` for env setup).
- Inference entrypoints: `python models/inference/inference_vision.py` or
  `python models/inference/inference_action.py` (paths may need edits).

## Coding Style & Naming Conventions
- Python-first repo: use 4-space indentation, snake_case for functions/vars,
  CamelCase for classes, UPPER_SNAKE for constants.
- Keep config names descriptive (e.g., `configs/moe_fast_video_pretrain.json`).
- No formatter is enforced; match surrounding style and keep diffs minimal.

## Testing Guidelines
- There is no unit-test suite; validation is done via benchmark evaluation.
- Run dataset processing before training using `tools/process/` and
  `tools/pickle_gen/` scripts.
- Quick sanity check: `python tools/process/cot_sanity_check.py sanity`.

## Commit & Pull Request Guidelines
- Commit history uses short, lowercase summaries (e.g., “fix bug”, “update”).
  Keep messages concise and action-oriented.
- PRs should state the benchmark/task, key configs or scripts touched, and the
  exact commands run. Include dataset/model path changes and checkpoints used.

## Configuration & Data Notes
- Many scripts reference absolute paths like `/share/project/...`; update these
  for your environment or parameterize them before running.
- Outputs typically land under `logs/` and script-defined `output_dir` paths.
