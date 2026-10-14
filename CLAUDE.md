# CLAUDE.md — ML Arena session bootstrap

Read this first in every new session. Then read `docs/STATUS.md` (where we are), then the doc for the current phase.

## What this project is
ML Arena: a small reproducible ML research lab for a college project. It benchmarks algorithms on three tasks (Titanic classification, California Housing regression, Mall Customers clustering), then tests **robustness** (noise, missing features, class imbalance), investigates **failures**, and presents everything in a **Streamlit dashboard**. It also covers a list of syllabus experiments as small separate labs.

Central question: *Which algorithms perform best per task, and does the best model on clean data stay the best when data gets harder?*

## Document map (read in this order)
| File | Purpose |
|---|---|
| `docs/STATUS.md` | **Live progress tracker, decision log, next action. Update at the end of every phase.** |
| `docs/PRD.md` | Vision, research questions, hypotheses, scope, requirements |
| `docs/PLAN.md` | Day-by-day roadmap, checkpoints/gates, scope cuts |
| `docs/EXPERIMENT_DESIGN.md` | Exact algorithms, grids, metrics, perturbations, protocol, result schemas |
| `docs/ARCHITECTURE.md` | Folder structure, module responsibilities, interfaces, dashboard pages |
| `docs/DATASETS.md` | Dataset validation, preprocessing, fallbacks |
| `docs/SYLLABUS_MAP.md` | Every syllabus item mapped to an experiment/notebook |
| `docs/ACCEPTANCE_CRITERIA.md` | Objective done-checklist |
| `docs/RISKS.md` | Risks, mitigations, fallbacks |
| `docs/VIVA_PREP.md` | Viva Q&A, grows as we build |

## Working rules (non-negotiable)
1. **Phased work, AUTONOMOUS MODE (user decision 2026-10-15).** Do Days 1-7 of `docs/PLAN.md` in order without waiting for confirmation. After each phase: run its gate, update `docs/STATUS.md`, commit, continue. Do not use multi-agent workflows; build one piece at a time. Never get stuck: wrap every shell command in `timeout`, use fallbacks from `docs/RISKS.md`, log the decision, move on. If the session restarts, resume from `docs/STATUS.md`.
2. **Never invent results.** All numbers in tables, README, report and viva notes must come from executed experiments saved in `results/`. Never claim a winner (e.g. XGBoost) before it is measured. Hypotheses are tested, not assumed. Report negative results honestly.
3. **No data leakage.** Split first; fit imputers/scalers/encoders on train only; tune only on train (inner CV); perturb the test set (imbalance perturbs the train set).
4. **The user must understand the code.** Prefer short, readable modules with comments on the *why*. After each phase give a brief walkthrough ("what this does, why") and add viva-style Q&A to `docs/VIVA_PREP.md`.
5. **Keep it simple.** No databases, microservices, cloud, custom frontend. CSV + Parquet-free (CSV only) results. Streamlit only.
6. **Reproducibility.** Every random op takes an explicit seed. Every experiment row records its seed and settings. One command (`python scripts/run_all.py`) regenerates all results.
7. **Verify, don't assume.** Run the code and tests for each phase; show real output. If something fails or is skipped, say so plainly.
8. **Git.** User authorised one commit per phase (autonomous mode). Never push. Do not commit `.venv/`, raw datasets with restrictive licenses, or large artifacts.
9. If a dataset or library causes real friction, use the fallback in `docs/DATASETS.md` / `docs/RISKS.md` and log the decision in `docs/STATUS.md`.

## Environment
- Repo root: `/home/ayush/ML_PROJECTS/ML_Arena` (WSL2 Linux). Virtualenv at `.venv/` (activate: `source .venv/bin/activate`). No GPU assumed.
- Verify on Day 1 which packages are already installed in `.venv` before installing anything; pin versions in `requirements.txt`.

## Current state
See `docs/STATUS.md`. Planning docs are written; **awaiting user approval of the plan** before any code.
