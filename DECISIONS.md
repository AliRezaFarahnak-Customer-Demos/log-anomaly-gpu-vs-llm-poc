# Decisions

Short log of choices made while building this repo, and why.

## Tooling
- Python 3.12 everywhere (image, CI, local). numpy 2.5.3 needs Python 3.12 or newer, so the 3.11 base image from the brief was not possible with these pins.
- `uv` for local virtualenvs. The system Python has no `ensurepip`.
- Versions are pinned with `==` in `pyproject.toml`. They are the versions that resolved when the repo was built.
- `ruff` handles lint and formatting. Rules: E, F, I, B, UP, SIM.

## Data
- Each split has its own random generator seeded with `"<seed>:<split>"`. A split does not change when another split is resized.
- Trace ids are drawn from a shared set, so all splits are disjoint.
- Timestamps for each split start on a different day. Concurrent traces overlap because the mean gap between trace starts (0.35 s) is shorter than a trace.
- `first_deviation_index` compares the anomalous trace with the normal version of the same trace, including its own benign extra lines.
- A visible error keeps step 5 (order confirmed). Only step 4 is replaced.
- Prepared data (templates, sequences) is derived and not committed. Every command that needs it runs `prepare` if it is missing.

## Repo hygiene
- `.vscode/`, `.github/copilot-instructions.md` (the build brief) and `notes/` (personal prep notes) are git-ignored.
