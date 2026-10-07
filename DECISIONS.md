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

## Production-like rehearsal (`data/synthetic/prod_like`)
- Invented 10-day extract shaped like a real one: 4 business flows, background noise, 2 incidents with an early warning phase, 80 labelled traces.
- `write_splits` builds the pipeline's split files from the logs, the labelled traces and the incident windows only, so the same step works on a real extract.
- test = the labelled traces. train/val = every other trace outside the incident windows, unlabelled, so a few anomalies leak in as they would in real data.
- Background lines (`corrId=-`) are dropped: both PoCs judge one trace at a time.

## Models
- PoC 1 uses Llama 2 7B, the base model of the LogLLaMA approach. `NousResearch/Llama-2-7b-hf` is an ungated copy; the official `meta-llama/Llama-2-7b-hf` needs `HF_TOKEN`. Llama 2 can no longer be downloaded from the Azure ML `azureml-meta` registry.
- PoC 2 uses `gpt-6-luna` with reasoning `none` on Foundry (Data Zone Standard, 333K tokens per minute, the quota maximum). The response schema is built per trace: enums bound the flow, verdict, type and line, and a `step_check` field makes the model match every expected step before it decides. Without it, reasoning `none` judged by line count and made mistakes.
- Azure counts prompt plus `max_completion_tokens` per request against the rate limit, so the call caps output at 1,000 tokens. One batch of 80 traces reserves about 240K tokens: at most one batch per minute.

## Azure
- One resource group in Italy North. Sweden Central refused new Container Apps environments (capacity).
- No storage account: tenant policy switches off shared key and public network access, so Azure Files cannot be mounted. The GPU job downloads the model to the replica disk (504 GB on the A100 profile) and prints its results as one `RESULT` log line.
- Image dependencies come from `docker/requirements.lock`, compiled against the Microsoft package proxy, which lags the `pyproject.toml` pins by one patch (torch 2.14.0, transformers 5.18.0, peft 0.21.1, openai 3.22.1).
- `configs/pricing.yaml` stays empty on purpose; `configs/pricing_azure_list.yaml` holds Azure retail list prices for `compare --pricing`.

## Repo hygiene
- `.vscode/`, `.github/copilot-instructions.md` (the build brief) and `notes/` (personal prep notes) are git-ignored.
