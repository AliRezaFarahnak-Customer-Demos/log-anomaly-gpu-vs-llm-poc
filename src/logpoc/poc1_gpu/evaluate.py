"""PoC 1 evaluation: per-token NLL from a fine-tuned model, aggregated to line and trace scores."""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

import yaml

from logpoc.common import runs
from logpoc.common.metrics import auroc, percentile_threshold
from logpoc.common.pricing import gpu_cost_per_1000, gpu_training_cost, load_pricing
from logpoc.common.report import write_eval
from logpoc.data.prepare import (
    TRACE_END,
    TRACE_START,
    load_labels,
    load_sequences,
)
from logpoc.poc1_gpu.train import set_hf_env


def line_starts(lines: Sequence[str]) -> list[int]:
    """Char offset of each line in the trace text, plus the TRACE END marker as the last entry."""
    pos = len(TRACE_START) + 1
    starts = []
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    starts.append(pos)
    return starts


def aggregate_tokens(
    text: str,
    offsets: Sequence[tuple[int, int]],
    nll: Sequence[float],
    starts: Sequence[int],
) -> dict:
    """Map token NLLs to lines.

    nll[i] is the loss of token i+1 given the tokens before it, so the first token is skipped.
    A token belongs to the line of its first non-space character. A pure whitespace token, such
    as the newline that ends a line, belongs to the line it ends. Header tokens are ignored.
    Returns line index -> mean NLL, plus the mean over all counted tokens.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    total = 0.0
    n_total = 0
    for i in range(1, len(offsets)):
        s, e = offsets[i]
        if e <= s:
            continue
        span = text[s:e]
        stripped = span.lstrip()
        pos = s + (len(span) - len(stripped)) if stripped else s
        line = bisect_right(starts, pos) - 1
        if line < 0:
            continue
        value = float(nll[i - 1])
        sums[line] = sums.get(line, 0.0) + value
        counts[line] = counts.get(line, 0) + 1
        total += value
        n_total += 1
    return {
        "line_scores": {k: sums[k] / counts[k] for k in sums},
        "mean_nll": total / n_total if n_total else 0.0,
    }


def trace_result(lines: Sequence[str], agg: dict) -> dict:
    scores = agg["line_scores"]
    if not scores:
        return {"score": 0.0, "mean_nll": agg["mean_nll"], "worst_index": None, "worst_text": ""}
    worst = max(scores, key=lambda k: scores[k])
    text = TRACE_END if worst >= len(lines) else lines[worst]
    return {
        "score": scores[worst],
        "mean_nll": agg["mean_nll"],
        "worst_index": worst,
        "worst_text": text,
    }


def score_sequences(model, tokenizer, seqs: list[dict], device: str, batch_size: int, max_len: int):
    import torch

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["text"]))
    results: list[dict | None] = [None] * len(seqs)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            texts = [seqs[i]["text"] for i in idx]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")
            batch = {k: v.to(device) for k, v in enc.items()}
            logits = model(**batch).logits[:, :-1].float()
            targets = batch["input_ids"][:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view(targets.shape)
            loss = loss.cpu()
            mask = enc["attention_mask"]
            for row, i in enumerate(idx):
                n = int(mask[row].sum())
                offs = [tuple(o) for o in offsets[row][:n].tolist()]
                agg = aggregate_tokens(
                    seqs[i]["text"],
                    offs,
                    loss[row, : n - 1].tolist(),
                    line_starts(seqs[i]["lines"]),
                )
                results[i] = trace_result(seqs[i]["lines"], agg)
    return results


def _pick_device(choice: str) -> str:
    import torch

    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _load_model(model_dir: Path, adapter_dir: Path, device: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    return model, tokenizer


def run(
    run_id: str,
    data_dir: Path,
    split: str = "test",
    device: str = "auto",
    percentile: float | None = None,
    batch_size: int = 8,
    scorer=None,
) -> dict:
    """scorer(seqs) -> list of trace_result dicts. Tests pass a fake, real runs load the model."""
    root = runs.ml_root()
    set_hf_env(root)
    rdir = runs.run_dir(run_id, root)
    if not runs.is_done(rdir):
        raise SystemExit(f"run {run_id} is not DONE (no adapter to evaluate): {rdir}")
    cfg = yaml.safe_load((rdir / "config.yaml").read_text())
    pct = percentile if percentile is not None else cfg["eval"]["threshold_percentile"]
    max_len = cfg["train"]["max_length"]
    dev_name = "fake"
    if scorer is None:
        dev_name = _pick_device(device)
        model, tokenizer = _load_model(
            root / cfg["base_model"]["local_dir"], rdir / "adapter", dev_name
        )

        def scorer(seqs):
            return score_sequences(model, tokenizer, seqs, dev_name, batch_size, max_len)

    val_cache = rdir / "val_scores.json"
    if val_cache.exists():
        val = runs.read_json(val_cache)
    else:
        t0 = time.time()
        res = scorer(load_sequences(data_dir, "val"))
        val = {"primary": [r["score"] for r in res], "seconds": round(time.time() - t0, 2)}
        runs.write_json(val_cache, val)
    threshold = percentile_threshold(val["primary"], pct)

    seqs = load_sequences(data_dir, split)
    labels = load_labels(data_dir, split)
    t0 = time.time()
    results = scorer(seqs)
    scoring_seconds = time.time() - t0

    records = []
    for s, r in zip(seqs, results, strict=True):
        lab = labels[s["trace_id"]]
        records.append(
            {
                "trace_id": s["trace_id"],
                "label": lab["label"],
                "anomaly_type": lab["anomaly_type"],
                "first_deviation_index": lab["first_deviation_index"],
                "score": r["score"],
                "flagged": r["score"] > threshold,
                "pred_index": r["worst_index"],
                "worst_line_text": r["worst_text"],
                "mean_nll": r["mean_nll"],
            }
        )

    pricing = load_pricing()
    train_metrics = runs.read_json(rdir / "train_metrics.json")
    meta = runs.read_meta(rdir)
    summary = {
        "variant": cfg.get("variant"),
        "scoring_seconds": scoring_seconds,
        "wall_seconds": scoring_seconds,
        "val_scoring_seconds": val["seconds"],
        "device": dev_name,
        "threshold_percentile": pct,
        "training_seconds": train_metrics.get("seconds"),
        "training_cost": gpu_training_cost(train_metrics.get("seconds") or 0, pricing),
        "cost_per_1000": gpu_cost_per_1000(scoring_seconds, len(records), pricing),
        "auroc_mean_nll": auroc([r["label"] for r in records], [r["mean_nll"] for r in records]),
        "image_tag": meta.get("image_tag"),
    }
    payload = write_eval(
        rdir,
        split,
        method="poc1-gpu",
        variant=str(cfg.get("variant")),
        records=records,
        summary=summary,
        threshold=threshold,
        title=f"PoC 1 fine-tuned model on {split}",
    )
    m = payload["metrics"]
    print(
        f"{run_id} {split}: precision={m['precision']} recall={m['recall']} "
        f"f1={m['f1']} fpr={m['false_positive_rate']} loc={m['localisation_hit_rate']}"
    )
    return payload
