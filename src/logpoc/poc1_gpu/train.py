"""PoC 1 training: LoRA fine-tune of a causal LM on normal traces (next line prediction)."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from logpoc.common import runs
from logpoc.data.prepare import data_manifest_hash, load_sequences


@dataclass
class FitContext:
    config: dict
    rdir: Path
    data_dir: Path
    resume_from: Path | None
    model_dir: Path


def set_hf_env(root: Path) -> None:
    """Jobs run offline with the cache on the share. The image sets the same defaults."""
    os.environ.setdefault("HF_HOME", str(root / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def hardware_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda": None, "gpu": None}
    if torch.cuda.is_available():
        return {"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    return {"cuda": torch.version.cuda, "gpu": None}


def base_model_info(model_dir: Path, repo_id: str) -> dict:
    manifest = model_dir / "model_manifest.json"
    commit = runs.read_json(manifest).get("commit") if manifest.exists() else None
    return {"repo_id": repo_id, "commit": commit, "local_path": str(model_dir)}


def _same_config(a: Path, b: Path) -> bool:
    return yaml.safe_load(a.read_text()) == yaml.safe_load(b.read_text())


def run(
    config_path: Path,
    data_dir: Path,
    fit: Callable[[FitContext], dict] | None = None,
) -> int:
    root = runs.ml_root()
    set_hf_env(root)
    run_id = runs.resolve_run_id()
    rdir = runs.run_dir(run_id, root)

    if runs.is_done(rdir):
        print(f"run {run_id} is already DONE, nothing to do: {rdir}")
        return 0

    config = yaml.safe_load(Path(config_path).read_text())
    rdir.mkdir(parents=True, exist_ok=True)
    saved_cfg = rdir / "config.yaml"
    if saved_cfg.exists():
        if not _same_config(saved_cfg, Path(config_path)):
            raise SystemExit(
                f"run {run_id} already has a different config.yaml. Use a new RUN_ID to change it."
            )
    else:
        shutil.copyfile(config_path, saved_cfg)

    model_dir = root / config["base_model"]["local_dir"]
    ckpt_dir = rdir / "checkpoints"
    resume_from = runs.find_resume_checkpoint(ckpt_dir)
    sha, dirty = runs.git_info()

    meta = runs.read_meta(rdir)
    previous_attempt = bool(meta.get("start_time"))
    if not previous_attempt:
        runs.update_meta(
            rdir,
            run_id=run_id,
            method=config.get("method", "poc1-gpu"),
            variant=config.get("variant"),
            git_sha=sha,
            git_dirty=dirty,
            image_tag=os.environ.get("IMAGE_TAG"),
            data_manifest_sha256=data_manifest_hash(data_dir),
            base_model=base_model_info(model_dir, config["base_model"]["repo_id"]),
            versions=runs.library_versions(),
            job_execution=os.environ.get("CONTAINER_APP_JOB_EXECUTION_NAME"),
            start_time=runs.utc_now(),
            resumes=[],
            attempt_seconds=[],
            **hardware_info(),
        )
    else:
        runs.record_resume(
            rdir,
            checkpoint=resume_from.name if resume_from else None,
            step=runs.checkpoint_step(resume_from) if resume_from else 0,
            reason="restart after interruption",
        )
        runs.update_meta(rdir, **hardware_info())
        where = resume_from.name if resume_from else "scratch"
        print(f"resuming run {run_id} from {where}")

    ctx = FitContext(config, rdir, Path(data_dir), resume_from, model_dir)
    t0 = time.time()
    try:
        result = (fit or _fit)(ctx)
    finally:
        meta = runs.read_meta(rdir)
        meta.setdefault("attempt_seconds", []).append(round(time.time() - t0, 1))
        runs.write_json(rdir / "meta.json", meta)

    meta = runs.read_meta(rdir)
    result = dict(result)
    result["seconds"] = round(sum(meta["attempt_seconds"]), 1)
    result["attempts"] = len(meta["attempt_seconds"])
    runs.write_json(rdir / "train_metrics.json", result)
    runs.update_meta(rdir, end_time=runs.utc_now())
    runs.mark_done(rdir)
    print(f"run {run_id} finished: {result}")
    return 0


def _fit(ctx: FitContext) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    cfg = ctx.config
    tcfg = cfg["train"]
    if not (ctx.model_dir / "config.json").exists():
        raise SystemExit(
            f"base model not found in {ctx.model_dir}. Run the download-model step first."
        )

    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(ctx.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        ctx.model_dir, dtype=torch.bfloat16 if use_cuda else torch.float32
    )
    if tcfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lcfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    seqs = load_sequences(ctx.data_dir, "train")
    enc = tokenizer(
        [s["text"] for s in seqs],
        truncation=True,
        max_length=tcfg["max_length"],
        add_special_tokens=False,
    )
    dataset = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in enc["input_ids"]]

    sleep_s = float(tcfg.get("demo_sleep_per_step_s") or 0)

    class StepHooks(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if sleep_s > 0:
                time.sleep(sleep_s)
            runs.maybe_inject_failure(ctx.rdir, state.global_step)

    args = TrainingArguments(
        output_dir=str(ctx.rdir / "checkpoints"),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        warmup_steps=tcfg["warmup_ratio"],
        num_train_epochs=tcfg["epochs"],
        max_steps=tcfg["max_steps"] or -1,
        seed=tcfg["seed"],
        logging_steps=tcfg["logging_steps"],
        save_strategy="steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        bf16=use_cuda,
        use_cpu=not use_cuda,
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[StepHooks()],
    )
    trainer.train(resume_from_checkpoint=str(ctx.resume_from) if ctx.resume_from else None)

    adapter_dir = ctx.rdir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    return {
        "final_loss": losses[-1] if losses else None,
        "steps": trainer.state.global_step,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
    }
