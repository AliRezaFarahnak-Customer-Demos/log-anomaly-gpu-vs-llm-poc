"""PoC 2: one structured LLM call per trace. No training and no GPU."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml

from logpoc.common.pricing import llm_cost_per_1000, load_pricing
from logpoc.common.report import write_eval
from logpoc.common.runs import (
    create_fresh_run,
    git_info,
    library_versions,
    mark_done,
    resolve_run_id,
    update_meta,
    utc_now,
)
from logpoc.data.prepare import data_manifest_hash, load_labels, load_sequences

TYPES = ["none", "visible_error", "silent_skip", "wrong_order", "truncated", "retry_storm"]


def build_schema(n_lines: int, flow_names: list[str] | None = None) -> dict:
    """Strict response schema for one trace. Enums keep every field inside its valid values.

    The model fills the properties in this order: flow and step check first, so even without
    reasoning tokens it matches the steps before it decides.
    """
    props: dict[str, dict] = {}
    if flow_names:
        props["flow"] = {
            "type": "string",
            "enum": [*flow_names, "unknown"],
            "description": "The expected flow this trace runs, recognised from its first line. "
            "unknown only when no expected flow starts like this trace.",
        }
    props["step_check"] = {
        "type": "string",
        "description": "Match every expected step of the flow, in order, to the number of the "
        "line where it occurs, as 'step: line' pairs separated by commas. Write 'missing' for "
        "a step with no line, 'ERROR n' for a step that failed on line n, and every line number "
        "of a step that occurs more than once. A line listed as optional_harmless can cover "
        "expected steps that do the same work, for example a cached result covers a request "
        "and its response: give those steps that line number. Match steps, never count lines.",
    }
    props["explanation"] = {
        "type": "string",
        "description": "One sentence for an operator, based on step_check: name the flow and "
        "the first step that failed, is missing, is out of order or repeats, with its line "
        "number. For a normal trace, name the flow and say it completed.",
    }
    props["verdict"] = {
        "type": "string",
        "enum": ["normal", "anomalous"],
        "description": "normal only when step_check finds every expected step exactly once, "
        "in the expected order, up to the flow's last step, and no line is an ERROR. Lines "
        "listed as optional_harmless are normal wherever they appear: WARN lines, slow "
        "timings, a single retry and a cached result do not make a trace anomalous. "
        "anomalous: anything else, also when the trace ends with the flow's last step.",
    }
    props["anomaly_type"] = {
        "type": "string",
        "enum": TYPES,
        "description": "The kind of the first deviation in step_check. none: the verdict is "
        "normal. visible_error: a line has level ERROR; choose this whenever an ERROR line is "
        "present, even if later steps look fine. silent_skip: an expected step is missing "
        "while a later step still occurs, and no line is an ERROR. wrong_order: every "
        "expected step is present but at least two are in the wrong order. truncated: the "
        "trace stops before the flow's last step and no line is an ERROR. retry_storm: one "
        "expected step is logged three or more times in a row instead of once.",
    }
    props["first_deviation_line"] = {
        "type": ["integer", "null"],
        "enum": [*range(n_lines + 1), None],
        "description": "0-based number of the first line that breaks the expected flow, null "
        "when the verdict is normal. visible_error: the ERROR line. silent_skip: the line that "
        "appears where the missing step should be. wrong_order: the first line that appears "
        "too early. retry_storm: the first repeated copy of the step (its second occurrence). "
        f"truncated: {n_lines}, the number of lines, where the next step should have appeared.",
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


SYSTEM = """You check one trace from production application logs: all log lines with the same \
correlation id, in time order. Lines are log templates where <*> is a masked value, each \
prefixed with its 0-based line number. Fill in the response fields in order, exactly as their \
descriptions define:
1. flow: the expected flow, recognised from the first line.
2. step_check: every expected step of that flow matched to a line, in order.
3. explanation, verdict, anomaly_type, first_deviation_line: decided from step_check and from \
any ERROR line.
Never decide from the number of lines or from the last line alone. Optional harmless lines add \
lines, a cached result can replace two steps, and a trace that ends with the flow's last step \
can still miss a step in the middle."""


def build_messages(lines: list[str], flows: str | None) -> list[dict]:
    system = SYSTEM
    if flows:
        system += "\n\nExpected flows. Lines listed as optional_harmless are normal:\n" + flows
    user = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_client():
    """Foundry with Entra ID only. Locally, set AZURE_TENANT_ID if the CLI default differs."""
    from azure.identity import AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
    from openai import OpenAI

    tenant = os.environ.get("AZURE_TENANT_ID")
    cred = AzureCliCredential(tenant_id=tenant) if tenant else DefaultAzureCredential()
    token = get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default")
    token()  # fill the token cache once, before many threads ask for it at the same time
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    # retries honour the 429 retry-after header of the Foundry rate limit
    return OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=token, max_retries=10)


class FakeClient:
    """Offline stand-in for dry runs and tests: flags a trace only when a line is an ERROR."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        lines = messages[-1]["content"].splitlines()
        bad = next((i for i, line in enumerate(lines) if " ERROR " in line), None)
        answer = {
            "verdict": "normal" if bad is None else "anomalous",
            "anomaly_type": "none" if bad is None else "visible_error",
            "first_deviation_line": bad,
            "explanation": "offline fake client",
        }
        message = SimpleNamespace(content=json.dumps(answer))
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def classify_one(
    client,
    model: str,
    lines: list[str],
    flows: str | None,
    effort: str | None = None,
    flow_names: list[str] | None = None,
) -> tuple[dict, int, int]:
    # effort None keeps the model's default reasoning level
    extra = {"reasoning_effort": effort} if effort else {}
    schema = build_schema(len(lines), flow_names)
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(lines, flows),
        # the answer is ~200 tokens; the cap also stops the rate limiter reserving the model maximum
        max_completion_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": schema},
        },
        **extra,
    )
    answer = json.loads(resp.choices[0].message.content)
    return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens


def run(
    data_dir: Path,
    split: str = "dev",
    context: str = "none",
    limit: int | None = None,
    only_flagged_by: str | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    concurrency: int = 4,
    client=None,
) -> int:
    if context == "few-shot" or only_flagged_by:
        raise SystemExit("few-shot context and --only-flagged-by are not implemented")
    t0 = time.time()
    data_dir = Path(data_dir)
    model = "offline-fake" if dry_run else os.environ.get("LLM_DEPLOYMENT", "gpt-5.6-luna")
    effort = os.environ.get("LLM_REASONING_EFFORT") or None
    flows = (data_dir / "flows.yaml").read_text() if context == "expected-flow" else None
    flow_names = list(yaml.safe_load(flows)) if flows else None
    seqs = load_sequences(data_dir, split)[:limit]
    labels = load_labels(data_dir, split)
    client = client or (FakeClient() if dry_run else make_client())

    rid = resolve_run_id(run_id, prefix="poc2-")
    rdir = create_fresh_run(rid)
    sha, dirty = git_info()
    update_meta(
        rdir,
        run_id=rid,
        method="poc2-llm",
        model=model,
        reasoning_effort=effort or "model default",
        context=context,
        git_sha=sha,
        git_dirty=dirty,
        split=split,
        data_manifest_sha256=data_manifest_hash(data_dir),
        versions=library_versions(),
        start_time=utc_now(),
    )

    def classify(s: dict) -> tuple[dict, int, int]:
        return classify_one(client, model, s["lines"], flows, effort, flow_names)

    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(classify, seqs))

    records = []
    with (rdir / "answers.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s, (answer, _, _) in zip(seqs, results, strict=True):
            f.write(json.dumps({"trace_id": s["trace_id"], **answer}) + "\n")
            lab = labels[s["trace_id"]]
            idx = answer["first_deviation_line"]
            records.append(
                {
                    "trace_id": s["trace_id"],
                    "label": lab["label"],
                    "anomaly_type": lab["anomaly_type"],
                    "first_deviation_index": lab["first_deviation_index"],
                    "flagged": answer["verdict"] == "anomalous",
                    "score": None,
                    "pred_index": idx,
                    "worst_line_text": answer["explanation"],
                }
            )
    tokens_in = sum(r[1] for r in results)
    tokens_out = sum(r[2] for r in results)
    wall = time.time() - t0
    payload = write_eval(
        rdir,
        split,
        method="poc2-llm",
        variant=f"{model} {context} reasoning={effort or 'default'}",
        records=records,
        summary={
            "wall_seconds": wall,
            "scoring_seconds": wall,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cost_per_1000": llm_cost_per_1000(tokens_in, tokens_out, len(records), load_pricing()),
        },
        title=f"PoC 2 LLM ({model}, context {context}) on {split}",
    )
    update_meta(rdir, end_time=utc_now())
    mark_done(rdir)
    m = payload["metrics"]
    print(
        f"{rid} {split}: precision={m['precision']} recall={m['recall']} f1={m['f1']} "
        f"tokens in={tokens_in} out={tokens_out}"
    )
    return 0
