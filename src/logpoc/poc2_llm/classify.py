"""PoC 2: one structured LLM call per trace. No training and no GPU."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

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
# The model fills the properties in this order, so the explanation (its reasoning) comes first.
SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "One sentence for an operator, written before the other fields. Name "
            "the flow and the step that failed, is missing, is out of order or repeats, with its "
            "line number. For a normal trace, name the flow and say it completed.",
        },
        "verdict": {
            "type": "string",
            "enum": ["normal", "anomalous"],
            "description": "normal: one complete run of one expected flow. Every expected step "
            "occurs once, in the expected order, up to the flow's last step, and no line is an "
            "ERROR. Lines listed as optional_harmless are normal wherever they appear, and one "
            "may stand in for expected steps that do the same work, such as a cached result "
            "instead of a request and a response. WARN lines, slow timings and a single retry "
            "are normal when the flow still completes. anomalous: anything else.",
        },
        "anomaly_type": {
            "type": "string",
            "enum": TYPES,
            "description": "The kind of the first deviation. none: the verdict is normal. "
            "visible_error: a line has level ERROR; choose this whenever an ERROR line is "
            "present, even if later steps look fine. silent_skip: an expected step is missing "
            "while later steps still occur, and no line is an ERROR. wrong_order: every "
            "expected step is present but at least two are in the wrong order. truncated: the "
            "trace stops before the flow's last step and no line is an ERROR. retry_storm: one "
            "expected step is logged three or more times in a row instead of once.",
        },
        "first_deviation_line": {
            "type": ["integer", "null"],
            "description": "0-based number of the first line that breaks the expected flow, "
            "null when the verdict is normal. visible_error: the ERROR line. silent_skip: the "
            "line that appears where the missing step should be. wrong_order: the first line "
            "that appears too early. retry_storm: the first repeated copy of the step (its "
            "second occurrence). truncated: the number of lines in the trace, where the next "
            "expected step should have appeared.",
        },
    },
    "required": ["explanation", "verdict", "anomaly_type", "first_deviation_line"],
    "additionalProperties": False,
}
SYSTEM = """You check one trace from production application logs: all log lines with the same \
correlation id, in time order. Lines are log templates where <*> is a masked value. Decide \
whether the trace is one complete, correct run of a business flow, and fill in every response \
field exactly as its description defines."""


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


def classify_one(client, model: str, lines: list[str], flows: str | None) -> tuple[dict, int, int]:
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(lines, flows),
        # the answer is ~100 tokens; the cap also stops the rate limiter reserving the model maximum
        max_completion_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA},
        },
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
    flows = (data_dir / "flows.yaml").read_text() if context == "expected-flow" else None
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
        context=context,
        git_sha=sha,
        git_dirty=dirty,
        split=split,
        data_manifest_sha256=data_manifest_hash(data_dir),
        versions=library_versions(),
        start_time=utc_now(),
    )
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda s: classify_one(client, model, s["lines"], flows), seqs))

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
        variant=f"{model} {context}",
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
