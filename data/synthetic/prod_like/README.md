# prod_like: synthetic stand-in for the customer's production extract

Invented data that mimics a production log extract a customer could bring to a workshop.
Nothing here is real. Regenerate with `make data-prod-like`
(config: [configs/data_prod_like.yaml](../../../configs/data_prod_like.yaml)).

## What a customer extract would contain

| Customer material | File here |
| --- | --- |
| 10 consecutive days of Splunk application logs, unlabelled | `logs/prod_<date>.log` |
| 2 major incidents inside the window | `incidents.csv` (known to the customer, not in the logs) |
| Drain template list with ids and text, values masked | `drain_templates.csv` |
| 50 to 100 normal and incident traces with expected verdict and explanation | `traces_labelled.jsonl` (80 traces: 50 normal, 30 anomalous) |
| Correlation ids | `corrId=` on every line of a trace, `-` on background lines |
| Expected behaviour of 3 to 5 critical business flows | `flows.yaml` (4 flows) |
| Current precision, recall and F1 | not generated: only the customer has these |

`truth/trace_truth.csv` labels every trace and exists only for this rehearsal. The customer
will not have it, so never feed it to a model or a prompt.

## Split files for the pipeline

`write_splits` in `generate_prod_like.py` turns the extract into the files `logpoc prepare`
reads, using only the customer material above (logs, labelled traces, incident windows):

| File | Content |
| --- | --- |
| `raw_test.log`, `labels_test.csv` | The 80 labelled traces. Both PoCs are scored on these |
| `raw_train.log`, `raw_val.log` | All other traces that started outside the incident windows, 90/10. Unlabelled and mostly normal, so a few anomalies leak in, as in real data |
| `raw_dev.log`, `raw_fewshot.log` | Empty, kept so the existing pipeline finds every split |

Background lines (`corrId=-`) are dropped: both PoCs judge one trace at a time.

## Shape

- Log line: `<ts> <LEVEL> <service> host=<pod> corrId=<id> <message>`, UTC+01:00.
- Flows: `mortgage_application`, `instalment_collection` (nightly batch),
  `loan_conversion_quote`, `disbursement`.
- Background noise with `corrId=-`: health probes, scheduler heartbeats, config refreshes, GC.
- Harmless variations the detector must not flag: MitID token refresh, valuation served from
  cache, a single ledger retry, and slow-but-complete valuations before INC-1.
- Anomaly types: `visible_error`, `silent_skip`, `wrong_order`, `truncated`, `retry_storm`.

## Incidents

| Id | When | What | Early signal |
| --- | --- | --- | --- |
| INC-1 | Thu 2026-03-05 14:10 to 15:25 | Valuation provider degraded. Applications skip the credit assessment (silent), fail visibly or stop early. | WARN `valuation provider latency p95 high` and slow retries from 13:35 |
| INC-2 | Tue 2026-03-10 02:00 to 04:20 | Ledger posting backlog in the nightly collection. Collections stop before posting, repeat the payment instruction or send the receipt before settlement. | WARN `posting queue depth high` from 01:30 |

A small number of random anomalies (about 0.4 percent of traces) also occur outside the incidents.
