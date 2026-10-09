import argparse
import asyncio
import csv
import inspect
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import query

TESTSET_KEYS = ("user_input", "reference", "reference_contexts", "_meta")

ABSTENTION_PROMPT = """You are auditing a retrieval-augmented QA system over a fixed set of \
documents.

Question asked by the user:
{question}

Passages the system retrieved:
{contexts}

The system's answer:
{answer}

Decide whether the ANSWER engaged with the question by providing substantive information,
or whether it declined/claimed it could not be answered from the passages.

Reply with exactly one word:
ANSWERED  - the answer provides substantive information drawn from the passages
ABSTAINED - the answer refuses, says the information is not in the documents, or says it
cannot determine the answer
"""


def use_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_testset(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in items:
        for key in TESTSET_KEYS:
            item.setdefault(key, None)
        item.setdefault("_meta", {})
    return items


def apply_selection(items, limit=None, answerable=None, ids=None):
    selected = items
    if answerable is not None:
        selected = [i for i in selected if bool(i["_meta"].get("answerable")) is answerable]
    if ids:
        wanted = set(ids)
        selected = [i for i in selected if i["_meta"].get("id") in wanted]
    if limit:
        selected = selected[:limit]
    return selected


DAILY_LIMIT_MARKERS = ("free-models-per-day", "free_tier_daily", "RateLimit-Reset")


def _is_daily_quota_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker.lower() in text for marker in DAILY_LIMIT_MARKERS)


def run_rag(items, out_path):
    if out_path.exists():
        cached = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        cached = {}
    cached = {k: v for k, v in cached.items() if isinstance(v, dict) and not v.get("error")}

    for position, item in enumerate(items, 1):
        item_id = item["_meta"].get("id") or f"idx{position:03d}"
        if item_id in cached:
            print(f"[{position}/{len(items)}] {item_id} cached")
            continue

        started = time.time()
        try:
            response = query.query_with_failover(item["user_input"])
        except Exception as exc:
            if _is_daily_quota_error(exc):
                _dump_json(cached, out_path)
                remaining = len(items) - position + 1
                print()
                print("=" * 66)
                print("ABORTED: OpenRouter free-tier daily quota is exhausted.")
                print(f"{len(cached)} answer(s) already generated are saved in {out_path}")
                print(f"{remaining} item(s) left. The quota resets daily (50 free requests/day).")
                print("Re-run the same command later to resume from where it stopped.")
                print("=" * 66)
                return cached
            elapsed = time.time() - started
            cached[item_id] = {
                "id": item_id,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(elapsed, 2),
            }
            print(f"[{position}/{len(items)}] {item_id} FAILED after {elapsed:.1f}s")
            _dump_json(cached, out_path)
            continue

        elapsed = time.time() - started
        model_used = getattr(response, "model_used", None)
        cached[item_id] = {
            "id": item_id,
            "model": model_used,
            "answer": response.response,
            "contexts": [n.node.get_content() for n in response.source_nodes],
            "sources": [
                {
                    "score": round(float(n.score), 4) if n.score is not None else None,
                    "file_name": n.node.metadata.get("file_name"),
                    "header_path": n.node.metadata.get("header_path"),
                }
                for n in response.source_nodes
            ],
            "seconds": round(elapsed, 2),
        }
        via = f" via {model_used}" if model_used and model_used != config.LLM_MODEL else ""
        print(f"[{position}/{len(items)}] {item_id} ok ({elapsed:.1f}s, "
              f"{len(response.source_nodes)} contexts){via}")
        _dump_json(cached, out_path)

    return cached


def _dump_json(data, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def score_ragas(items, outputs, out_path):
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    metric_classes = {
        "faithfulness": (Faithfulness, False),
        "answer_relevancy": (AnswerRelevancy, True),
        "context_precision": (ContextPrecision, True),
        "context_recall": (ContextRecall, True),
    }
    selected = [name for name in config.EVAL_METRICS if name in metric_classes]
    unknown = [name for name in config.EVAL_METRICS if name not in metric_classes]
    if unknown:
        print(f"warning: unknown metrics ignored: {', '.join(unknown)}")
    if not selected:
        print("warning: no known metrics selected, skipping RAGAS scoring")
        return None

    rows = []
    scored_ids = []
    for position, item in enumerate(items, 1):
        item_id = item["_meta"].get("id") or f"idx{position:03d}"
        output = outputs.get(item_id)
        if not output or output.get("error") or not output.get("answer"):
            continue
        if not item.get("reference_contexts"):
            continue
        rows.append(
            {
                "user_input": item["user_input"],
                "retrieved_contexts": output["contexts"],
                "reference_contexts": item["reference_contexts"],
                "response": output["answer"],
                "reference": item["reference"],
            }
        )
        scored_ids.append(item_id)

    if not rows:
        print("warning: no scorable rows for RAGAS, skipping")
        return None

    print(f"\nRAGAS scoring {len(rows)} item(s) x {len(selected)} metric(s) "
          f"with {config.describe_judge()} ...")
    llm = config.get_eval_judge_llm()
    needs_embeddings = any(metric_classes[name][1] for name in selected)
    embeddings = config.get_eval_embeddings() if needs_embeddings else None

    metrics = []
    for name in selected:
        metric_class, uses_embeddings = metric_classes[name]
        kwargs = {"llm": llm}
        if uses_embeddings:
            kwargs["embeddings"] = embeddings
        metrics.append((name, metric_class(**kwargs)))

    scores = asyncio.run(_score_all(metrics, rows, scored_ids, selected))
    _dump_json(scores, out_path)
    return rows, scored_ids, scores


async def _score_all(metrics, rows, scored_ids, selected):
    semaphore = asyncio.Semaphore(max(1, config.EVAL_BATCH_SIZE))
    state = {item_id: {name: None for name in selected} for item_id in scored_ids}
    reasons = {item_id: {name: None for name in selected} for item_id in scored_ids}
    total = len(rows) * len(metrics)
    done = 0

    async def run_one(name, metric, item_id, row, fields):
        nonlocal done
        async with semaphore:
            try:
                result = await metric.ascore(**fields)
                state[item_id][name] = _as_float(result.value)
                reasons[item_id][name] = getattr(result, "reason", None)
            except Exception as exc:
                state[item_id][name] = None
                reasons[item_id][name] = f"{type(exc).__name__}: {exc}"
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  ragas {done}/{total} scored")

    tasks = []
    for name, metric in metrics:
        wanted = [p for p in inspect.signature(type(metric).ascore).parameters if p != "self"]
        missing = [p for p in wanted if p not in rows[0]]
        if missing:
            raise RuntimeError(f"metric '{name}' needs unsupported fields: {', '.join(missing)}")
        for item_id, row in zip(scored_ids, rows):
            tasks.append(run_one(name, metric, item_id, row, {p: row[p] for p in wanted}))

    await asyncio.gather(*tasks)

    return [
        {"id": item_id, **state[item_id], "reasons": reasons[item_id]}
        for item_id in scored_ids
    ]


def judge_abstention(items, outputs, out_path):
    client = config.get_judge_client()
    model = config.EVAL_JUDGE_MODEL
    cache = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    for position, item in enumerate(items, 1):
        item_id = item["_meta"].get("id") or f"idx{position:03d}"
        output = outputs.get(item_id)
        if not output or output.get("error") or not output.get("answer"):
            continue
        if item_id in cache:
            continue

        joined = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(output["contexts"], 1)) or "(none)"
        prompt = ABSTENTION_PROMPT.format(
            question=item["user_input"],
            contexts=joined,
            answer=output["answer"],
        )
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=config.EVAL_JUDGE_TEMPERATURE,
            )
            raw = (completion.choices[0].message.content or "").strip()
            label = "ABSTAINED" if raw.upper().startswith("ABSTAIN") else "ANSWERED"
            cache[item_id] = {"id": item_id, "label": label, "raw": raw}
        except Exception as exc:
            cache[item_id] = {
                "id": item_id,
                "label": "ERROR",
                "raw": f"{type(exc).__name__}: {exc}",
            }
        print(f"[abstention {position}/{len(items)}] {item_id} -> {cache[item_id]['label']}")
        _dump_json(cache, out_path)

    return cache


def summarize(items, outputs, scored_ids, ragas_scores, abstention, run_dir):
    metric_names = [m for m in config.EVAL_METRICS if ragas_scores]
    scores_by_id = {}
    if ragas_scores:
        scores_by_id = {row["id"]: row for row in ragas_scores}

    abstention_by_id = {v["id"]: v for v in abstention.values() if "id" in v}
    per_item = {}

    for position, item in enumerate(items, 1):
        item_id = item["_meta"].get("id") or f"idx{position:03d}"
        output = outputs.get(item_id) or {}
        row = {
            "id": item_id,
            "answerable": bool(item["_meta"].get("answerable")),
            "type": item["_meta"].get("type"),
            "sources_expected": ";".join(item["_meta"].get("sources") or []),
            "question": item["user_input"],
            "answer": output.get("answer", ""),
            "reference": item.get("reference"),
            "n_contexts": len(output.get("contexts") or []),
            "files_retrieved": ";".join(
                sorted({s["file_name"] for s in output.get("sources") or [] if s["file_name"]})
            ),
            "generation_seconds": output.get("seconds"),
            "model_used": output.get("model"),
            "error": output.get("error"),
        }
        scores = scores_by_id.get(item_id) or {}
        row.update({name: _as_float(scores.get(name)) for name in metric_names})
        row["ragas_errors"] = ";".join(
            name for name in metric_names if scores.get("reasons", {}).get(name)
        ) or None
        label = abstention_by_id.get(item_id, {}).get("label")
        if label:
            row["abstention_label"] = label
        per_item[item_id] = row

    summary = {
        "run_dir": str(run_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "generator_model": config.LLM_MODEL,
            "generator_model_used": _models_used(outputs),
            "generator_fallbacks": config.LLM_FALLBACK_MODELS,
            "judge_model": config.describe_judge(),
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "top_k": config.TOP_K,
            "rerank_top_n": config.RERANK_TOP_N,
            "hybrid_alpha": config.HYBRID_ALPHA,
            "enable_hybrid": config.ENABLE_HYBRID,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "chunker": config.CHUNKER,
            "metrics": config.EVAL_METRICS,
        },
        "metric_failures": _metric_failures(per_item),
        "counts": _counts(items, outputs, scored_ids, abstention),
        "averages": _averages(per_item),
    }
    return summary, per_item


def _models_used(outputs):
    """Tally which model actually produced each cached answer.

    ``config.LLM_MODEL`` is only the *configured* generator — a fallback may have
    answered, or the cache may predate provenance tracking.
    """
    counts = {}
    for out in outputs.values():
        if out.get("error") or not out.get("answer"):
            continue
        model = out.get("model") or "unknown (cached before provenance was tracked)"
        counts[model] = counts.get(model, 0) + 1
    return counts


def _metric_failures(per_item):
    failures = {}
    for row in per_item.values():
        for name in (row.get("ragas_errors") or "").split(";"):
            if name:
                failures[name] = failures.get(name, 0) + 1
    return failures


def _counts(items, outputs, scored_ids, abstention):
    answerable = sum(1 for i in items if i["_meta"].get("answerable"))
    failed = sum(1 for o in outputs.values() if o.get("error"))
    return {
        "total": len(items),
        "answerable": answerable,
        "unanswerable": len(items) - answerable,
        "generation_failed": failed,
        "scored_by_ragas": len(scored_ids or []),
        "abstention_judged": len(abstention or {}),
    }


def _averages(per_item):
    metrics = [m for m in config.EVAL_METRICS if m in ("faithfulness", "answer_relevancy",
                                                       "context_precision", "context_recall")]
    out = {"answerable_only": {}, "all_scored": {}}
    scored = [r for r in per_item.values() if not r.get("error")]
    answerable_scored = [r for r in scored if r["answerable"]]
    for bucket, rows in (("answerable_only", answerable_scored), ("all_scored", scored)):
        for metric in metrics:
            values = [_as_float(r.get(metric)) for r in rows]
            values = [v for v in values if v is not None]
            out[bucket][metric] = round(sum(values) / len(values), 4) if values else None
        out[bucket]["scored_items"] = len(rows)

    abstain_labels = [r["abstention_label"] for r in scored if "abstention_label" in r]
    unans = [r for r in scored if not r["answerable"] and "abstention_label" in r]
    ans = [r for r in scored if r["answerable"] and "abstention_label" in r]
    out["abstention"] = {
        "abstention_rate_unanswerable": round(
            sum(1 for r in unans if r["abstention_label"] == "ABSTAINED") / len(unans), 4
        ) if unans else None,
        "answer_rate_answerable": round(
            sum(1 for r in ans if r["abstention_label"] == "ANSWERED") / len(ans), 4
        ) if ans else None,
        "unanswerable_judged": len(unans),
        "answerable_judged": len(ans),
        "errors": sum(1 for label in abstain_labels if label == "ERROR"),
    }
    return out


def _as_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return round(number, 4)


def write_csv(per_item, path):
    columns = [
        "id", "answerable", "type", "sources_expected", "files_retrieved", "n_contexts",
        "generation_seconds", "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "abstention_label", "ragas_errors", "error", "question",
        "answer", "reference",
    ]
    rows = [{c: r.get(c) for c in columns} for r in per_item.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    config_block = summary["config"]
    counts = summary["counts"]
    averages = summary["averages"]

    print("\n" + "=" * 66)
    print("EVALUATION SUMMARY")
    print("=" * 66)
    print(f"generator : {config_block['generator_model']}")
    used = config_block.get("generator_model_used") or {}
    if used:
        tally = ", ".join(f"{model} x{n}" for model, n in sorted(used.items(), key=lambda kv: -kv[1]))
        if set(used) - {config_block["generator_model"]}:
            print(f"            ^ ANSWERED BY DIFFERENT MODEL: {tally}")
        else:
            print(f"            ^ answers by model: {tally}")
    print(f"judge     : {config_block['judge_model']}")
    print(f"retrieval : hybrid={config_block['enable_hybrid']} alpha={config_block['hybrid_alpha']} "
          f"top_k={config_block['top_k']} rerank_n={config_block['rerank_top_n']}")
    print(f"items     : {counts['total']} total / {counts['answerable']} answerable / "
          f"{counts['unanswerable']} unanswerable / {counts['generation_failed']} failed")

    print("\nRAGAS metrics (answerable items only)")
    print("-" * 66)
    for metric, value in averages["answerable_only"].items():
        if metric == "scored_items":
            continue
        bar = _bar(value)
        print(f"  {metric:<20} {value if value is not None else 'n/a':>7}  {bar}")

    abstention = averages["abstention"]
    print("\nAbstention behaviour")
    print("-" * 66)
    print(f"  abstains on unanswerable : {abstention['abstention_rate_unanswerable']} "
          f"(n={abstention['unanswerable_judged']})  <- higher is better")
    print(f"  answers when it can     : {abstention['answer_rate_answerable']} "
          f"(n={abstention['answerable_judged']})  <- higher is better")
    if abstention["errors"]:
        print(f"  judge errors            : {abstention['errors']}")

    failures = summary.get("metric_failures") or {}
    if failures:
        print("\nMetric failures (scored as None, excluded from averages)")
        print("-" * 66)
        for name, count in sorted(failures.items()):
            print(f"  {name:<20} {count}")
    print()


def _bar(value):
    if value is None:
        return ""
    filled = int(round(value * 20))
    return "[" + "#" * filled + "." * (20 - filled) + "]"


def main():
    use_utf8_stdout()

    argp = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS")
    argp.add_argument("--testset", default=config.EVAL_TESTSET_FILE, help="Testset JSON path")
    argp.add_argument("--run-dir", default=None, help="Output dir (default: eval_runs/<timestamp>)")
    argp.add_argument("--limit", type=int, default=None, help="Only evaluate the first N items")
    argp.add_argument("--id", action="append", dest="ids", help="Evaluate only this testset id")
    argp.add_argument("--answerable", choices=["yes", "no"], default=None,
                      help="Restrict to answerable or unanswerable items")
    argp.add_argument("--rerun-rag", action="store_true", help="Ignore the cached RAG outputs")
    argp.add_argument("--no-ragas", action="store_true", help="Skip RAGAS metrics, abstention only")
    argp.add_argument("--no-abstention", action="store_true", help="Skip the abstention judge")
    args = argp.parse_args()

    items = apply_selection(
        load_testset(args.testset),
        limit=args.limit,
        answerable=None if args.answerable is None else args.answerable == "yes",
        ids=args.ids,
    )
    if not items:
        print("error: no testset items selected")
        sys.exit(2)

    run_dir = Path(args.run_dir or Path(config.EVAL_OUTPUT_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testset   : {args.testset}")
    print(f"Items     : {len(items)}")
    print(f"Run dir   : {run_dir}")
    print(f"Generator : {config.LLM_MODEL}")
    print(f"Judge     : {config.describe_judge()}")
    print()

    rag_path = run_dir / "rag_outputs.json"
    if args.rerun_rag and rag_path.exists():
        rag_path.unlink()

    if config.LLM_MODEL.endswith(":free"):
        done = 0
        if rag_path.exists():
            done = len(json.loads(rag_path.read_text(encoding="utf-8")))
        todo = len(items) - done
        print(f"note: generator '{config.LLM_MODEL}' uses the free tier, which OpenRouter "
              f"caps at 50 requests/day for all free models combined.")
        if todo > 50:
            print(f"      {todo} generations needed -> this run cannot finish today.")
        elif todo > 0:
            print(f"      {todo} generations needed, only {50 - todo} requests of headroom.")
        print()

    outputs = run_rag(items, rag_path)

    ragas_result = None
    scored_ids = []
    if not args.no_ragas and config.EVAL_SCORE_ANSWERABLE:
        outcome = score_ragas(items, outputs, run_dir / "ragas_scores.json")
        if outcome:
            _, scored_ids, ragas_result = outcome

    abstention = {}
    if not args.no_abstention and config.EVAL_SCORE_ABSTENTION:
        abstention = judge_abstention(items, outputs, run_dir / "abstention.json")

    summary, per_item = summarize(
        items, outputs, scored_ids, ragas_result, abstention, run_dir
    )
    _dump_json(summary, run_dir / "summary.json")
    write_csv(per_item, run_dir / "per_item.csv")
    print_summary(summary)
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
