"""
run_eval.py
Evaluate retrieval quality against the 25-query evaluation set.

Metrics:
  Recall@3  — fraction of queries where at least one expected citation
               appears in the top-3 results
  MRR       — Mean Reciprocal Rank of the first correct result (over top_k=10)

Targets: Recall@3 >= 0.85, MRR >= 0.75

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --top-k 5 --language en
    python -m eval.run_eval --topic deductions
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from eval.query_set import QUERY_SET


# ---------------------------------------------------------------------------
# Citation normalisation
# ---------------------------------------------------------------------------

def _normalise_citation(cit: str) -> str:
    """Lowercase, strip whitespace, collapse multiple spaces."""
    return " ".join(cit.lower().split())


def _citation_match(retrieved: list[str], expected: list[str]) -> int:
    """
    Return the 1-based rank of the first retrieved citation that matches
    any of the expected citations.  Returns 0 if no match in top-k.

    Matching is done by checking whether an expected citation is a prefix of
    a retrieved citation (to handle sub-paragraph vs section matching).
    For example, expected "ITA s.20(1)" matches retrieved "ITA s.20(1)(c)".
    """
    norm_expected = [_normalise_citation(e) for e in expected]
    for rank, cit in enumerate(retrieved, start=1):
        norm_cit = _normalise_citation(cit)
        for exp in norm_expected:
            if norm_cit.startswith(exp) or exp.startswith(norm_cit):
                return rank
    return 0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    top_k: int = 10,
    language: str = "en",
    as_of: str | None = None,
    topic_filter: str | None = None,
    quiet: bool = False,
) -> dict:
    """
    Run the evaluation suite.

    Returns a dict with overall and per-topic metrics:
      {
        "overall": {"recall_at_3": float, "mrr": float, "n": int},
        "by_topic": {topic: {"recall_at_3": float, "mrr": float, "n": int}},
        "failures": [{"query": str, "expected": list, "retrieved": list}],
      }
    """
    from api.retriever import search

    if as_of is None:
        as_of = date.today().isoformat()

    queries = QUERY_SET
    if topic_filter:
        queries = [q for q in queries if q["topic"] == topic_filter]
        if not queries:
            print(f"[eval] No queries found for topic '{topic_filter}'")
            return {}

    print(f"[eval] Running {len(queries)} queries  (top_k={top_k}, language={language}, as_of={as_of})")
    if topic_filter:
        print(f"[eval] Topic filter: {topic_filter}")

    reciprocal_ranks: list[float] = []
    recall_at_3_hits: list[bool] = []
    failures: list[dict] = []

    topic_data: dict[str, dict] = defaultdict(lambda: {"rr": [], "r3": []})

    for i, item in enumerate(queries, start=1):
        query = item["query"]
        expected = item["expected_citations"]
        topic = item["topic"]

        try:
            results = search(query=query, top_k=top_k, language=language, as_of=as_of)
        except Exception as exc:
            print(f"[eval] ERROR on query {i}: {exc}", file=sys.stderr)
            reciprocal_ranks.append(0.0)
            recall_at_3_hits.append(False)
            failures.append({"query": query, "expected": expected, "retrieved": [], "error": str(exc)})
            topic_data[topic]["rr"].append(0.0)
            topic_data[topic]["r3"].append(False)
            continue

        retrieved_citations = [r["citation"] for r in results]

        rank = _citation_match(retrieved_citations, expected)
        rr = 1.0 / rank if rank > 0 else 0.0
        hit_at_3 = rank > 0 and rank <= 3

        reciprocal_ranks.append(rr)
        recall_at_3_hits.append(hit_at_3)
        topic_data[topic]["rr"].append(rr)
        topic_data[topic]["r3"].append(hit_at_3)

        if not quiet:
            status = "HIT " if hit_at_3 else ("LATE" if rank > 0 else "MISS")
            rank_str = f"rank={rank}" if rank > 0 else "rank=-"
            print(f"  [{status}] Q{i:02d} ({topic:20s}) {rank_str:8s} | {query[:60]}")
            if not hit_at_3:
                print(f"         Expected: {expected}")
                print(f"         Top-3:    {retrieved_citations[:3]}")

        if not hit_at_3:
            failures.append(
                {
                    "query": query,
                    "expected": expected,
                    "retrieved": retrieved_citations[:5],
                    "rank": rank,
                    "topic": topic,
                }
            )

    overall_n = len(queries)
    overall_recall_at_3 = sum(recall_at_3_hits) / overall_n if overall_n else 0.0
    overall_mrr = sum(reciprocal_ranks) / overall_n if overall_n else 0.0

    # Per-topic aggregation
    by_topic = {}
    for topic, data in topic_data.items():
        n = len(data["rr"])
        r3 = sum(data["r3"]) / n if n else 0.0
        mrr = sum(data["rr"]) / n if n else 0.0
        by_topic[topic] = {"recall_at_3": round(r3, 4), "mrr": round(mrr, 4), "n": n}

    return {
        "overall": {
            "recall_at_3": round(overall_recall_at_3, 4),
            "mrr": round(overall_mrr, 4),
            "n": overall_n,
        },
        "by_topic": by_topic,
        "failures": failures,
    }


def print_report(results: dict) -> None:
    """Pretty-print the eval results."""
    if not results:
        return

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    overall = results["overall"]
    r3 = overall["recall_at_3"]
    mrr = overall["mrr"]
    n = overall["n"]

    r3_ok = "PASS" if r3 >= 0.85 else "FAIL"
    mrr_ok = "PASS" if mrr >= 0.75 else "FAIL"

    print(f"\nOverall  (n={n})")
    print(f"  Recall@3 : {r3:.4f}   target ≥ 0.85  [{r3_ok}]")
    print(f"  MRR      : {mrr:.4f}   target ≥ 0.75  [{mrr_ok}]")

    print(f"\nPer-topic breakdown:")
    header = f"  {'Topic':25s} {'n':>4}  {'Recall@3':>10}  {'MRR':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for topic, data in sorted(results["by_topic"].items()):
        print(
            f"  {topic:25s} {data['n']:>4}  {data['recall_at_3']:>10.4f}  {data['mrr']:>8.4f}"
        )

    failures = results.get("failures", [])
    if failures:
        print(f"\nMisses / late hits ({len(failures)} queries):")
        for f in failures:
            rank_str = f"rank={f.get('rank', '?')}" if f.get("rank", 0) > 0 else "MISS"
            print(f"  [{rank_str:8s}] {f['query'][:65]}")
            print(f"             expected : {f['expected']}")
            print(f"             top-3    : {f.get('retrieved', [])[:3]}")
    else:
        print("\nAll queries were hits in top-3.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for the ITA legislative search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve per query")
    parser.add_argument("--language", default="en", choices=["en", "fr"])
    parser.add_argument("--as-of", default=None, help="ISO date for temporal filtering")
    parser.add_argument("--topic", default=None, help="Filter to a single topic")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query output")
    args = parser.parse_args()

    results = run_eval(
        top_k=args.top_k,
        language=args.language,
        as_of=args.as_of,
        topic_filter=args.topic,
        quiet=args.quiet,
    )
    print_report(results)

    # Exit with non-zero status if targets not met
    overall = results.get("overall", {})
    if overall.get("recall_at_3", 0) < 0.85 or overall.get("mrr", 0) < 0.75:
        sys.exit(1)
