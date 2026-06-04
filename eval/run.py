"""
Evaluation harness for the cinematography RAG system.

Measures, against the labelled cases in eval/cases.py:
  - Retrieval precision  : of the notes retrieved, how many were relevant
  - Retrieval recall      : of the relevant notes, how many were retrieved
  - Retrieval latency     : time for the vector search (per case)
  - End-to-end latency    : retrieval + local LLM generation (per case)
  - Grounding rate        : fraction of [Cited] titles in the answer that were
                            actually retrieved (i.e. not invented)

Run (from repo root, Ollama running, index built):
    python -m eval.run                # retrieval metrics only (fast)
    python -m eval.run --full         # also generate answers (slower; needs LLM)

Prints a table and overall medians/means suitable for quoting on a CV.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time

from src.agent.analyst import _gather_notes
from src.agent import analyze

from .cases import CASES, EvalCase

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _prf(retrieved_titles: set[str], expected: set[str]) -> tuple[float, float]:
    if not retrieved_titles:
        return 0.0, 0.0
    true_pos = len(retrieved_titles & expected)
    precision = true_pos / len(retrieved_titles)
    recall = true_pos / len(expected) if expected else 0.0
    return precision, recall


def eval_retrieval(case: EvalCase) -> dict:
    t0 = time.perf_counter()
    notes = _gather_notes(case.features, question=None)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    titles = {n.title for n in notes}
    precision, recall = _prf(titles, case.expected_titles)
    hit = case.expected_titles & titles
    return {
        "name": case.name,
        "precision": precision,
        "recall": recall,
        "retrieval_ms": retrieval_ms,
        "retrieved": titles,
        "expected_hit": hit,
        "notes": notes,
    }


def eval_grounding(case: EvalCase, retrieved_titles: set[str]) -> dict:
    t0 = time.perf_counter()
    result = analyze(case.features)
    e2e_ms = (time.perf_counter() - t0) * 1000

    cited = set(CITATION_RE.findall(result.interpretation))
    # A citation is "grounded" if its title matches a retrieved note.
    grounded = {c for c in cited if c in retrieved_titles}
    grounding_rate = (len(grounded) / len(cited)) if cited else 1.0
    return {"e2e_ms": e2e_ms, "cited": cited, "grounding_rate": grounding_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also generate answers (needs LLM)")
    args = parser.parse_args()

    rows = []
    precisions, recalls, ret_lat = [], [], []
    e2e_lat, grounding = [], []

    print(f"\nEvaluating {len(CASES)} labelled scenes...\n")
    print(f"{'scene':<22}{'prec':>6}{'recall':>8}{'ret ms':>9}", end="")
    print(f"{'e2e ms':>9}{'ground':>8}" if args.full else "")
    print("-" * (45 + (17 if args.full else 0)))

    for case in CASES:
        r = eval_retrieval(case)
        precisions.append(r["precision"])
        recalls.append(r["recall"])
        ret_lat.append(r["retrieval_ms"])

        line = f"{r['name']:<22}{r['precision']:>6.2f}{r['recall']:>8.2f}{r['retrieval_ms']:>9.1f}"
        if args.full:
            g = eval_grounding(case, r["retrieved"])
            e2e_lat.append(g["e2e_ms"])
            grounding.append(g["grounding_rate"])
            line += f"{g['e2e_ms']:>9.0f}{g['grounding_rate']:>8.2f}"
        print(line)

    print("-" * (45 + (17 if args.full else 0)))
    print("\n=== SUMMARY ===")
    print(f"Mean retrieval precision : {statistics.mean(precisions):.0%}")
    print(f"Mean retrieval recall    : {statistics.mean(recalls):.0%}")
    print(f"Median retrieval latency : {statistics.median(ret_lat):.1f} ms")
    if args.full and e2e_lat:
        print(f"Median end-to-end latency: {statistics.median(e2e_lat)/1000:.1f} s")
        print(f"Mean grounding rate      : {statistics.mean(grounding):.0%}")
    print()


if __name__ == "__main__":
    main()
