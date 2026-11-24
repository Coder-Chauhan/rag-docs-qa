"""Evaluate the RAG pipeline with RAGAS.

Ingests a folder of documents into a throwaway in-memory collection, runs every question in an
evaluation dataset through the same retrieval and generation code the API uses, then scores the
results with four RAGAS metrics using an LLM as judge:

- faithfulness:       are the answer's claims supported by the retrieved context? (hallucination)
- answer_relevancy:   does the answer address the question?
- context_precision:  are the relevant chunks ranked near the top of the retrieved context?
- context_recall:     does the retrieved context contain everything the reference answer needs?

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --top-k 3 --fail-under 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.rag import SUPPORTED_EXTENSIONS, RAGService

ROOT = Path(__file__).resolve().parent.parent
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalSample:
    question: str
    reference: str
    response: str = ""
    contexts: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class Metric(Protocol):
    name: str

    async def ascore(self, **kwargs: Any) -> Any: ...


# ---------- dataset and answers ----------


def load_dataset(path: Path) -> list[EvalSample]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalSample(question=r["question"], reference=r["reference"]) for r in rows]


def ingest_folder(rag: RAGService, docs_dir: Path) -> int:
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported documents found in {docs_dir}")
    return sum(rag.ingest_file(p, p.name) for p in files)


def generate_answers(rag: RAGService, samples: list[EvalSample], k: int | None = None) -> list[EvalSample]:
    for s in samples:
        answer, docs = rag.answer_with_contexts(s.question, k)
        s.response = answer
        s.contexts = [d.page_content for d in docs]
    return samples


# ---------- scoring ----------


def metric_inputs(metric_name: str, s: EvalSample) -> dict[str, Any]:
    """Map a sample onto the arguments each RAGAS metric expects."""
    if metric_name == "faithfulness":
        return {"user_input": s.question, "response": s.response, "retrieved_contexts": s.contexts}
    if metric_name == "answer_relevancy":
        return {"user_input": s.question, "response": s.response}
    if metric_name in ("context_precision", "context_recall"):
        return {"user_input": s.question, "reference": s.reference, "retrieved_contexts": s.contexts}
    raise ValueError(f"Unknown metric '{metric_name}'")


async def _score_one(metric: Metric, sample: EvalSample, sem: asyncio.Semaphore) -> None:
    async with sem:
        try:
            result = await metric.ascore(**metric_inputs(metric.name, sample))
            value = getattr(result, "value", result)
            sample.scores[metric.name] = float(value) if value is not None else math.nan
        except Exception as exc:  # one failed judgement should not sink the whole run
            print(f"  ! {metric.name} failed for '{sample.question[:50]}': {exc}", file=sys.stderr)
            sample.scores[metric.name] = math.nan


async def score_samples(samples: list[EvalSample], metrics: list[Metric], concurrency: int = 4) -> list[EvalSample]:
    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*(_score_one(m, s, sem) for s in samples for m in metrics))
    return samples


def summarise(samples: list[EvalSample], metric_names: list[str]) -> dict[str, float]:
    summary = {}
    for name in metric_names:
        values = [s.scores.get(name, math.nan) for s in samples]
        valid = [v for v in values if not math.isnan(v)]
        summary[name] = round(sum(valid) / len(valid), 4) if valid else math.nan
    return summary


def failing_metrics(summary: dict[str, float], threshold: float) -> list[str]:
    return [name for name, value in summary.items() if math.isnan(value) or value < threshold]


# ---------- output ----------


def write_results(samples: list[EvalSample], summary: dict[str, float], config: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(summary)
    with (out_dir / "ragas_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "response", "reference", *names, "contexts"])
        for s in samples:
            writer.writerow(
                [s.question, s.response, s.reference, *[s.scores.get(n) for n in names], " ||| ".join(s.contexts)]
            )
    (out_dir / "ragas_summary.json").write_text(
        json.dumps({"config": config, "summary": summary, "samples": [asdict(s) for s in samples]}, indent=2),
        encoding="utf-8",
    )


def print_summary(summary: dict[str, float], threshold: float | None) -> None:
    print("\nRAGAS results")
    print("-" * 40)
    for name, value in summary.items():
        flag = ""
        if threshold is not None:
            flag = "  FAIL" if math.isnan(value) or value < threshold else "  ok"
        print(f"{name:<20} {value:>6.3f}{flag}")
    print("-" * 40)


# ---------- wiring for a real run ----------


def build_metrics(judge_model: str, embedding_model: str, api_key: str) -> list[Metric]:
    """Create RAGAS metrics backed by OpenAI. Imported lazily so tests don't need ragas installed."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    client = AsyncOpenAI(api_key=api_key)
    llm = llm_factory(judge_model, client=client)
    embeddings = RagasOpenAIEmbeddings(client=client, model=embedding_model)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    # Normalise names so results line up with METRIC_NAMES regardless of library defaults.
    for metric, name in zip(metrics, METRIC_NAMES):
        metric.name = name
    return metrics


def build_eval_service(top_k: int) -> RAGService:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    from app.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    return RAGService(
        embeddings=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
        llm=ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key),
        persist_dir=None,  # in-memory: never touches the API's persistent store
        collection_name=f"ragas-eval-{uuid.uuid4().hex[:8]}",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        default_k=top_k,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS")
    p.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.json")
    p.add_argument("--docs", type=Path, default=ROOT / "sample_docs")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--judge-model", default="gpt-4o-mini", help="LLM used by RAGAS to grade answers")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--fail-under", type=float, default=None, help="Exit 1 if any metric averages below this")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    args = parse_args(argv)
    settings = get_settings()

    rag = build_eval_service(args.top_k)
    chunks = ingest_folder(rag, args.docs)
    samples = load_dataset(args.dataset)
    print(f"Ingested {chunks} chunks from {args.docs}. Evaluating {len(samples)} questions (top_k={args.top_k})...")

    generate_answers(rag, samples, args.top_k)
    metrics = build_metrics(args.judge_model, settings.embedding_model, settings.openai_api_key)
    asyncio.run(score_samples(samples, metrics, args.concurrency))

    summary = summarise(samples, METRIC_NAMES)
    config = {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "questions": len(samples),
    }
    write_results(samples, summary, config, args.out)
    print_summary(summary, args.fail_under)
    print(f"Per-question results: {args.out / 'ragas_results.csv'}")

    if args.fail_under is not None and (failed := failing_metrics(summary, args.fail_under)):
        print(f"Below threshold {args.fail_under}: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
