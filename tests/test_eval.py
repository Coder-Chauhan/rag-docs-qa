"""Offline tests for the RAGAS evaluation harness (no API key, no ragas calls)."""

import math
import uuid
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from eval.ragas_eval import (
    METRIC_NAMES,
    EvalSample,
    failing_metrics,
    generate_answers,
    ingest_folder,
    load_dataset,
    metric_inputs,
    score_samples,
    summarise,
    write_results,
)
from app.rag import RAGService

ROOT = Path(__file__).resolve().parent.parent


class FakeResult:
    def __init__(self, value):
        self.value = value


class FakeMetric:
    """Stands in for a RAGAS metric: records the inputs and returns a fixed score."""

    def __init__(self, name, value=0.8, fail=False):
        self.name = name
        self.value = value
        self.fail = fail
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("judge timed out")
        return FakeResult(self.value)


@pytest.fixture
def rag():
    return RAGService(
        embeddings=DeterministicFakeEmbedding(size=32),
        llm=FakeListChatModel(responses=["Up to three days per week [1]."]),
        persist_dir=None,
        collection_name=f"eval-{uuid.uuid4().hex}",
        chunk_size=300,
        chunk_overlap=30,
    )


def test_dataset_is_well_formed():
    samples = load_dataset(ROOT / "eval" / "dataset.json")
    assert len(samples) >= 5
    assert all(s.question.strip() and s.reference.strip() for s in samples)


def test_generate_answers_collects_contexts(rag):
    assert ingest_folder(rag, ROOT / "sample_docs") > 0
    samples = [EvalSample(question="How many remote days?", reference="Three days.")]
    generate_answers(rag, samples, k=2)
    assert samples[0].response == "Up to three days per week [1]."
    assert len(samples[0].contexts) == 2


def test_metric_inputs_match_ragas_signatures():
    s = EvalSample(question="q", reference="r", response="a", contexts=["c"])
    assert metric_inputs("faithfulness", s) == {"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}
    assert metric_inputs("answer_relevancy", s) == {"user_input": "q", "response": "a"}
    assert metric_inputs("context_recall", s) == {"user_input": "q", "reference": "r", "retrieved_contexts": ["c"]}
    with pytest.raises(ValueError):
        metric_inputs("bogus", s)


async def _score(samples, metrics):
    return await score_samples(samples, metrics, concurrency=2)


def test_scoring_summary_and_threshold(tmp_path):
    import asyncio

    samples = [EvalSample(question=f"q{i}", reference="r", response="a", contexts=["c"]) for i in range(3)]
    metrics = [
        FakeMetric("faithfulness", 0.9),
        FakeMetric("answer_relevancy", 0.8),
        FakeMetric("context_precision", 0.5),
        FakeMetric("context_recall", fail=True),
    ]
    asyncio.run(_score(samples, metrics))

    assert all(len(m.calls) == 3 for m in metrics)
    summary = summarise(samples, METRIC_NAMES)
    assert summary["faithfulness"] == 0.9
    assert summary["context_precision"] == 0.5
    assert math.isnan(summary["context_recall"])  # every judgement failed
    assert failing_metrics(summary, 0.7) == ["context_precision", "context_recall"]

    write_results(samples, summary, {"top_k": 4}, tmp_path)
    assert (tmp_path / "ragas_results.csv").read_text().startswith("question,response,reference")
    assert (tmp_path / "ragas_summary.json").exists()
