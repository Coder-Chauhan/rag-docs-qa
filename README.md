# RAG Docs QA

![CI](https://github.com/Coder-Chauhan/rag-docs-qa/actions/workflows/ci.yml/badge.svg)

A retrieval-augmented generation (RAG) API that answers questions about your own documents.
Upload PDFs, Markdown or text files, then ask questions in natural language and get grounded
answers with numbered source citations.

Built with **FastAPI**, **LangChain**, **ChromaDB** and the **OpenAI API**.

## How it works

```
Upload (.pdf/.md/.txt)
   -> extract text (pypdf)
   -> split into overlapping chunks (RecursiveCharacterTextSplitter)
   -> embed (OpenAI text-embedding-3-small)
   -> store in ChromaDB (persisted to disk)

Question
   -> embed question -> top-k similarity search in ChromaDB
   -> build prompt with numbered context chunks
   -> GPT-4o-mini answers using only that context, citing [1], [2] ...
   -> return answer + sources (file, page, snippet)
```

## Features

- REST API with automatic Swagger docs at `/docs`
- PDF, Markdown and plain-text ingestion with page-level metadata
- Grounded answers: the model is instructed to answer only from retrieved context and to say so when the answer is not in the documents
- Source citations with file name, page number and snippet
- Persistent vector store (survives restarts)
- Configurable models, chunk size, overlap and top-k via environment variables
- RAGAS evaluation suite (faithfulness, answer relevancy, context precision, context recall) with a CI quality gate
- Dependency-injected service layer, so the test suite runs fully offline with fake embeddings and a fake LLM (no API key or cost)
- Docker support and GitHub Actions CI

## Project structure

```
rag-docs-qa/
├── app/
│   ├── main.py        # FastAPI app and endpoints
│   ├── rag.py         # RAG service: loading, chunking, retrieval, generation
│   ├── schemas.py     # Pydantic request/response models
│   └── config.py      # Settings from environment / .env
├── eval/
│   ├── dataset.json   # Evaluation questions with reference answers
│   └── ragas_eval.py  # RAGAS evaluation runner
├── tests/
│   ├── test_api.py    # Offline API tests (pytest)
│   └── test_eval.py   # Offline tests for the evaluation harness
├── sample_docs/       # Example document to try it out
├── .github/workflows/  # ci.yml (tests), ragas-eval.yml (on-demand evaluation)
├── Dockerfile
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/Coder-Chauhan/rag-docs-qa.git
cd rag-docs-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

### With Docker

```bash
docker build -t rag-docs-qa .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data rag-docs-qa
```

## API

| Method | Endpoint     | Description                                  |
|--------|--------------|----------------------------------------------|
| GET    | `/health`    | Health check                                 |
| POST   | `/documents` | Upload and index a `.pdf`, `.md` or `.txt`   |
| GET    | `/documents` | List indexed documents and chunk count       |
| DELETE | `/documents` | Clear the vector store                       |
| POST   | `/query`     | Ask a question, get an answer with sources   |

### Example

```bash
curl -F "file=@sample_docs/company_handbook.md" http://localhost:8000/documents

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days can I work remotely?", "top_k": 3}'
```

```json
{
  "answer": "Employees can work remotely up to three days per week; fully remote work needs department head approval [1].",
  "sources": [
    { "id": 1, "source": "company_handbook.md", "page": null, "snippet": "## Remote Work\n\nEmployees can work remotely up to three days per week..." }
  ]
}
```

## Configuration

| Variable          | Default                  | Description                      |
|-------------------|--------------------------|----------------------------------|
| `OPENAI_API_KEY`  | (required)               | Your OpenAI API key              |
| `CHAT_MODEL`      | `gpt-4o-mini`            | Model used to generate answers   |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model                  |
| `CHROMA_DIR`      | `./data/chroma`          | Where the vector store is saved  |
| `CHUNK_SIZE`      | `1000`                   | Characters per chunk             |
| `CHUNK_OVERLAP`   | `150`                    | Overlap between chunks           |
| `TOP_K`           | `4`                      | Chunks retrieved per question    |
| `MAX_UPLOAD_MB`   | `20`                     | Maximum upload size              |

## Evaluation with RAGAS

The `eval/` folder measures answer quality with [RAGAS](https://docs.ragas.io), using an LLM as judge.
It ingests `sample_docs/` into a temporary in-memory collection, so it never touches the API's data,
and runs each question in `eval/dataset.json` through the same retrieval and generation code the API uses.

| Metric              | Question it answers                                                  | Needs reference |
|---------------------|----------------------------------------------------------------------|-----------------|
| `faithfulness`      | Are the answer's claims supported by the retrieved context?          | No              |
| `answer_relevancy`  | Does the answer actually address the question?                       | No              |
| `context_precision` | Are the relevant chunks ranked at the top of the results?            | Yes             |
| `context_recall`    | Does the retrieved context contain everything the answer needs?      | Yes             |

Faithfulness and answer relevancy grade generation; context precision and recall grade retrieval.

```bash
pip install -r requirements-eval.txt
python -m eval.ragas_eval                      # default: top_k=4, judge=gpt-4o-mini
python -m eval.ragas_eval --top-k 2 --fail-under 0.7
```

It prints a summary table and writes per-question scores to `eval/results/ragas_results.csv`, plus
`ragas_summary.json` with the run configuration. With `--fail-under`, the command exits with code 1
if any metric averages below the threshold, so it can act as a quality gate.

To evaluate your own documents, point `--docs` at a folder and `--dataset` at a JSON file of
`{"question": ..., "reference": ...}` objects. Rerun after changing chunk size, overlap, top-k or
the prompt to compare configurations.

**In GitHub Actions:** add an `OPENAI_API_KEY` repository secret, then run the *RAGAS evaluation*
workflow from the Actions tab. Results are uploaded as a build artifact.

## Testing

```bash
pytest -q
```

Tests use LangChain's `DeterministicFakeEmbedding` and `FakeListChatModel` with an in-memory
Chroma collection, so they need no API key and run in about a second. The evaluation harness is
tested the same way, with stand-in metrics in place of RAGAS.

## Possible extensions

- Hybrid search (BM25 + vectors) and re-ranking
- Streaming responses
- Per-user collections and authentication
