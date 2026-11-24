import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.rag import SUPPORTED_EXTENSIONS, RAGService
from app.schemas import IngestResponse, QueryRequest, QueryResponse, StatsResponse


def build_default_service() -> RAGService:
    """Create the production service backed by OpenAI and a persistent Chroma store."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    return RAGService(
        embeddings=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
        llm=ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
        ),
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        default_k=settings.top_k,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests can set app.state.rag beforehand to inject a fake service.
    if getattr(app.state, "rag", None) is None:
        app.state.rag = build_default_service()
    yield


app = FastAPI(
    title="RAG Docs QA",
    description="Ask questions about your documents using retrieval-augmented generation.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["documents"])
async def upload_document(file: UploadFile = File(...), rag: RAGService = Depends(get_rag)) -> IngestResponse:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / filename
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if tmp_path.stat().st_size > max_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

        try:
            added = rag.ingest_file(tmp_path, filename)
        except Exception as exc:  # loader or parsing errors
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Could not process file: {exc}")

    if added == 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="No extractable text found")

    return IngestResponse(filename=filename, chunks_added=added, total_chunks=rag.count())


@app.get("/documents", response_model=StatsResponse, tags=["documents"])
def list_documents(rag: RAGService = Depends(get_rag)) -> StatsResponse:
    return StatsResponse(collection=rag.collection_name, total_chunks=rag.count(), documents=rag.list_documents())


@app.delete("/documents", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
def reset_documents(rag: RAGService = Depends(get_rag)) -> None:
    rag.reset()


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(body: QueryRequest, rag: RAGService = Depends(get_rag)) -> QueryResponse:
    return QueryResponse(**rag.ask(body.question, body.top_k))
