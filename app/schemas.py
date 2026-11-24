from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What is the refund policy?"])
    top_k: int | None = Field(default=None, ge=1, le=20, description="Number of chunks to retrieve")


class Source(BaseModel):
    id: int
    source: str
    page: int | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int


class StatsResponse(BaseModel):
    collection: str
    total_chunks: int
    documents: list[str]
