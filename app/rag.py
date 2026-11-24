"""Core retrieval-augmented generation logic.

The service is built from injectable components (embeddings, chat model, vector store
location) so it can run against OpenAI in production and against fakes in tests.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.

Rules:
- Use only the information in the context below. Do not use outside knowledge.
- If the context does not contain the answer, say "I couldn't find that in the uploaded documents."
- Cite the sources you used with their bracketed numbers, for example [1] or [2][3].
- Keep answers concise and factual.

Context:
{context}"""


class RAGService:
    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseChatModel,
        persist_dir: str | None,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        default_k: int = 4,
    ) -> None:
        self.embeddings = embeddings
        self.llm = llm
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.default_k = default_k
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.vectorstore = self._build_store()

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    # ---------- ingestion ----------

    @staticmethod
    def load_file(path: Path, display_name: str) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {sorted(SUPPORTED_EXTENSIONS)}")

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return [
                Document(page_content=page.extract_text() or "", metadata={"source": display_name, "page": i})
                for i, page in enumerate(reader.pages, start=1)  # 1-based pages for humans
            ]

        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": display_name})]

    def ingest_documents(self, docs: list[Document]) -> int:
        chunks = [c for c in self.splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            return 0
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def ingest_file(self, path: Path, display_name: str) -> int:
        return self.ingest_documents(self.load_file(path, display_name))

    # ---------- querying ----------

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        blocks = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            label = f"{doc.metadata.get('source', 'unknown')}" + (f", page {page}" if page else "")
            blocks.append(f"[{i}] ({label})\n{doc.page_content}")
        return "\n\n".join(blocks)

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        return self.vectorstore.similarity_search(question, k=k or self.default_k)

    def answer_with_contexts(self, question: str, k: int | None = None) -> tuple[str, list[Document]]:
        """Retrieve context and generate an answer. Returns the answer and the retrieved chunks."""
        docs = self.retrieve(question, k)
        answer = self.chain.invoke({"context": self._format_context(docs), "question": question})
        return answer, docs

    def ask(self, question: str, k: int | None = None) -> dict:
        if self.count() == 0:
            return {
                "answer": "No documents have been ingested yet. Upload a document first.",
                "sources": [],
            }

        answer, docs = self.answer_with_contexts(question, k)

        sources = [
            {
                "id": i,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:300],
            }
            for i, d in enumerate(docs, start=1)
        ]
        return {"answer": answer, "sources": sources}

    # ---------- management ----------

    def count(self) -> int:
        return self.vectorstore._collection.count()

    def list_documents(self) -> list[str]:
        data = self.vectorstore.get(include=["metadatas"])
        return sorted({m.get("source", "unknown") for m in data["metadatas"] if m})

    def reset(self) -> None:
        self.vectorstore.delete_collection()
        self.vectorstore = self._build_store()
