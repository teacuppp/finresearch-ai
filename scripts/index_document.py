from pathlib import Path

from app.rag.embeddings import EmbeddingModel
from app.rag.ingestion import process_pdf
from app.rag.vector_store import VectorStore


PDF_PATH = Path(
    "data/documents/nasdaq-aapl-2025-10K-251437791.pdf"
)


def main():
    print("Processing PDF...")

    chunks = process_pdf(PDF_PATH)

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")

    embedding_model = EmbeddingModel()

    print("Generating embeddings...")

    texts = [chunk.text for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    vector_store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    print("Saving chunks to Chroma...")

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Indexing completed.")


if __name__ == "__main__":
    main()