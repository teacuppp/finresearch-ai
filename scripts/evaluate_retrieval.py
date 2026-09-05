from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


QUERIES = [
    "What was Apple's total revenue in 2025?",
    "What was Apple's net income in 2025?",
    "How much revenue did Apple generate from services in 2025?",
    "How much revenue did Apple generate from products in 2025?",
    "What were Apple's operating cash flows in 2025?",
    "What were Apple's research and development expenses in 2025?",
]


def main():
    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    for query in QUERIES:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(f"\nRank #{rank}")
            print(f"Page: {result.page}")
            print(f"Chunk: {result.chunk_index}")
            print(f"Distance: {result.distance:.4f}")

            preview = result.text[:700].replace("\n", " ")

            print(f"Text: {preview}")


if __name__ == "__main__":
    main()