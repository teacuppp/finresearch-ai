from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


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

    query = input("Question: ")

    results = retriever.retrieve(
    query=query,
    top_k=5,
    )

    print("\nTop results:\n")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)

        print(f"Result #{index}")
        print(f"Document: {result.document}")
        print(f"Page: {result.page}")
        print(f"Chunk: {result.chunk_index}")
        print(f"Distance: {result.distance:.4f}")

        print("\nText:")
        print(result.text)

        print()


if __name__ == "__main__":
    main()