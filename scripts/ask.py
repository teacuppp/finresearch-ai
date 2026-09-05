from app.rag.embeddings import EmbeddingModel
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import RAGPipeline
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

    generator = AnswerGenerator(
        model="qwen3:4b",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    question = input("Question: ")

    result = pipeline.ask(
        question=question,
        top_k=5,
    )

    print("\nAnswer:\n")
    print(result.answer)

    print("\nRetrieved sources:\n")

    for index, source in enumerate(
        result.sources,
        start=1,
    ):
        print(
            f"[Source {index}] "
            f"{source.document}, "
            f"page {source.page}, "
            f"chunk {source.chunk_index}, "
            f"distance={source.distance:.4f}"
        )


if __name__ == "__main__":
    main()



#    python -m scripts.ask