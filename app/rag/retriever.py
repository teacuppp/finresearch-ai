from app.rag.embeddings import EmbeddingModel
from app.rag.models import RetrievedChunk
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                RetrievedChunk(
                    text=document,
                    document=metadata["document"],
                    page=metadata["page"],
                    chunk_index=metadata["chunk_index"],
                    distance=float(distance),
                    company=metadata.get("company"),
                    ticker=metadata.get("ticker"),
                    fiscal_year=metadata.get("fiscal_year"),
                    document_type=metadata.get("document_type"),
                )
            )

        return retrieved_chunks