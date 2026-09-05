from pathlib import Path

import chromadb

from app.rag.models import DocumentChunk


class VectorStore:
    def __init__(
        self,
        path: str = "data/chroma",
        collection_name: str = "financial_documents",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=path,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings"
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = (
                f"{chunk.document}"
                f"-p{chunk.page}"
                f"-c{chunk.chunk_index}"
            )

            ids.append(chunk_id)
            documents.append(chunk.text)

            metadata = {
                "document": chunk.document,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            }

            if chunk.company is not None:
                metadata["company"] = chunk.company

            if chunk.ticker is not None:
                metadata["ticker"] = chunk.ticker

            if chunk.fiscal_year is not None:
                metadata["fiscal_year"] = chunk.fiscal_year

            if chunk.document_type is not None:
                metadata["document_type"] = chunk.document_type

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where is not None:
            query_kwargs["where"] = where

        return self.collection.query(
            #字典解包运算符（Dictionary Unpacking Operator）”
            **query_kwargs
        )

    def count(
        self,
    ) -> int:
        return self.collection.count()

    def delete_document(
        self,
        document: str,
    ) -> None:
        self.collection.delete(
            where={
                "document": {
                    "$eq": document
                }
            }
        )

    def list_tickers(
        self,
    ) -> set[str]:
        results = self.collection.get(
            include=["metadatas"],
        )

        tickers = set()

        for metadata in results["metadatas"]:
            ticker = metadata.get("ticker")

            if ticker:
                tickers.add(ticker)

        return tickers


