from app.rag.vector_store import VectorStore

# KEYWORD = "416,161"
# TICKER = "AAPL"

# KEYWORD = "109,158"
# TICKER = "AAPL"

KEYWORD = "128,528"
TICKER = "MSFT"

#查找chunks
def main():
    store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    results = store.collection.get(
        where={
            "ticker": {
                "$eq": TICKER
            }
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    rows = []

    for text, metadata in zip(
        results["documents"],
        results["metadatas"],
    ):
        rows.append(
            (
                metadata["page"],
                metadata["chunk_index"],
                text,
            )
        )

    rows.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    for page, chunk_index, text in rows:

        if KEYWORD.lower() not in text.lower():
            continue

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"Page {page} | Chunk {chunk_index}"
        )

        print(
            "=" * 100
        )

        print(text)


if __name__ == "__main__":
    main()


#    python -m scripts.inspect_document_chunks     
# # 假设文件名为 check_chunks.py