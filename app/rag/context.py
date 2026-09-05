from app.rag.models import RetrievedChunk


def build_context(
    chunks: list[RetrievedChunk],
) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Document: {chunk.document}",
                    f"Page: {chunk.page}",
                    f"Chunk: {chunk.chunk_index}",
                    "Content:",
                    chunk.text,
                ]
            )
        )

    return "\n\n".join(context_parts)