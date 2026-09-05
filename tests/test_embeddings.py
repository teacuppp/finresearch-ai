from app.rag.embeddings import EmbeddingModel


def test_embed_documents():
    model = EmbeddingModel()

    texts = [
        "Tencent reported strong revenue growth.",
        "Alibaba increased cloud computing revenue.",
    ]

    embeddings = model.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])


def test_embed_query():
    model = EmbeddingModel()

    embedding = model.embed_query(
        "What was Tencent's revenue?"
    )

    assert len(embedding) > 0