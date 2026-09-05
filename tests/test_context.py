from app.rag.context import build_context
from app.rag.models import RetrievedChunk


def test_build_context():
    chunks = [
        RetrievedChunk(
            text="Apple reported total net sales of $416.161 billion.",
            document="apple.pdf",
            page=35,
            chunk_index=0,
            distance=0.5,
        ),
        RetrievedChunk(
            text="Apple reported net income of $112.010 billion.",
            document="apple.pdf",
            page=36,
            chunk_index=0,
            distance=0.6,
        ),
    ]

    context = build_context(chunks)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "apple.pdf" in context
    assert "Page: 35" in context
    assert "$416.161 billion" in context