from pdfqanda.splitter import TextChunk
from pdfqanda.vectorstore import TfidfVectorStore


def test_vectorstore_query_returns_ranked_results():
    chunks = [
        TextChunk(content="The sky is blue and clear.", index=0, start_char=0, end_char=25),
        TextChunk(content="Grass is green and lush.", index=1, start_char=26, end_char=48),
        TextChunk(content="Roses are red and violets are blue.", index=2, start_char=49, end_char=86),
    ]

    store = TfidfVectorStore()
    store.fit(chunks)

    results = store.query("What color is the sky?", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.index == 0
    assert results[0].score >= results[1].score
