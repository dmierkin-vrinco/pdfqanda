from pdfqanda.qa import PdfQaEngine
from pdfqanda.splitter import TextChunk
from pdfqanda.vectorstore import TfidfVectorStore


def test_engine_returns_answers(monkeypatch):
    engine = PdfQaEngine()

    # Patch store to avoid PDF I/O and chunking
    chunks = [
        TextChunk(content="Python is a programming language.", index=0, start_char=0, end_char=35),
        TextChunk(content="It is popular for data science.", index=1, start_char=36, end_char=68),
    ]

    store = TfidfVectorStore()
    store.fit(chunks)
    engine.store = store

    answers = engine.query("What is Python?")

    assert answers
    assert "programming" in answers[0].text.lower()
