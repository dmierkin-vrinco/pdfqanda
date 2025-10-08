from pdfqanda.splitter import TextSplitter


def test_splitter_respects_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    splitter = TextSplitter(chunk_size=10, overlap=2)
    chunks = splitter.split_text(text)

    assert len(chunks) == 3
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content.startswith("ijklmnop")
    assert chunks[1].start_char == 8
    assert chunks[1].end_char == 18
