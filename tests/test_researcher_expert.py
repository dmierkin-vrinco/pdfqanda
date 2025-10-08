from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from pdfqanda.db import Database
from pdfqanda.expert import CitationError, Expert
from pdfqanda.ingest import PdfIngestPipeline
from pdfqanda.models import ResearchHit
from pdfqanda.researcher import Researcher

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "input" / "sample.pdf"


def _ingest(tmp_path):
    db_path = tmp_path / "kb.sqlite"
    database = Database(f"sqlite:///{db_path}")
    database.initialize()
    pdf_path = tmp_path / "sample.pdf"
    copyfile(SAMPLE_PATH, pdf_path)
    pipeline = PdfIngestPipeline(database)
    pipeline.ingest(pdf_path)
    return database


def test_researcher_returns_cited_hits(tmp_path):
    database = _ingest(tmp_path)
    researcher = Researcher(database)

    output = researcher.search("What does the summary page emphasize?", top_k=2)
    assert output.hits
    assert any("ingestion" in hit.content.lower() for hit in output.hits)
    assert all("【" in hit.citation for hit in output.hits)

    expert = Expert()
    answer = expert.compose_answer("What does it say?", output.hits)
    assert "【" in answer and "】" in answer


def test_expert_requires_citations():
    expert = Expert()
    hit = ResearchHit(
        document_id="doc",
        section_id="sec",
        content="Statement without cite",
        score=1.0,
        citation="",
        start_page=0,
        end_page=0,
        start_line=1,
        end_line=1,
    )
    try:
        expert.compose_answer("question", [hit])
    except CitationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected CitationError")
