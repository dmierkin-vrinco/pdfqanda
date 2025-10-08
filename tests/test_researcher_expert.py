from __future__ import annotations

from pdfqanda.db import Database
from pdfqanda.expert import CitationError, Expert
from pdfqanda.ingest import PdfIngestPipeline
from pdfqanda.models import ResearchHit
from pdfqanda.researcher import Researcher

SAMPLE_PDF = b"""%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R>>>>>>endobj\n4 0 obj<< /Length 67>>stream\nBT\n/F1 24 Tf\n72 120 Td\n(Hello PDF Document) Tj\nET\nendstream\nendobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000113 00000 n\n0000000230 00000 n\n0000000310 00000 n\ntrailer<< /Root 1 0 R /Size 6>>\nstartxref\n368\n%%EOF\n"""


def _ingest(tmp_path):
    db_path = tmp_path / "kb.sqlite"
    database = Database(f"sqlite:///{db_path}")
    database.initialize()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(SAMPLE_PDF)
    pipeline = PdfIngestPipeline(database)
    pipeline.ingest(pdf_path)
    return database


def test_researcher_returns_cited_hits(tmp_path):
    database = _ingest(tmp_path)
    researcher = Researcher(database)

    output = researcher.search("What does the sample pdf say?", top_k=2)
    assert output.hits
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
