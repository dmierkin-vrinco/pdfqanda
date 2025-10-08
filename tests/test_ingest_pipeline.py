from __future__ import annotations

import json
from pdfqanda.db import Database
from pdfqanda.ingest import PdfIngestPipeline

SAMPLE_PDF = b"""%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R>>>>>>endobj\n4 0 obj<< /Length 67>>stream\nBT\n/F1 24 Tf\n72 120 Td\n(Hello PDF Document) Tj\nET\nendstream\nendobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000113 00000 n\n0000000230 00000 n\n0000000310 00000 n\ntrailer<< /Root 1 0 R /Size 6>>\nstartxref\n368\n%%EOF\n"""


def test_ingest_populates_markdowns(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.sqlite"
    monkeypatch.setenv("PDFQANDA_DATABASE_URL", f"sqlite:///{db_path}")

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(SAMPLE_PDF)

    database = Database(f"sqlite:///{db_path}")
    database.initialize()

    pipeline = PdfIngestPipeline(database)
    artifacts = pipeline.ingest(pdf_path)

    assert artifacts.artifact_path.exists()

    rows = database.fetch_markdowns()
    assert rows, "expected markdown rows in kb_markdowns"

    embedding = json.loads(rows[0]["emb"])
    assert len(embedding) == 3072
    assert any(value != 0 for value in embedding)
