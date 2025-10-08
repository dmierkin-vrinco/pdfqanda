from pathlib import Path

from pdfqanda.ingest import extract_text_from_pdf

SAMPLE_PDF = b"""%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources<< /Font<< /F1 5 0 R>>>>>>endobj\n4 0 obj<< /Length 67>>stream\nBT\n/F1 24 Tf\n72 120 Td\n(Hello PDF Document) Tj\nET\nendstream\nendobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000113 00000 n\n0000000230 00000 n\n0000000310 00000 n\ntrailer<< /Root 1 0 R /Size 6>>\nstartxref\n368\n%%EOF\n"""


def test_extract_text_from_pdf(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(SAMPLE_PDF)

    extracted = extract_text_from_pdf(pdf_path)

    assert "Hello" in extracted
    assert "Document" in extracted
