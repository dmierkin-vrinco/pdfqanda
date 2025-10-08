from __future__ import annotations

from pathlib import Path

import pytest

from pdfqanda.qa import PdfQaEngine

PDF_PATH = Path(__file__).resolve().parent.parent / "input" / "pdfs" / "FedEx_Standard_List_Rates_2025.pdf"

QUESTIONS_AND_ANSWERS = [
    ("Standard Overnight, Zone 2, 1 lb", "$38.55"),
    ("Standard Overnight, Zone 2, 3 lb", "$44.68"),
    ("Standard Overnight, Zone 2, 10 lb", "$58.48"),
    ("Standard Overnight, Zone 2, 20 lb", "$81.68"),
    ("2Day, Zone 2, 1 lb", "$25.23"),
    ("2Day, Zone 2, 5 lb", "$33.64"),
    ("2Day, Zone 2, 10 lb", "$41.74"),
    ("Express Saver, Zone 2, 1 lb", "$22.95"),
    ("Express Saver, Zone 2, 3 lb", "$24.74"),
    ("Express Saver, Zone 2, 5 lb", "$25.70"),
    ("Standard Overnight, Zone 3, 1 lb", "$41.08"),
    ("Standard Overnight, Zone 3, 3 lb", "$48.01"),
    ("Standard Overnight, Zone 3, 10 lb", "$64.53"),
    ("Standard Overnight, Zone 3, 20 lb", "$89.67"),
    ("2Day, Zone 3, 1 lb", "$28.26"),
    ("2Day, Zone 3, 5 lb", "$36.92"),
    ("2Day, Zone 3, 10 lb", "$45.66"),
    ("Express Saver, Zone 3, 1 lb", "$24.39"),
    ("Express Saver, Zone 3, 3 lb", "$26.67"),
    ("Express Saver, Zone 3, 5 lb", "$28.48"),
    ("Standard Overnight, Zone 4, 1 lb", "$45.55"),
    ("Standard Overnight, Zone 4, 3 lb", "$53.60"),
    ("Standard Overnight, Zone 4, 10 lb", "$72.64"),
    ("Standard Overnight, Zone 4, 20 lb", "$101.19"),
    ("2Day, Zone 4, 1 lb", "$31.69"),
    ("2Day, Zone 4, 5 lb", "$41.86"),
    ("2Day, Zone 4, 10 lb", "$51.48"),
    ("Express Saver, Zone 4, 1 lb", "$27.03"),
    ("Express Saver, Zone 4, 3 lb", "$29.88"),
    ("Express Saver, Zone 4, 5 lb", "$32.28"),
    ("Standard Overnight, Zone 5, 1 lb", "$53.51"),
    ("Standard Overnight, Zone 5, 3 lb", "$96.87"),
    ("Standard Overnight, Zone 5, 10 lb", "$153.32"),
    ("Standard Overnight, Zone 5, 20 lb", "$235.76"),
    ("2Day, Zone 5, 1 lb", "$33.31"),
    ("2Day, Zone 5, 5 lb", "$51.70"),
    ("2Day, Zone 5, 10 lb", "$72.25"),
    ("Express Saver, Zone 5, 1 lb", "$29.88"),
    ("Express Saver, Zone 5, 3 lb", "$33.73"),
    ("Express Saver, Zone 5, 5 lb", "$39.46"),
    ("Standard Overnight, Zone 6, 1 lb", "$55.02"),
    ("Standard Overnight, Zone 6, 3 lb", "$101.44"),
    ("Standard Overnight, Zone 6, 10 lb", "$163.08"),
    ("Standard Overnight, Zone 6, 20 lb", "$252.98"),
    ("2Day, Zone 6, 1 lb", "$35.38"),
    ("2Day, Zone 6, 5 lb", "$54.76"),
    ("2Day, Zone 6, 10 lb", "$76.61"),
    ("Express Saver, Zone 6, 1 lb", "$31.74"),
    ("Express Saver, Zone 6, 3 lb", "$35.63"),
    ("Express Saver, Zone 6, 5 lb", "$41.71"),
    ("Standard Overnight, Zone 7, 1 lb", "$57.29"),
    ("Standard Overnight, Zone 7, 3 lb", "$104.26"),
    ("Standard Overnight, Zone 7, 10 lb", "$170.49"),
    ("Standard Overnight, Zone 7, 20 lb", "$267.63"),
    ("2Day, Zone 7, 1 lb", "$36.72"),
    ("2Day, Zone 7, 5 lb", "$57.48"),
    ("2Day, Zone 7, 10 lb", "$80.37"),
    ("Express Saver, Zone 7, 1 lb", "$33.30"),
    ("Express Saver, Zone 7, 3 lb", "$37.50"),
    ("Express Saver, Zone 7, 5 lb", "$43.72"),
    ("Standard Overnight, Zone 8, 1 lb", "$61.52"),
    ("Standard Overnight, Zone 8, 3 lb", "$112.26"),
    ("Standard Overnight, Zone 8, 10 lb", "$183.43"),
    ("Standard Overnight, Zone 8, 20 lb", "$288.86"),
    ("2Day, Zone 8, 1 lb", "$38.95"),
    ("2Day, Zone 8, 5 lb", "$61.29"),
    ("2Day, Zone 8, 10 lb", "$87.31"),
    ("Express Saver, Zone 8, 1 lb", "$35.49"),
    ("Express Saver, Zone 8, 3 lb", "$39.80"),
    ("Express Saver, Zone 8, 5 lb", "$46.24"),
    ("Ground, Zone 2, 1 lb", "$11.32"),
    ("Ground, Zone 2, 10 lb", "$14.98"),
    ("Ground, Zone 3, 1 lb", "$11.65"),
    ("Ground, Zone 3, 10 lb", "$16.09"),
    ("Ground, Zone 4, 1 lb", "$12.69"),
    ("Ground, Zone 4, 10 lb", "$17.59"),
    ("Ground, Zone 5, 1 lb", "$13.27"),
    ("Ground, Zone 5, 10 lb", "$19.39"),
    ("Ground, Zone 6, 1 lb", "$13.72"),
    ("Ground, Zone 6, 10 lb", "$20.00"),
    ("Ground, Zone 7, 1 lb", "$13.93"),
    ("Ground, Zone 7, 10 lb", "$22.54"),
    ("Ground, Zone 8, 1 lb", "$14.17"),
    ("Ground, Zone 8, 10 lb", "$24.91"),
    ("Home Delivery, Zone 2, 5 lb", "$19.24"),
    ("Home Delivery, Zone 3, 5 lb", "$20.71"),
    ("Home Delivery, Zone 4, 5 lb", "$21.73"),
    ("Home Delivery, Zone 5, 5 lb", "$23.51"),
    ("Home Delivery, Zone 6, 5 lb", "$24.21"),
    ("Home Delivery, Zone 7, 5 lb", "$25.28"),
    ("Home Delivery, Zone 8, 5 lb", "$26.43"),
    ("Express Saver, Zone 2, 10 lb", "$33.37"),
    ("Express Saver, Zone 3, 10 lb", "$36.66"),
    ("Express Saver, Zone 4, 10 lb", "$41.78"),
    ("Express Saver, Zone 5, 10 lb", "$56.99"),
    ("Express Saver, Zone 6, 10 lb", "$60.54"),
    ("Express Saver, Zone 7, 10 lb", "$66.38"),
    ("Express Saver, Zone 8, 10 lb", "$72.17"),
    ("2Day A.M., Zone 5, 10 lb", "$93.04"),
    ("Priority Overnight, Zone 3, 5 lb", "$71.87"),
]


@pytest.fixture(scope="module")
def fedex_engine() -> PdfQaEngine:
    engine = PdfQaEngine()
    engine.build_index([PDF_PATH])
    return engine


@pytest.mark.parametrize(("question", "expected_answer"), QUESTIONS_AND_ANSWERS)
def test_fedex_rates_answers(fedex_engine: PdfQaEngine, question: str, expected_answer: str) -> None:
    answers = fedex_engine.query(question)
    assert answers, "No answers returned from the engine"
    assert any(
        expected_answer in answer.text
        for answer in answers
    ), f"Expected {expected_answer!r} in answers for {question!r}"
