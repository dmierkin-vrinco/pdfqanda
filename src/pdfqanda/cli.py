"""Command line interface for pdfqanda."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .config import get_settings
from .ingest import PdfIngestor
from .retrieval import Retriever, format_answer
from .util.db import Database

app = typer.Typer(help="PDF Q&A pipeline backed by a SQLite spine.")
db_app = typer.Typer(help="Database management commands.")
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init() -> None:
    """Initialise database schemas and extensions."""

    settings = get_settings()
    database = Database(settings.db_path)
    database.initialize()
    typer.echo("Database initialised")


@app.command()
def ingest(
    pdfs: Annotated[list[Path], typer.Argument(..., exists=True, readable=True, allow_dash=False)],
    title: Annotated[str | None, typer.Option(None, help="Optional title override for a single PDF")],
) -> None:
    """Ingest one or more PDFs into the knowledge base."""

    settings = get_settings()
    database = Database(settings.db_path)
    database.initialize()

    ingestor = PdfIngestor(database)
    for pdf_path in pdfs:
        result = ingestor.ingest(pdf_path, title=title)
        typer.echo(
            f"Ingested {pdf_path.name} -> doc {result.document_id[:8]} (chunks: {result.chunk_count})"
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(..., help="Question to ask about the knowledge base.")],
    k: Annotated[int, typer.Option(6, help="Number of chunks to include in the answer.")] = 6,
) -> None:
    """Query the database for relevant snippets and return a cited answer."""

    settings = get_settings()
    database = Database(settings.db_path)
    retriever = Retriever(database)

    hits = retriever.search(question, k=k)
    answer = format_answer(hits)
    if not hits or "【doc:" not in answer:
        typer.echo("No cited answer available.")
        raise typer.Exit(code=1)

    typer.echo(answer)


if __name__ == "__main__":  # pragma: no cover
    app()
