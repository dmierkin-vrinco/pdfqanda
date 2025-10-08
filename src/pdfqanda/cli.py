"""Command line interface for pdfqanda."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .db import Database
from .ingest import PdfIngestor
from .retrieval import Retriever

app = typer.Typer(help="PDF Q&A pipeline backed by Postgres/pgvector.")
db_app = typer.Typer(help="Database management commands.")
app.add_typer(db_app, name="db")

console = Console()


@db_app.command("init")
def db_init() -> None:
    """Initialise database schemas and extensions."""

    settings = get_settings()
    database = Database(settings.db_dsn)
    database.initialize()
    console.print("[bold green]Database initialised[/bold green]")


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., exists=True, file_okay=True, readable=True),
    title: str | None = typer.Option(None, help="Optional title override."),
) -> None:
    """Ingest a PDF into the knowledge base."""

    settings = get_settings()
    database = Database(settings.db_dsn)
    database.initialize()

    ingestor = PdfIngestor(database)
    result = ingestor.ingest(pdf_path, title=title)
    console.print(
        f"[bold green]Ingested[/bold green] {pdf_path.name} -> doc {result.document_id[:8]}"
    )
    console.print(f"Chunks stored: {result.chunk_count}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the knowledge base."),
    k: int = typer.Option(6, help="Number of snippets to return."),
) -> None:
    """Query the database for relevant snippets."""

    settings = get_settings()
    database = Database(settings.db_dsn)
    retriever = Retriever(database)

    hits = retriever.search(question, k=k)
    if not hits:
        console.print("[yellow]No matches found.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Score")
    table.add_column("Snippet")
    table.add_column("Citation")
    for hit in hits:
        table.add_row(f"{hit.score:.3f}", hit.content.strip()[:200], hit.citation)
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
