"""Command line interface for pdfqanda."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import get_settings
from .db import Database
from .expert import CitationError, Expert
from .ingest import PdfIngestPipeline
from .researcher import Researcher

app = typer.Typer(help="Hybrid retrieval Q&A over PDFs with hard citations.")
console = Console()


def _load_database() -> Database:
    settings = get_settings()
    database = Database(settings.database_url)
    database.initialize()
    return database


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., help="Path to a PDF file to ingest."),
    title: Optional[str] = typer.Option(None, help="Optional title override for the document."),
) -> None:
    """Ingest a PDF into the knowledge base."""

    database = _load_database()
    pipeline = PdfIngestPipeline(database)
    artifacts = pipeline.ingest(pdf_path, title=title)
    console.print(
        f"[bold green]Ingested[/bold green] {pdf_path.name}"
        f" [dim](sha={artifacts.doc_hash[:12]})[/dim]"
    )
    console.print(f"Artifacts cached at [italic]{artifacts.artifact_path}[/italic]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the ingested documents."),
    top_k: int = typer.Option(6, help="Maximum number of evidence snippets to surface."),
    json_output: Optional[Path] = typer.Option(None, "--json", help="Write evidence payload to JSON."),
) -> None:
    """Ask a question using the Researcher→Expert pipeline."""

    database = _load_database()
    researcher = Researcher(database)
    expert = Expert()

    output = researcher.search(question, top_k=top_k)
    if not output.hits:
        console.print("[yellow]No evidence found for the question.[/yellow]")
        raise typer.Exit(code=1)

    try:
        answer = expert.compose_answer(question, output.hits)
    except CitationError as exc:
        console.print(f"[bold red]CITATION_CHECK_FAILED[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(answer)
    if output.sql:
        console.print(f"\n[dim]Suggested SQL:[/dim] {output.sql}")

    if json_output:
        payload = {
            "question": question,
            "answer": answer,
            "hits": [hit.__dict__ for hit in output.hits],
            "sql": output.sql,
        }
        json_output.write_text(json.dumps(payload, indent=2))
        console.print(f"Evidence exported to [italic]{json_output}[/italic]")


if __name__ == "__main__":  # pragma: no cover
    app()
