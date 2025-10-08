"""Command line interface for pdfqanda."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .qa import PdfQaEngine

app = typer.Typer(help="Build and query lightweight QA indexes for PDF documents.")
console = Console()


@app.command()
def build(
    pdfs: list[Path] = typer.Argument(..., help="One or more PDF files to index."),
    output: Path = typer.Option("index.pkl", help="Path to save the generated index."),
    chunk_size: int = typer.Option(1000, help="Chunk size in characters."),
    overlap: int = typer.Option(200, help="Overlap size between chunks."),
) -> None:
    """Build a TF-IDF index from the provided PDFs."""

    engine = PdfQaEngine(chunk_size=chunk_size, overlap=overlap)
    console.print("[bold green]Building index...[/bold green]")
    engine.build_index(pdfs)
    engine.save(output)
    console.print(f"Index saved to [bold]{output}[/bold]")


@app.command()
def ask(
    index_path: Path = typer.Option(..., "--index", "-i", help="Path to a saved index."),
    question: str = typer.Argument(..., help="Question to ask about the indexed PDFs."),
    top_k: int = typer.Option(3, help="Number of answers to return."),
    json_output: Optional[Path] = typer.Option(None, help="Optional path to export answers as JSON."),
) -> None:
    """Ask a question about the indexed PDFs."""

    engine = PdfQaEngine.load(index_path)
    engine.top_k = top_k
    answers = engine.query(question)

    if not answers:
        console.print("[bold yellow]No answers found.[/bold yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Top Answers")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Excerpt", overflow="fold")

    for i, answer in enumerate(answers, start=1):
        table.add_row(str(i), f"{answer.score:.3f}", answer.text)

    console.print(table)

    if json_output:
        payload = [answer.__dict__ for answer in answers]
        json_output.write_text(json.dumps(payload, indent=2))
        console.print(f"Answers saved to [bold]{json_output}[/bold]")


if __name__ == "__main__":  # pragma: no cover
    app()
