"""Command line interface for pdfqanda."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .ingest import build_document_artifact
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

    console.print("[bold green]Writing document artifacts...[/bold green]")
    for pdf in pdfs:
        doc_hash, artifact_path, _ = build_document_artifact(pdf)
        console.print(
            f" • [bold]{doc_hash}[/bold] → [italic]{artifact_path}[/italic]",
        )


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


@app.command()
def peek(
    pdf_path: Path = typer.Argument(..., help="Path to a PDF file."),
    page: Optional[int] = typer.Option(None, "--page", "-p", help="Page index to inspect."),
) -> None:
    """Quickly inspect detected notes and graphics for a PDF."""

    doc_hash, artifact_path, pages = build_document_artifact(pdf_path)
    console.print(f"[bold]Document hash:[/bold] {doc_hash}")
    console.print(f"[bold]Artifact:[/bold] {artifact_path}")

    selected_pages = pages
    if page is not None:
        selected_pages = [p for p in pages if p.index == page]
        if not selected_pages:
            console.print(f"[bold red]Page {page} not found in document.[/bold red]")
            raise typer.Exit(code=1)

    for page_obj in selected_pages:
        console.print(f"\n[bold underline]Page {page_obj.index}[/bold underline]")

        if page_obj.notes:
            console.print("[bold cyan]Notes:[/bold cyan]")
            for note in page_obj.notes:
                bbox = ", ".join(f"{value:.2f}" for value in note.bbox.to_list())
                ref_info = f" ref={note.ref}" if note.ref else ""
                console.print(f" • ({note.kind}{ref_info}) {note.text} [bbox={bbox}]")
        else:
            console.print("[dim]No notes detected.[/dim]")

        if page_obj.graphics:
            console.print("[bold magenta]Graphics:[/bold magenta]")
            for graphic in page_obj.graphics:
                bbox = ", ".join(f"{value:.2f}" for value in graphic.bbox.to_list())
                sha_preview = graphic.sha256[:12]
                nearby = f" – {graphic.nearby_text}" if graphic.nearby_text else ""
                console.print(
                    f" • {graphic.path} [sha={sha_preview}] [bbox={bbox}]{nearby}",
                )
        else:
            console.print("[dim]No graphics detected.[/dim]")


if __name__ == "__main__":  # pragma: no cover
    app()
