import json
import logging
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from matcher import __version__

load_dotenv()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool):
    """Freight Doc Matcher — pair BOLs with Rate Confirmations."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@cli.command()
def hello():
    """Smoke test that the package is installed."""
    click.echo("Freight Doc Matcher ready.")


@cli.command()
def version():
    """Print the current version."""
    click.echo(f"freight-doc-matcher {__version__}")


def run_matcher(docs, strategy: str, model_path: str = "models/crossencoder"):
    """Match documents with the chosen strategy.

    ``ml`` uses the fine-tuned bi-encoder + cross-encoder pipeline; it falls back to
    the heuristic scorer (with a warning) if torch or a trained model is missing.
    """
    from matcher.matcher import match_documents

    if strategy == "heuristic":
        return match_documents(docs)

    if strategy == "ml":
        try:
            from matcher.linkage.match import build_ml_matcher, match_documents_ml
            scorer, candidate_fn = build_ml_matcher(docs, model_path=model_path)
            return match_documents_ml(docs, scorer, candidate_fn)
        except Exception as e:
            click.echo(f"  [!] ML matcher unavailable ({e}); falling back to heuristic.", err=True)
            return match_documents(docs)

    raise click.BadParameter(f"unknown matcher strategy: {strategy}")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", default="output", type=click.Path(path_type=Path), show_default=True)
@click.option("--no-claude", is_flag=True, help="Disable Claude API fallback.")
@click.option("--dry-run", is_flag=True, help="Process without writing files or spreadsheet.")
@click.option("--matcher", "matcher_strategy", type=click.Choice(["heuristic", "ml"]),
              default="heuristic", show_default=True, help="Matching strategy.")
def process(input_dir: Path, output: Path, no_claude: bool, dry_run: bool, matcher_strategy: str):
    """Process a folder of BOL and Rate Confirmation PDFs."""
    from matcher.pipeline import process_folder
    from matcher.organize import organize_files
    from matcher.spreadsheet import write_spreadsheet

    start = time.perf_counter()
    use_claude = not no_claude

    click.echo(f"Scanning {input_dir} ...")
    docs = process_folder(input_dir, use_claude=use_claude)
    matches, unmatched = run_matcher(docs, matcher_strategy)

    native = sum(1 for d in docs if d.source_type.value == "native_pdf")
    scanned = sum(1 for d in docs if d.source_type.value == "clean_scan")
    photos = sum(1 for d in docs if d.source_type.value == "phone_photo")
    regex_n = sum(1 for d in docs if d.extraction_method in ("native", "tesseract"))
    claude_n = sum(1 for d in docs if d.extraction_method == "claude")
    auto = sum(1 for m in matches if m.match_type != "manual_review")
    review = sum(1 for m in matches if m.match_type == "manual_review")

    api_cost = 0.0
    costs_file = Path("costs.jsonl")
    if costs_file.exists():
        try:
            api_cost = sum(json.loads(l)["usd"] for l in costs_file.read_text().splitlines() if l.strip())
        except Exception:
            pass

    runtime = time.perf_counter() - start
    xlsx_path = output / "loads.xlsx"

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        path_map = organize_files(docs, matches, output)
        write_spreadsheet(matches, unmatched, path_map, xlsx_path,
                          runtime_seconds=runtime, api_cost_usd=api_cost)
    else:
        click.echo("(dry run — no files written)")
        path_map = {}

    click.echo(f"\nProcessed: {len(docs)} documents ({runtime:.1f}s)")
    click.echo(f"  - Native PDFs:    {native}")
    click.echo(f"  - Scanned:        {scanned}")
    click.echo(f"  - Phone photos:   {photos}")
    click.echo("Extraction methods:")
    click.echo(f"  - Regex/native:   {regex_n}")
    click.echo(f"  - Claude fallback:{claude_n}")
    click.echo(f"Matches: {auto} paired, {review} needs review, {len(unmatched)} unmatched")
    click.echo(f"API cost: ${api_cost:.4f}")
    if not dry_run:
        click.echo(f"Output: {xlsx_path}")


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def classify(file: Path):
    """Show the source type classification for a single file."""
    from matcher.classify import classify_pdf
    result = classify_pdf(file)
    click.echo(f"{file.name}: {result.value}")


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--no-claude", is_flag=True)
def extract(file: Path, no_claude: bool):
    """Show extracted fields for a single file (debugging)."""
    from matcher.pipeline import process_document
    doc = process_document(file, use_claude=not no_claude)
    data = doc.model_dump(mode="json", exclude={"raw_text"})
    data["source_path"] = str(data["source_path"])
    click.echo(json.dumps(data, indent=2, default=str))


@cli.group()
def analytics():
    """SQL analytics over matched-load output (requires the [bi] extra)."""


@analytics.command("build")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db", default="output/analytics.duckdb", show_default=True)
@click.option("--no-claude", is_flag=True, help="Disable Claude API fallback.")
@click.option("--matcher", "matcher_strategy", type=click.Choice(["heuristic", "ml"]),
              default="heuristic", show_default=True)
def analytics_build(input_dir: Path, db: str, no_claude: bool, matcher_strategy: str):
    """Process a folder and load matched results into the DuckDB warehouse."""
    from matcher.pipeline import process_folder
    from matcher.analytics.load import load_to_duckdb
    from matcher.analytics.metrics import headline_kpis

    click.echo(f"Scanning {input_dir} ...")
    docs = process_folder(input_dir, use_claude=not no_claude)
    matches, unmatched = run_matcher(docs, matcher_strategy)
    load_to_duckdb(matches, unmatched, db).close()

    kpi = headline_kpis(db)
    click.echo(f"Loaded {kpi['documents']} rows into {db}")
    click.echo(f"  matched: {kpi['matched']}  match rate: {kpi['match_rate_pct']}%  "
               f"lanes: {kpi['lanes']}  revenue: ${kpi['total_revenue']:,.2f}")


@analytics.command("query")
@click.argument("name")
@click.option("--db", default="output/analytics.duckdb", show_default=True)
def analytics_query(name: str, db: str):
    """Print a named analytical view (lane_summary, broker_scorecard, ...)."""
    from matcher.analytics.metrics import run_query
    df = run_query(name, db)
    click.echo(df.to_string(index=False))


@analytics.command("export")
@click.option("--db", default="output/analytics.duckdb", show_default=True)
@click.option("--format", "fmt", type=click.Choice(["parquet", "csv"]), default="parquet",
              show_default=True)
@click.option("--out", default="output/exports", show_default=True)
def analytics_export(db: str, fmt: str, out: str):
    """Export views to Parquet/CSV for Tableau / Power BI."""
    from matcher.analytics.export import export_views
    paths = export_views(db, out, fmt)
    click.echo(f"Wrote {len(paths)} files to {out}/:")
    for p in paths:
        click.echo(f"  {p.name}")
