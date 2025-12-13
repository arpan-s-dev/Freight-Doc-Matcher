"""Export the analytical views to CSV / Parquet for Tableau and Power BI.

Both tools connect directly to Parquet (and CSV); exporting the marts gives a
license-free, BI-tool-ready star of files derived entirely from real output.
"""

from pathlib import Path

from matcher.analytics.load import DEFAULT_DB
from matcher.analytics.queries import QUERIES


def export_views(db_path: str = DEFAULT_DB, out_dir: str = "output/exports",
                 fmt: str = "parquet") -> list[Path]:
    """Write every named view to ``out_dir`` in ``fmt`` (parquet|csv). Returns paths."""
    import duckdb

    if fmt not in ("parquet", "csv"):
        raise ValueError("fmt must be 'parquet' or 'csv'")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path, read_only=True)
    written: list[Path] = []
    try:
        for name, sql in QUERIES.items():
            path = out / f"{name}.{fmt}"
            copy_opts = "(FORMAT PARQUET)" if fmt == "parquet" else "(FORMAT CSV, HEADER)"
            con.execute(f"COPY ({sql}) TO '{path.as_posix()}' {copy_opts}")
            written.append(path)
        # Also export the base table for free-form analysis in the BI tool.
        base = out / f"loads.{fmt}"
        copy_opts = "(FORMAT PARQUET)" if fmt == "parquet" else "(FORMAT CSV, HEADER)"
        con.execute(f"COPY loads TO '{base.as_posix()}' {copy_opts}")
        written.append(base)
    finally:
        con.close()
    return written
