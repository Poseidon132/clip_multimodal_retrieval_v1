"""Dependency-light utilities for the release dashboard and result exports."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping

SCORE_FIELDS = (
    "t2i_R@1", "t2i_R@5", "t2i_R@10",
    "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_R@1",
)


def read_report_rows(path: str | Path) -> list[dict[str, object]]:
    """Read a comparable experiment table, preserving missing data as None."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in SCORE_FIELDS:
            value = row.get(field, "")
            row[field] = float(value) if value not in (None, "") else None
    return rows


def excel_safe_cell(value: object) -> object:
    """Prevent spreadsheet formula injection in human-readable exports."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def rows_to_csv_bytes(rows: Iterable[Mapping[str, object]], fieldnames: list[str]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: excel_safe_cell(row.get(field, "")) for field in fieldnames})
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def format_score(score: float, reranked: bool) -> str:
    """BLIP ITM ranking confidence is not calibrated model accuracy."""
    return f"{score:.3f}" if reranked else f"{score:.4f}"
