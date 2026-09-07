"""Build the SQLite source snapshot used by the metric-reconciliation report."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


OUT = Path(__file__).resolve().parent


def main() -> None:
    with (OUT / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0])
    database = OUT / "report_snapshot.sqlite"
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE comparison (" + ", ".join(f'"{column}" TEXT' for column in columns) + ")"
        )
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO comparison VALUES ({placeholders})",
            [[row[column] for column in columns] for row in rows],
        )


if __name__ == "__main__":
    main()
