#!/usr/bin/env python3
"""Report search-source/category yield from the latest manual audit per item."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def grouped(
    conn: sqlite3.Connection,
    table: str,
    rubric: str,
    pillar: str,
    dimension: str,
    usable: str,
):
    rows = conn.execute(
        f"""
        WITH ranked AS (
          SELECT j.*,ROW_NUMBER() OVER (
            PARTITION BY j.item_id
            ORDER BY j.pass_index DESC,j.audited_at DESC,j.judgment_id DESC
          ) rank
          FROM {table} j WHERE j.rubric_version=?
        )
        SELECT COALESCE(i.{dimension},'NULL') value,COUNT(*) reviewed,
               SUM(CASE WHEN r.decision IN ({usable}) THEN 1 ELSE 0 END) usable
        FROM ranked r JOIN items i ON i.item_id=r.item_id
        WHERE r.rank=1 AND i.pillar=?
        GROUP BY COALESCE(i.{dimension},'NULL')
        ORDER BY reviewed DESC,value
        """,
        (rubric, pillar),
    ).fetchall()
    return [
        {
            "value": row[0],
            "reviewed": row[1],
            "usable": row[2],
            "usable_rate": row[2] / row[1],
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    report = {
        "basis": "latest manual judgment per item; source decisions only",
        "instructional": {
            "by_query_source": grouped(
                conn, "judgments", "instructional_v4", "instructional", "query_source",
                "'accept','accept_with_repairs'",
            ),
            "by_category": grouped(
                conn, "judgments", "instructional_v4", "instructional", "category",
                "'accept','accept_with_repairs'",
            ),
        },
        "witnessed": {
            "by_query_source": grouped(
                conn, "witnessed_judgments", "witnessed_v1", "witnessed", "query_source",
                "'accept_after_splice'",
            ),
            "by_category": grouped(
                conn, "witnessed_judgments", "witnessed_v1", "witnessed", "category",
                "'accept_after_splice'",
            ),
        },
        "commentary": {
            "by_query_source": grouped(
                conn, "commentary_judgments", "commentary_v2", "commentary", "query_source",
                "'accept','accept_after_relabel'",
            ),
            "by_category": grouped(
                conn, "commentary_judgments", "commentary_v2", "commentary", "category",
                "'accept','accept_after_relabel'",
            ),
        },
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({pillar: sum(x["reviewed"] for x in values["by_query_source"])
                      for pillar, values in report.items() if isinstance(values, dict)}))
    conn.close()


if __name__ == "__main__":
    main()
