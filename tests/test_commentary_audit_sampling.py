from scripts.export_commentary_audit_batch import select_balanced
from scripts.visual_audit_ledger import connect


def add_commentary_item(conn, item_id, uid, index):
    conn.execute(
        """
        INSERT INTO items(
            item_id,pillar,uid,item_index,metadata_path,clip_path,has_clip,
            category,polarity,present,source_mtime,discovered_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            item_id, "commentary", uid, index, f"{uid}.json", None, 0,
            "comm_aita", "judgment", 1, 1.0, 1.0, 1.0,
        ),
    )


def test_commentary_sampler_excludes_source_siblings_after_any_batch_assignment(tmp_path):
    conn = connect(tmp_path / "audit.db")
    add_commentary_item(conn, "commentary:source-a:0", "source-a", 0)
    add_commentary_item(conn, "commentary:source-a:1", "source-a", 1)
    add_commentary_item(conn, "commentary:source-b:0", "source-b", 0)
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "prior-commentary", "commentary", "commentary_v2", "manual",
            "prior sample", "seed", "manifest.json", 1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO batch_items(
            batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
        ) VALUES (?,?,?,?,?)
        """,
        ("prior-commentary", "commentary:source-a:0", 0, 0, "content"),
    )
    selected = select_balanced(
        conn,
        n=10,
        rubric="commentary_v2",
        model="manual",
        seed="commentary-source-disjoint",
        query_sources=[],
        categories=[],
        uids=[],
        item_ids=[],
    )
    assert [row["item_id"] for row in selected] == ["commentary:source-b:0"]
    conn.close()
