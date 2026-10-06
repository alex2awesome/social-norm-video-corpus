from scripts.export_visual_audit_batch import select_balanced
from scripts.export_instructional_audio_repair_audit import resolve_source_clip
from scripts.visual_audit_ledger import connect


def add_item(conn, item_id, index, *, uid=None, pillar="instructional"):
    conn.execute(
        """
        INSERT INTO items(
            item_id,pillar,uid,item_index,metadata_path,clip_path,has_clip,
            category,polarity,present,source_mtime,discovered_at,updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            item_id,
            pillar,
            uid or f"uid-{index}",
            index,
            f"metadata-{index}.json",
            f"clip-{index}.mp4",
            1,
            "instr_family",
            "violation",
            1,
            1.0,
            1.0,
            1.0,
        ),
    )


def test_fresh_sampler_excludes_prior_judgment_from_other_rubric_and_model(tmp_path):
    conn = connect(tmp_path / "audit.db")
    add_item(conn, "instructional:old:0", 0)
    add_item(conn, "instructional:fresh:0", 1)
    add_item(conn, "instructional:queued:0", 2)
    conn.execute(
        """
        INSERT INTO judgments(
            item_id,rubric_version,model,prompt_sha256,frame_manifest_sha256,
            pass_index,is_social_norm,visual_demo_present,medium,
            clip_composition,norm_supported,polarity_supported,
            explanation_supports_norm,dialogue_grounded,
            visual_norm_supported_without_audio,narration_leak,
            label_leak_visible,off_topic,decision,required_repairs_json,
            normalized_behavior,normalized_norm,rejection_reasons_json,
            evidence_frames_json,description,raw_response_json,audited_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "instructional:old:0",
            "instructional_v3",
            "older-manual-model",
            "prompt",
            "frames",
            0,
            "yes",
            "no",
            "talking_head",
            "talking_head_only",
            "yes",
            "uncertain",
            "yes",
            "no",
            "no",
            "yes",
            "no",
            "no",
            "reject",
            "[]",
            "",
            "",
            "[]",
            "[]",
            "prior review",
            "{}",
            1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "prior-batch",
            "instructional",
            "instructional_v4",
            "some-model",
            "prior sample",
            "seed",
            "manifest.json",
            1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO batch_items(
            batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
        ) VALUES (?,?,?,?,?)
        """,
        ("prior-batch", "instructional:queued:0", 0, 3, "queued-frames"),
    )
    selected = select_balanced(
        conn,
        pillar="instructional",
        n=10,
        rubric="instructional_v4",
        model="new-model",
        seed="seed",
        polarities=["violation"],
        categories=[],
        query_regex="",
        query_sources=[],
        uids=[],
        item_ids=[],
    )
    assert [row["item_id"] for row in selected] == ["instructional:fresh:0"]
    conn.close()


def test_explicit_item_list_can_escalate_a_prior_batch_assignment(tmp_path):
    conn = connect(tmp_path / "audit.db")
    add_item(conn, "instructional:queued:0", 0)
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "lightweight-batch", "instructional", "instructional_lightweight",
            "manual", "frozen hourly sample", "seed", "manifest.json", 1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO batch_items(
            batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
        ) VALUES (?,?,?,?,?)
        """,
        ("lightweight-batch", "instructional:queued:0", 0, 3, "frames"),
    )
    selected = select_balanced(
        conn,
        pillar="instructional",
        n=10,
        rubric="instructional_v4",
        model="manual",
        seed="dense-escalation",
        polarities=[],
        categories=[],
        query_regex="",
        query_sources=[],
        uids=[],
        item_ids=["instructional:queued:0"],
    )
    assert [row["item_id"] for row in selected] == ["instructional:queued:0"]
    conn.close()


def test_fresh_sampler_excludes_every_item_from_a_previously_assigned_source(tmp_path):
    conn = connect(tmp_path / "audit.db")
    add_item(conn, "instructional:source-a:0", 0, uid="source-a")
    add_item(conn, "instructional:source-a:1", 1, uid="source-a")
    add_item(conn, "instructional:source-b:0", 0, uid="source-b")
    conn.execute(
        """
        INSERT INTO batches(
            batch_id,pillar,rubric_version,intended_model,selection_strategy,
            seed,manifest_path,created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            "prior-source-batch", "instructional", "instructional_v4", "manual",
            "prior sample", "seed", "manifest.json", 1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO batch_items(
            batch_id,item_id,ordinal,frame_count,frame_manifest_sha256
        ) VALUES (?,?,?,?,?)
        """,
        ("prior-source-batch", "instructional:source-a:0", 0, 3, "frames"),
    )
    selected = select_balanced(
        conn,
        pillar="instructional",
        n=10,
        rubric="instructional_v4",
        model="manual",
        seed="source-disjoint",
        polarities=[],
        categories=[],
        query_regex="",
        query_sources=[],
        uids=[],
        item_ids=[],
    )
    assert [row["item_id"] for row in selected] == ["instructional:source-b:0"]
    conn.close()


def test_witnessed_sampler_uses_witnessed_judgments_and_excludes_source_siblings(tmp_path):
    conn = connect(tmp_path / "audit.db")
    add_item(conn, "witnessed:source-a:0", 0, uid="source-a", pillar="witnessed")
    add_item(conn, "witnessed:source-a:1", 1, uid="source-a", pillar="witnessed")
    add_item(conn, "witnessed:source-b:0", 0, uid="source-b", pillar="witnessed")
    conn.execute(
        """
        INSERT INTO witnessed_judgments(
            item_id,rubric_version,model,prompt_sha256,frame_manifest_sha256,
            pass_index,is_social_norm,social_action_visible,reaction_visible,
            action_before_reaction,reaction_targets_action,reaction_is_normative,
            behavior_label_supported,clean_pre_reaction_demo,authenticity,decision,
            rejection_reasons_json,action_evidence_frames_json,
            reaction_evidence_frames_json,description,raw_response_json,audited_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "witnessed:source-a:0", "witnessed_v1", "manual", "prompt", "frames",
            0, "yes", "no", "no", "uncertain", "uncertain", "no", "no", "no",
            "uncertain", "reject", "[]", "[]", "[]", "prior review", "{}", 1.0,
        ),
    )
    selected = select_balanced(
        conn,
        pillar="witnessed",
        n=10,
        rubric="witnessed_v1",
        model="manual",
        seed="witnessed-source-disjoint",
        polarities=[],
        categories=[],
        query_regex="",
        query_sources=[],
        uids=[],
        item_ids=[],
    )
    assert [row["item_id"] for row in selected] == ["witnessed:source-b:0"]
    conn.close()


def test_audio_repair_can_chain_from_exact_repaired_clip(tmp_path):
    manifest = tmp_path / "batch" / "manifest.json"
    item = {
        "clip_path": "data/instructional/source.mp4",
        "repaired_clip_path": "repaired_clips/exact.mp4",
    }
    assert resolve_source_clip(manifest, item, "repaired_clip_path", tmp_path) == (
        tmp_path / "batch" / "repaired_clips" / "exact.mp4"
    )
    assert resolve_source_clip(manifest, item, "clip_path", tmp_path) == (
        tmp_path / "data" / "instructional" / "source.mp4"
    )
