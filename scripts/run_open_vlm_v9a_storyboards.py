#!/usr/bin/env python3
"""Run label-blind V9A over frozen dense temporal storyboards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from run_open_vlm_scene_benchmark import SYSTEM_V9A, parse_json
except ModuleNotFoundError:  # Imported as scripts.run_open_vlm_v9a_storyboards.
    from scripts.run_open_vlm_scene_benchmark import SYSTEM_V9A, parse_json


STORYBOARD_INSTRUCTION = """The image is a dense temporal storyboard from one
saved video clip. Frames are ordered left-to-right and then top-to-bottom, with
timestamps printed in the cells. Treat adjacent cells as time, not as unrelated
simultaneous images. Record only a literal event that is visible across these
frames. You have no title, transcript, category, norm, polarity, or explanation.
Do not infer them."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def successful_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        (row["item_id"], row["model"])
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def request_one(
    endpoint: str,
    model: str,
    row: dict,
    timeout: int,
    retries: int,
) -> dict:
    sheet = Path(row["sheet_path"])
    if not sheet.is_absolute():
        sheet = Path(row["_manifest_dir"]) / sheet
    if sha256(sheet) != row["sheet_sha256"]:
        raise ValueError(f"storyboard hash mismatch: {row['item_id']}")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": SYSTEM_V9A},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": sheet.resolve().as_uri()},
                    },
                    {"type": "text", "text": STORYBOARD_INSTRUCTION},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload).encode()
    last_error = ""
    last_content = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("empty response")
            last_content = content
            parsed = parse_json(content, "v9a")
            return {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "rubric": "v9a_storyboard",
                "model": model,
                "sheet_sha256": row["sheet_sha256"],
                "frame_count": row["frame_count"],
                "result": parsed,
                "raw_response": content,
                "usage": body.get("usage"),
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:2000]
            except Exception:
                detail = ""
            last_error = f"HTTPError {exc.code}: {detail}"
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2**attempt)
    return {
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": row["pillar"],
        "rubric": "v9a_storyboard",
        "model": model,
        "sheet_sha256": row["sheet_sha256"],
        "frame_count": row["frame_count"],
        "result": None,
        "raw_response": last_content,
        "usage": None,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8271/v1")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    for row in rows:
        row["_manifest_dir"] = str(args.manifest.parent)
    completed = successful_keys(load_jsonl(args.out)) if args.out.exists() else set()
    pending = [
        row for row in rows if (row["item_id"], args.model) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = {
            pool.submit(
                request_one,
                args.endpoint,
                args.model,
                row,
                args.timeout,
                args.retries,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {row['item_id']} "
                f"{'ok' if result['error'] is None else result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
