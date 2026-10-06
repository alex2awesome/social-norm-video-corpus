#!/usr/bin/env python3
"""Fetch Reddit post metadata + comment trees via Arctic Shift (append-only).

For every ``reddit__<postid>`` uid in the corpus, pull the post record
(title, subreddit, flair, score) in batches and the top comments per post —
third-party normative judgments ("what an entitled jerk" vs "sick fight")
that discriminate social-norm violations from generic spectacle, which the
2026-09-01 spot-check identified as the dominant witnessed failure mode.

Polite by construction: single-threaded, fixed sleep between requests,
resumable journal, and a circuit breaker that aborts after consecutive
failures rather than hammering the endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

FETCHER_VERSION = "fetch_reddit_context_v1"
API = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "norm-scraper-research/1.0 (academic corpus annotation)"

POST_BATCH = 100
MAX_COMMENTS = 60
SLEEP_SEC = 1.5
CIRCUIT_BREAKER_FAILURES = 8


def http_json(url: str, timeout: int = 30) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def compact_post(post: dict[str, Any]) -> dict[str, Any]:
    return {
        key: post.get(key)
        for key in ("id", "title", "subreddit", "score", "num_comments",
                    "link_flair_text", "over_18", "created_utc", "author",
                    "selftext", "url")
        if post.get(key) not in (None, "")
    }


def compact_comment(comment: dict[str, Any]) -> dict[str, Any]:
    body = str(comment.get("body") or "")
    return {"body": body[:1500], "score": comment.get("score"),
            "id": comment.get("id")}


def fetch_posts(post_ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for start in range(0, len(post_ids), POST_BATCH):
        batch = post_ids[start:start + POST_BATCH]
        url = f"{API}/posts/ids?ids={','.join(batch)}"
        payload = http_json(url)
        for post in payload.get("data") or []:
            result[str(post.get("id"))] = compact_post(post)
        time.sleep(SLEEP_SEC)
    return result


def fetch_comments(post_id: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({
        "link_id": post_id, "limit": "100",
        "fields": "id,body,score",
    })
    try:
        payload = http_json(f"{API}/comments/search?{query}")
    except Exception:
        # Some posts 422 on the bare id but resolve with the fullname prefix.
        query = urllib.parse.urlencode({
            "link_id": f"t3_{post_id}", "limit": "100",
            "fields": "id,body,score",
        })
        payload = http_json(f"{API}/comments/search?{query}")
    comments = [compact_comment(c) for c in payload.get("data") or []]
    comments = [c for c in comments if c["body"] and c["body"] not in
                ("[deleted]", "[removed]")]
    comments.sort(key=lambda c: -(c.get("score") or 0))
    return comments[:MAX_COMMENTS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uids", type=Path, required=True,
                        help="text file, one reddit__<postid> per line")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        with args.out.open() as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["uid"])
    uids = [u.strip() for u in args.uids.read_text().splitlines()
            if u.strip().startswith("reddit__") and u.strip() not in done]
    if args.limit is not None:
        uids = uids[:args.limit]
    post_ids = [u.split("__", 1)[1] for u in uids]
    posts = fetch_posts(post_ids) if post_ids else {}
    print(f"posts fetched: {len(posts)}/{len(post_ids)} "
          f"(skipped done: {len(done)})", flush=True)
    failures = 0
    written = 0
    with args.out.open("a") as out:
        for uid in uids:
            post_id = uid.split("__", 1)[1]
            try:
                comments = fetch_comments(post_id)
                failures = 0
            except Exception as error:
                failures += 1
                print(f"comment fetch failed for {uid}: {error}", flush=True)
                if failures >= CIRCUIT_BREAKER_FAILURES:
                    print("circuit breaker: aborting (resumable)", flush=True)
                    break
                time.sleep(SLEEP_SEC * 4)
                continue
            out.write(json.dumps({
                "uid": uid, "post": posts.get(post_id),
                "comments": comments, "n_comments_kept": len(comments),
                "fetcher_version": FETCHER_VERSION,
                "fetched_unix": int(time.time()),
            }, sort_keys=True) + "\n")
            written += 1
            if written % 100 == 0:
                out.flush()
                print(f"{written}/{len(uids)} sources", flush=True)
            time.sleep(SLEEP_SEC)
    print(json.dumps({"fetcher_version": FETCHER_VERSION, "written": written,
                      "already_done": len(done)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
