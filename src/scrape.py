"""yt-dlp wrapper: search enumeration + video download, proxy-aware.

Search enumeration uses `--flat-playlist --dump-json` (metadata only, no media)
so we can dedupe candidates against state.db *before* spending bandwidth.
Download uses a 720p-capped merged mp4 with duration / live filters.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Optional

from . import state

log = logging.getLogger("scrape")

_PROXY_POOL: Optional[list[str]] = None


def _parse_webshare_line(line: str, scheme: str) -> Optional[str]:
    """Convert a Webshare 'ip:port:user:pass' line to a proxy URL.

    Also accepts lines that are already full URLs (http://...:.../).
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "://" in line:
        return line
    parts = line.split(":")
    if len(parts) == 4:
        ip, port, user, pwd = parts
        return f"{scheme}://{user}:{pwd}@{ip}:{port}"
    if len(parts) == 2:
        ip, port = parts
        return f"{scheme}://{ip}:{port}"
    return None


def _load_proxy_pool(cfg: dict) -> list[str]:
    global _PROXY_POOL
    if _PROXY_POOL is not None:
        return _PROXY_POOL
    pool: list[str] = []
    net = cfg["network"]
    list_file = net.get("proxy_list_file")
    scheme = net.get("proxy_scheme", "http")
    if list_file and Path(list_file).expanduser().exists():
        for line in Path(list_file).expanduser().read_text().splitlines():
            url = _parse_webshare_line(line, scheme)
            if url:
                pool.append(url)
        log.info("loaded %d proxies from %s", len(pool), list_file)
    _PROXY_POOL = pool
    return pool


# Proxies that failed with a connection/auth error this session. ~40% of the
# stale Webshare pool is dead; once a proxy 407s or times out we stop drawing it
# so the loop converges on the live ones instead of re-hanging on corpses.
_DEAD: set[str] = set()


def _proxy(cfg: dict) -> Optional[str]:
    """Pick a live proxy: random from (pool - blacklist), else env fallback.

    Returns None when use_proxy is false (direct connection) -- the measured
    default, since datacenter proxies are bot-walled. If every pooled proxy is
    blacklisted we clear the blacklist and try again -- dead proxies sometimes
    come back, and a fully-empty pool is worse than retrying.
    """
    if not cfg["network"].get("use_proxy", False):
        return None
    pool = _load_proxy_pool(cfg)
    if pool:
        live = [p for p in pool if p not in _DEAD]
        if not live:
            log.warning("all %d proxies blacklisted; resetting blacklist", len(pool))
            _DEAD.clear()
            live = pool
        return random.choice(live)
    env_name = cfg["network"].get("proxy_env", "YTDLP_PROXY")
    return os.environ.get(env_name) or None


_PROXY_ERR = re.compile(
    r"proxy|tunnel connection failed|407|timed out|timeout|"
    r"connection (reset|refused|aborted)|unable to connect|failed to establish",
    re.IGNORECASE,
)


def _is_proxy_error(stderr: str) -> bool:
    return bool(stderr and _PROXY_ERR.search(stderr))


def _base_args(cfg: dict) -> list[str]:
    """yt-dlp args common to every call EXCEPT the proxy (chosen per attempt)."""
    net = cfg["network"]
    args: list[str] = []
    cookies = net.get("cookies_file")
    if cookies:
        args += ["--cookies", str(state.resolve_path(cookies))]
    args += [
        "--retries", str(net.get("retries", 5)),
        "--socket-timeout", str(net.get("socket_timeout", 20)),
        "--sleep-requests", str(random.randint(
            net.get("sleep_requests_min", 1), net.get("sleep_requests_max", 3))),
        "--no-warnings",
        "--ignore-config",
    ]
    if net.get("limit_rate"):
        args += ["--limit-rate", str(net["limit_rate"])]
    # anti-bot: rotate User-Agent + (YouTube) player-client per call
    uas = net.get("user_agents") or []
    if uas:
        args += ["--user-agent", random.choice(uas)]
    pcs = net.get("youtube_player_clients") or []
    if pcs:
        args += ["--extractor-args", "youtube:player_client=" + random.choice(pcs)]
    return args


def _run_ytdlp(tail: list[str], cfg: dict, timeout: int) -> subprocess.CompletedProcess:
    """Run yt-dlp, rotating to a fresh proxy on connection failure.

    A proxy/connection error blacklists that proxy and retries with another. A
    *content* error (private/unavailable/age-gated video -- fails identically on
    every proxy) stops retrying immediately. A success (rc==0, including a
    match-filter skip which also exits 0) returns at once. `--socket-timeout`
    bounds how long a dead proxy can hang each attempt.
    """
    attempts = max(1, cfg["network"].get("proxy_attempts", 6))
    base = _base_args(cfg)
    last: Optional[subprocess.CompletedProcess] = None
    for i in range(attempts):
        proxy = _proxy(cfg)
        pargs = (["--proxy", proxy] if proxy else [])
        cmd = ["yt-dlp", *pargs, *base, *tail]
        log.debug("yt-dlp attempt %d/%d (proxy=%s)", i + 1, attempts,
                  "set" if proxy else "none")
        last = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if last.returncode == 0:
            return last
        stderr = (last.stderr or "").strip()
        if _is_proxy_error(stderr):
            if proxy:
                _DEAD.add(proxy)
            log.warning("yt-dlp attempt %d/%d proxy error (blacklisted, %d dead): %s",
                        i + 1, attempts, len(_DEAD), stderr[:160])
            continue
        # content-side failure: same on every proxy -> don't waste attempts
        log.warning("yt-dlp content error (no retry): %s", stderr[:200])
        return last
    return last


def proxy_ready(cfg: dict) -> bool:
    """True if a proxy is configured, or the config does not require one."""
    if not cfg["network"].get("require_proxy", True):
        return True
    return _proxy(cfg) is not None


def search_query(query: str, cfg: dict) -> list[dict]:
    """Enumerate candidate videos for a search query (metadata only)."""
    n = cfg["search"].get("results_per_query", 50)
    tail = [
        f"ytsearch{n}:{query}",
        "--flat-playlist",
        "--dump-json",
    ]
    try:
        proc = _run_ytdlp(tail, cfg, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("search timeout: %s", query)
        return []
    if proc is None:
        return []
    if proc.returncode != 0:
        log.warning("search failed (%s): %s", query, proc.stderr.strip()[:300])
    results: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = d.get("id")
        if not vid:
            continue
        results.append({
            "video_id": vid,
            "title": d.get("title"),
            "channel": d.get("channel") or d.get("uploader"),
            "duration": d.get("duration"),
            "url": d.get("url") or f"https://www.youtube.com/watch?v={vid}",
        })
    log.info("query '%s' -> %d candidates", query, len(results))
    return results


def download(cand: dict, cfg: dict) -> Optional[Path]:
    """Download one candidate's media (<=720p) to raw_video/{uid}.ext.

    Downloads `cand['media_url']` (the v.redd.it DASH manifest for Reddit, the
    page URL elsewhere) and names the output by uid so we can find it regardless
    of the extractor. Returns the path, or None if skipped/failed. Duration/live
    filtering is done from enumeration metadata by the caller; the yt-dlp
    match-filter here is a backstop (it is silently ignored when the extractor
    does not expose duration, e.g. a raw DASH manifest).
    """
    if not proxy_ready(cfg):
        raise RuntimeError(
            "proxy required but unavailable; set network.proxy_list_file / %s, "
            "or network.require_proxy: false" % cfg["network"].get("proxy_env"))

    uid = cand["uid"]
    media_url = cand.get("media_url") or cand["url"]
    raw_dir = state.resolve_path(cfg["paths"]["raw_video"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(raw_dir / f"{uid}.%(ext)s")

    s = cfg["search"]
    # `<?` lets the comparison PASS when the field is absent (e.g. a raw v.redd.it
    # DASH manifest exposes no duration); real duration filtering is done from
    # enumeration metadata before we ever get here.
    filters = [f"duration <? {s.get('max_duration_sec', 1800)}"]
    if s.get("skip_live", True):
        filters.append("!is_live")

    tail = [
        media_url,
        "-f", f"bestvideo[height<={s.get('max_height', 720)}]+bestaudio/best/"
              f"best[height<={s.get('max_height', 720)}]/best",
        "--merge-output-format", "mp4",
        "-o", out_tmpl,
        "--match-filter", " & ".join(filters),
        "--no-playlist",
        "--write-info-json",
    ]
    try:
        proc = _run_ytdlp(tail, cfg, timeout=cfg.get("download_timeout", 2400))
    except subprocess.TimeoutExpired:
        log.warning("download timeout: %s", uid)
        return None
    if proc is None or proc.returncode != 0:
        log.warning("download failed (%s) after retries", uid)
        return None

    for ext in ("mp4", "mkv", "webm", "m4a", "mp3", "opus", "webp"):
        p = raw_dir / f"{uid}.{ext}"
        if p.exists():
            return p
    log.info("no media for %s (filtered out or unavailable)", uid)
    return None


def save_metadata(cand: dict, query: str, cfg: dict, query_source: str = None) -> Path:
    """Persist metadata immediately, merging the richer .info.json if present."""
    meta_dir = state.resolve_path(cfg["paths"]["metadata"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = state.resolve_path(cfg["paths"]["raw_video"])
    info_path = raw_dir / f"{cand['uid']}.info.json"

    record = {
        "uid": cand["uid"],
        "source": cand.get("source"),          # platform adapter (youtube/dailymotion/...)
        "native_id": cand.get("native_id"),
        "url": cand.get("url"),
        "title": cand.get("title"),
        "channel": cand.get("channel"),
        "duration": cand.get("duration"),
        "found_by_query": query,               # the query STRING that surfaced this video
        "query_source": query_source,          # the query's TYPE (seed/instructional/exploit/...)
    }
    if cand.get("meta"):                    # rich source metadata (e.g. YouTube Data API)
        record["source_meta"] = cand["meta"]
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            record.update({
                "title": info.get("title", record["title"]),
                "channel": info.get("channel") or info.get("uploader") or record["channel"],
                "duration": info.get("duration", record["duration"]),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
            })
        except (json.JSONDecodeError, OSError):
            pass

    out = meta_dir / f"{cand['uid']}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return out


def purge_raw(uid: str, cfg: dict) -> None:
    """Delete the raw media + info.json for a miss (transcript is kept elsewhere)."""
    raw_dir = state.resolve_path(cfg["paths"]["raw_video"])
    for p in raw_dir.glob(f"{uid}.*"):
        try:
            p.unlink()
        except OSError as e:
            log.warning("could not purge %s: %s", p, e)


def check_proxies(cfg: dict, timeout: int = 15) -> None:
    """Probe each proxy in the pool with a quick HTTPS request; print a tally."""
    import urllib.request

    pool = _load_proxy_pool(cfg)
    if not pool:
        print("no proxy pool configured (network.proxy_list_file)")
        return
    live = 0
    for i, p in enumerate(pool):
        handler = urllib.request.ProxyHandler({"http": p, "https": p})
        opener = urllib.request.build_opener(handler)
        try:
            opener.open("https://api.ipify.org", timeout=timeout).read()
            live += 1
            mark = "ok"
        except Exception:
            mark = "DEAD"
        print(f"[{i:3d}/{len(pool)}] {mark}")
    print(f"\n{live}/{len(pool)} proxies live")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="scrape utilities")
    ap.add_argument("--check-proxies", action="store_true",
                    help="probe the proxy pool and report how many are live")
    ap.add_argument("--search", metavar="QUERY", help="enumerate candidates for a query")
    args = ap.parse_args()
    cfg = state.load_config()
    if args.check_proxies:
        check_proxies(cfg)
    elif args.search:
        for r in search_query(args.search, cfg):
            print(r["video_id"], "-", r.get("title"))
