"""Source adapters: a platform + query/target -> a list of unified candidates.

Each adapter has the signature:
    fn(query: str, cfg: dict, cursor: str | None) -> (candidates, next_cursor)

`candidate` is a dict:
    {source, native_id, uid, url, media_url, title, channel, duration}
  - uid:       globally-unique id used for dedup + filenames ("{source}__{id}")
  - url:       canonical page URL (stored in metadata)
  - media_url: what yt-dlp actually downloads (== url for most; the v.redd.it
               DASH manifest for Reddit, since the reddit permalink now needs login)

Enumeration is always metadata-only (no media) and, except for YouTube, needs no
auth and no proxy. `cursor` enables pagination across repeated calls to the same
target so the loop keeps making progress instead of re-seeing the same page.

Measured 2026-06-01: YouTube bot-walls datacenter proxies; Reddit/Dailymotion/
Odysee all enumerate + download fine direct from sk3.
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from . import scrape

log = logging.getLogger("sources")

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_UA = "Mozilla/5.0 (X11; Linux x86_64) norm-scraper-research/0.1"


def _safe(s) -> str:
    return _SAFE.sub("_", str(s))


def _get(url: str, params: dict, cfg: dict, timeout: int = 40, force_direct: bool = False):
    """HTTP GET through the rotating Webshare proxy pool (to dodge API rate-limits
    like Arctic Shift's 403s), with retry on a fresh proxy and a direct fallback.

    Controlled by api.use_proxy (default true). `force_direct` skips the proxy pool
    entirely (for official keyed APIs like the YouTube Data API, where the proxy
    only adds failed attempts). Returns a Response or None.
    """
    headers = {"User-Agent": _UA}
    pool = [] if force_direct else (
        scrape._load_proxy_pool(cfg) if cfg.get("api", {}).get("use_proxy", True) else [])
    attempts = max(1, cfg["network"].get("proxy_attempts", 6))
    host = (urlparse(url).hostname or "").casefold()
    host_limits = (cfg.get("api", {}).get("host_attempt_limits", {}) or {})
    for configured_host, limit in host_limits.items():
        configured_host = str(configured_host).casefold()
        if host == configured_host or host.endswith("." + configured_host):
            attempts = min(attempts, max(1, int(limit)))
            break
    tried: list[str] = []
    for i in range(attempts if pool else 1):
        proxies = None
        if pool:
            choices = [p for p in pool if p not in tried] or pool
            p = random.choice(choices)
            tried.append(p)
            proxies = {"http": p, "https": p}
        try:
            r = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=timeout)
            if r.status_code == 200:
                return r
            log.warning("GET %s -> HTTP %s (attempt %d/%d)", url, r.status_code, i + 1, attempts)
        except Exception as e:
            log.warning("GET %s failed (attempt %d): %s", url, i + 1, str(e)[:120])
    # direct fallback (no proxy)
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        return r if r.status_code == 200 else None
    except Exception as e:
        log.warning("GET %s direct fallback failed: %s", url, str(e)[:120])
        return None


def make_candidate(source: str, native_id, url: str, title=None, channel=None,
                   duration=None, media_url: Optional[str] = None) -> dict:
    return {
        "source": source,
        "native_id": str(native_id),
        "uid": f"{source}__{_safe(native_id)}",
        "url": url,
        "media_url": media_url or url,
        "title": title,
        "channel": channel,
        "duration": float(duration) if duration not in (None, "") else None,
    }


# --------------------------------------------------------------------------- #
# YouTube (kept; behind the proxy/cookies toggle in settings)
# --------------------------------------------------------------------------- #
_YT_API_KEY: Optional[str] = None


def _youtube_api_key(cfg: dict) -> Optional[str]:
    """Read + cache the YouTube Data API key from youtube.api_key_file."""
    global _YT_API_KEY
    if _YT_API_KEY is not None:
        return _YT_API_KEY or None
    path = cfg.get("youtube", {}).get("api_key_file", "~/.youtube-data-api-key.txt")
    p = Path(path).expanduser()
    _YT_API_KEY = p.read_text().strip() if p.exists() else ""
    if not _YT_API_KEY:
        log.warning("youtube api key not found at %s", p)
    return _YT_API_KEY or None


def _iso8601_seconds(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return None
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return float(h * 3600 + mi * 60 + se)


def _youtube_videos_meta(ids: list, key: str, cfg: dict) -> dict:
    """videos.list (1 unit/call, up to 50 ids) -> rich metadata per video id."""
    out = {}
    for i in range(0, len(ids), 50):
        r = _get("https://www.googleapis.com/youtube/v3/videos",
                 {"part": "snippet,statistics,contentDetails",
                  "id": ",".join(ids[i:i + 50]), "key": key}, cfg, timeout=30, force_direct=True)
        if r is None:
            continue
        try:
            for it in r.json().get("items", []):
                out[it["id"]] = it
        except Exception as e:
            log.warning("youtube videos.list bad json: %s", e)
    return out


def search_youtube_api(query: str, cfg: dict, cursor=None):
    """Enumerate via the YouTube Data API v3 (search.list, 100 units) + enrich with
    videos.list metadata (views/likes/category/tags/duration). Official + keyed, so
    it does NOT touch the scraping/IP-throttle surface; yt-dlp still does the
    download. `cursor` is the API nextPageToken. Rich metadata is stashed on each
    candidate (`meta`) and persisted by scrape.save_metadata for classification."""
    key = _youtube_api_key(cfg)
    if not key:
        return search_youtube(query, cfg, cursor, _force_ytdlp=True)  # fallback
    n = min(50, cfg["search"].get("results_per_query", 50))
    params = {"part": "snippet", "q": query, "type": "video",
              "maxResults": n, "key": key, "safeSearch": "none"}
    if cursor:
        params["pageToken"] = cursor
    r = _get("https://www.googleapis.com/youtube/v3/search", params, cfg, timeout=30, force_direct=True)
    if r is None:
        return [], None
    try:
        data = r.json()
    except Exception as e:
        log.warning("youtube search.list bad json: %s", e)
        return [], None
    ids = [it["id"]["videoId"] for it in data.get("items", [])
           if it.get("id", {}).get("videoId")]
    nxt = data.get("nextPageToken")
    if not ids:
        return [], nxt
    meta = _youtube_videos_meta(ids, key, cfg)
    cands = []
    for vid in ids:
        m = meta.get(vid, {})
        sn, st, cd = m.get("snippet", {}), m.get("statistics", {}), m.get("contentDetails", {})
        c = make_candidate("youtube", vid, f"https://www.youtube.com/watch?v={vid}",
                           sn.get("title"), sn.get("channelTitle"),
                           _iso8601_seconds(cd.get("duration")))
        c["meta"] = {
            "view_count": int(st["viewCount"]) if st.get("viewCount") else None,
            "like_count": int(st["likeCount"]) if st.get("likeCount") else None,
            "comment_count": int(st["commentCount"]) if st.get("commentCount") else None,
            "published_at": sn.get("publishedAt"),
            "channel_id": sn.get("channelId"),
            "category_id": sn.get("categoryId"),
            "tags": sn.get("tags"),
            "description": (sn.get("description") or "")[:800],
        }
        cands.append(c)
    return cands, nxt


def search_youtube(query: str, cfg: dict, cursor=None, _force_ytdlp: bool = False):
    # Prefer the Data API (official enumeration + rich metadata, no scraping); fall
    # back to yt-dlp ytsearch only if the API is off / key missing.
    if not _force_ytdlp and cfg.get("youtube", {}).get("use_api", False):
        return search_youtube_api(query, cfg, cursor)
    cands = [
        make_candidate("youtube", r["video_id"], r["url"], r.get("title"),
                       r.get("channel"), r.get("duration"))
        for r in scrape.search_query(query, cfg)
    ]
    return cands, None  # ytsearchN has no natural cursor


# --------------------------------------------------------------------------- #
# Dailymotion (official Data API, no key)
# --------------------------------------------------------------------------- #
def search_dailymotion(query: str, cfg: dict, cursor=None):
    n = min(100, cfg["search"].get("results_per_query", 50))
    page = int(cursor) if cursor else 1
    r = _get("https://api.dailymotion.com/videos",
             {"search": query, "fields": "id,title,duration,owner.screenname",
              "limit": n, "page": page, "sort": "relevance"}, cfg, timeout=30)
    if r is None:
        return [], None
    try:
        data = r.json()
    except Exception as e:
        log.warning("dailymotion bad json (%s): %s", query, e)
        return [], None
    cands = [
        make_candidate("dailymotion", v["id"],
                       f"https://www.dailymotion.com/video/{v['id']}",
                       v.get("title"), v.get("owner.screenname"), v.get("duration"))
        for v in data.get("list", [])
    ]
    nxt = str(page + 1) if data.get("has_more") else None
    return cands, nxt


# --------------------------------------------------------------------------- #
# Dailymotion RELATED — recommendation-snowball (Loubbrad/yt-scrape style).
# `query` is a seed dailymotion video id; we enqueue its related videos so the
# crawl follows the content graph instead of running out of search terms. A
# neighbor of a hit is likely a hit, and it stays in the seed's neighborhood
# (seed quiet -> follow quiet). Candidates are normal dailymotion videos.
# --------------------------------------------------------------------------- #
def related_dailymotion(seed_id: str, cfg: dict, cursor=None):
    n = min(100, cfg["search"].get("results_per_query", 50))
    page = int(cursor) if cursor else 1
    r = _get(f"https://api.dailymotion.com/video/{seed_id}/related",
             {"fields": "id,title,duration,owner.screenname", "limit": n, "page": page},
             cfg, timeout=30)
    if r is None:
        return [], None
    try:
        data = r.json()
    except Exception as e:
        log.warning("dailymotion related bad json (%s): %s", seed_id, e)
        return [], None
    cands = [
        make_candidate("dailymotion", v["id"],
                       f"https://www.dailymotion.com/video/{v['id']}",
                       v.get("title"), v.get("owner.screenname"), v.get("duration"))
        for v in data.get("list", []) if v.get("id") != seed_id
    ]
    # breadth-first snowball: cap how deep any single seed is paginated so the
    # crawl spreads across MANY neighbors rather than drilling one neighborhood
    # (snowball.max_pages; 0 = unlimited). Breadth comes from the growing set of
    # distinct seeds, not deep pagination of a few.
    max_pages = cfg.get("snowball", {}).get("max_pages", 0)
    more = data.get("has_more") and (not max_pages or page < max_pages)
    nxt = str(page + 1) if more else None
    return cands, nxt


# --------------------------------------------------------------------------- #
# Odysee / LBRY (Lighthouse search API, no key)
# --------------------------------------------------------------------------- #
def search_odysee(query: str, cfg: dict, cursor=None):
    n = min(50, cfg["search"].get("results_per_query", 50))
    frm = int(cursor) if cursor else 0
    r = _get("https://lighthouse.odysee.tv/search",
             {"s": query, "size": n, "from": frm, "mediaType": "video"}, cfg, timeout=30)
    if r is None:
        return [], None
    try:
        data = r.json()
    except Exception as e:
        log.warning("odysee bad json (%s): %s", query, e)
        return [], None
    cands = []
    for v in data:
        name, claim = v.get("name"), v.get("claimId")
        if not (name and claim):
            continue
        cands.append(make_candidate(
            "odysee", claim, f"https://odysee.com/{name}:{claim}",
            name, v.get("channel"), v.get("duration")))
    nxt = str(frm + n) if len(data) >= n else None
    return cands, nxt


# --------------------------------------------------------------------------- #
# Rumble (no-auth, server-rendered HTML search; yt-dlp Rumble extractor downloads)
# Highest-genre-density alternative to Dailymotion (native Road Rage / Public
# Freakouts categories). Search results are static HTML: anchors look like
#   /v7208bq-some-slug.html?e9s=...   ->  id "v7208bq", page URL strips the query.
# --------------------------------------------------------------------------- #
_RUMBLE_HREF = re.compile(r'href="(/v[0-9a-z]+-[^"]*?\.html)')


def search_rumble(query: str, cfg: dict, cursor=None):
    page = int(cursor) if cursor else 1
    r = _get("https://rumble.com/search/video", {"q": query, "page": page}, cfg, timeout=30)
    if r is None:
        return [], None
    cands = []
    for href in dict.fromkeys(_RUMBLE_HREF.findall(r.text)):
        path = href.split("?", 1)[0]            # drop tracking query string
        m = re.match(r"/(v[0-9a-z]+)-(.+)\.html$", path)
        if not m:
            continue
        vid, slug = m.group(1), m.group(2)
        cands.append(make_candidate(
            "rumble", vid, "https://rumble.com" + path, slug.replace("-", " ")))
    nxt = str(page + 1) if cands else None      # stop when a page yields nothing
    return cands, nxt


# --------------------------------------------------------------------------- #
# Reddit (Arctic Shift API for enumeration + v.redd.it DASH for download)
# --------------------------------------------------------------------------- #
def _reddit_video_cands(posts: list, cfg: dict) -> list:
    """Convert Arctic Shift post objects -> video candidates (those with a
    reddit-hosted DASH stream). Carries `crosspost_parent` for the snowball."""
    allow_nsfw = cfg.get("reddit", {}).get("allow_nsfw", False)
    cands = []
    for p in posts:
        rv = (p.get("media") or {}).get("reddit_video") or {}
        dash = rv.get("dash_url") or rv.get("hls_url")  # NOT fallback_url (no audio)
        if not dash:
            continue  # not a reddit-hosted video
        if p.get("over_18") and not allow_nsfw:
            continue
        c = make_candidate(
            "reddit", p["id"], "https://www.reddit.com" + p.get("permalink", ""),
            p.get("title"), p.get("subreddit"), rv.get("duration"), media_url=dash)
        c["crosspost_parent"] = p.get("crosspost_parent")  # t3_<id> or None
        cands.append(c)
    return cands


def _arctic_posts(params: dict, cfg: dict, what: str):
    r = _get("https://arctic-shift.photon-reddit.com/api/posts/search", params, cfg, timeout=50)
    if r is None:
        log.warning("reddit/arctic-shift failed (%s)", what)
        return None
    try:
        return r.json().get("data", [])
    except Exception as e:
        log.warning("reddit/arctic-shift bad json (%s): %s", what, e)
        return None


def list_reddit(target: str, cfg: dict, cursor=None):
    """`target` is a subreddit name (e.g. 'PublicFreakout'). Paginates backwards
    in time via the Arctic Shift `before` cursor (epoch seconds)."""
    n = min(100, cfg["search"].get("results_per_query", 50))
    params = {"subreddit": target, "limit": n, "sort": "desc"}
    if cursor:
        params["before"] = cursor
    posts = _arctic_posts(params, cfg, target)
    if posts is None:
        return [], None
    oldest = None
    for p in posts:
        cu = p.get("created_utc")
        if cu is not None:
            oldest = cu if oldest is None else min(oldest, cu)
    nxt = str(int(oldest)) if (oldest and len(posts) >= n) else None
    return _reddit_video_cands(posts, cfg), nxt


def crosspost_subreddits(seed_id: str, cfg: dict) -> set:
    """Reddit recommendation-snowball (subreddit discovery).

    `seed_id` is a hit's parent post id (t3_<id>). Arctic Shift's
    `crosspost_parent_id` reverse-index returns every post that crossposted the
    same video, no-auth in one call. Because crossposts are the SAME v.redd.it
    video (re-downloading would just duplicate it), the value is the *set of
    subreddits* they land in -- new norm-violation communities we then add to the
    firehose to mine their full history. This is the Reddit analog of Dailymotion
    related-videos, and it sidesteps Reddit's OAuth wall on /duplicates.json."""
    n = min(100, cfg["search"].get("results_per_query", 50))
    posts = _arctic_posts({"crosspost_parent_id": seed_id, "limit": n,
                           "fields": "subreddit"}, cfg, seed_id)
    if not posts:
        return set()
    # drop user-profile pseudo-subreddits ("u_<name>") -- they aren't communities
    # and would be dead-end firehose queries.
    return {s for p in posts if (s := p.get("subreddit")) and not s.startswith("u_")}


def search_reddit(term: str, cfg: dict, cursor=None):
    """Anchored full-text search: sweep a norm-category `term` across the curated
    `reddit.search_subreddits`. Arctic Shift requires `query` to be paired with a
    subreddit, so the cursor "subIdx:before" walks one sub at a time (paginating
    by time within each, then advancing to the next sub)."""
    subs = cfg.get("reddit", {}).get("search_subreddits") or []
    if not subs:
        return [], None
    sub_idx, before = 0, None
    if cursor:
        parts = cursor.split(":", 1)
        sub_idx = int(parts[0]) if parts[0] else 0
        before = parts[1] if len(parts) > 1 and parts[1] else None
    if sub_idx >= len(subs):
        return [], None  # swept every sub -> exhausted
    n = min(100, cfg["search"].get("results_per_query", 50))
    params = {"query": term, "subreddit": subs[sub_idx], "limit": n, "sort": "desc"}
    if before:
        params["before"] = before
    _adv = f"{sub_idx + 1}:" if sub_idx + 1 < len(subs) else None  # next sub / done
    posts = _arctic_posts(params, cfg, f"{term}@{subs[sub_idx]}")
    if posts is None:
        return [], _adv
    oldest = None
    for p in posts:
        cu = p.get("created_utc")
        if cu is not None:
            oldest = cu if oldest is None else min(oldest, cu)
    # more pages in this sub? else advance to the next sub
    nxt = f"{sub_idx}:{int(oldest)}" if (len(posts) >= n and oldest) else _adv
    return _reddit_video_cands(posts, cfg), nxt


ADAPTERS = {
    "youtube": search_youtube,
    "dailymotion": search_dailymotion,
    "dmrelated": related_dailymotion,   # recommendation-snowball from a seed video
    "rumble": search_rumble,
    "odysee": search_odysee,
    "reddit": list_reddit,              # per-subreddit firehose
    "rdsearch": search_reddit,          # norm-term full-text sweep across subs
}
# Reddit crosspost-snowball is NOT an adapter: crossposts are the same video, so
# instead of enqueuing them we harvest their subreddits (crosspost_subreddits)
# and add those to the `reddit` firehose -- handled in search_loop.process_video.

# Platforms whose queries are free text the query_generator can mutate/expand.
# Reddit targets/structured queries (reddit/rdsearch) and snowball seeds
# (dmrelated/rdrelated) are excluded -- their "query" is a name/term/id.
TEXT_QUERY_PLATFORMS = {"youtube", "dailymotion", "odysee", "rumble"}


def enumerate_candidates(platform: str, query: str, cfg: dict, cursor=None):
    fn = ADAPTERS.get(platform)
    if not fn:
        log.warning("unknown platform: %s", platform)
        return [], None
    return fn(query, cfg, cursor)
