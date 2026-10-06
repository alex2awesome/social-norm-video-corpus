# Norm taxonomy: literature review → search-term mapping

**Date:** 2026-06-02
**Purpose:** Ground the crawler's search terms in the academic literature on social
norms, so the corpus has (a) defensible *coverage* of the norm space and (b)
per-clip *provenance* back to a norm category for the eventual paper.

This note documents the research process, what the field has done (and whether
anyone is doing what we're doing), the synthesized taxonomy, and how the
taxonomy maps to the live search terms in `config/norm_taxonomy.yaml`.

---

## 1. Process

Three parallel literature sweeps (web search across arXiv / ACL Anthology /
CVPR / Google Scholar / social-science sources), then a synthesis pass:

1. **Video & multimodal norm work + "is anyone doing this?"** — benchmarks,
   datasets, and any project that scrapes in-the-wild video or uses bystander
   reactions as a labeling signal.
2. **Text & knowledge-base norm datasets + their taxonomies** — the NLP norm
   datasets and how each one carves up the norm space (incl. Reddit-specific
   norm-violation work, since we also crawl Reddit).
3. **Social-science norm taxonomies** — the theoretical frameworks that define
   the *total space* of social norms (moral psychology, values theory,
   politeness, proxemics, etiquette, cross-cultural dimensions).

The synthesis collapses all category systems into **18 norm categories**
(Section 4), each tagged with the frameworks that motivate it.

---

## 2. Is anyone doing what we're doing?

**No located published work does exactly this** — *scrape in-the-wild web video at
scale, use spontaneous **audible bystander reactions** (gasp, laughter, "excuse
me", confrontation) as the automatic label/anchor for a norm violation, and save
a ~20s-pre-reaction clip + transcript.* The closest precedents each share one or
two ingredients but not the combination:

| Work | Year | What it shares | What's different from us |
|---|---|---|---|
| **NormVio** (Park et al., EMNLP-F 2021) | 2021 | "community reaction = violation label" (Reddit mod-removal); a 9-category rule taxonomy | Text-only online conversations; the reaction is a moderator action, not an audible human one |
| **MUStARD** (Castro et al., ACL 2019) | 2019 | uses **audience laughter track** as a labeling cue | Scripted TV sitcoms; sarcasm not norms; canned laughter, not spontaneous bystanders |
| **SMILE** (Hyun et al., NAACL-F 2024) | 2024 | laughter-as-unit, with explanations of *why* people laugh | Curated sitcom/TED; laughter only; no norm-violation framing |
| **EgoNormia** (arXiv 2502.20490) | 2025 | **physical social norms in video**; a clean 7-category scheme (we reuse it) | Curated egocentric Ego4D clips, scripted MCQ benchmark; no in-the-wild mining, no audible-reaction anchor |
| **Cheng et al.** (PNAS 2023) | 2023 | argues **social emotions signal norm violations** — our core premise | Text vignettes + GPT-3, not video |
| **Bystander Affect Detection** (arXiv 2303.04835) | 2023 | "bystander reaction = failure signal" | Facial affect for robot-failure HRI; not social norms, not audible |
| XD-Violence / DVD / RWF-2000 | 2020–25 | in-the-wild (YouTube) video + audio | Target *violence* detection, not everyday norm violations; no reaction-anchored moments |

**Our novelty = (in-the-wild web video at scale) × (spontaneous audible bystander
reaction as auto-label) × (20s-pre-roll "moment" + transcript).** MUStARD/SMILE
have reaction-as-signal but scripted; NormVio has reaction-as-label but text;
EgoNormia has video norms but curated/MCQ. None combine all three.

---

## 3. The space of norms (frameworks consulted)

**Moral psychology / values**
- **Moral Foundations Theory** (Haidt): care/harm, fairness/cheating,
  loyalty/betrayal, authority/subversion, sanctity/degradation, liberty/oppression.
- **Schwartz Basic Human Values** (10 values: self-direction, stimulation,
  hedonism, achievement, power, security, conformity, tradition, benevolence,
  universalism) — each implies a violation profile.
- **Turiel Social-Domain Theory**: moral vs conventional vs personal; **taboos**
  = affect-laden prohibitions (disgust).
- **Cialdini**: descriptive vs injunctive norms (what people *do* vs *approve*).

**Interaction / space / etiquette**
- **Brown & Levinson politeness**: positive vs negative face; face-threatening acts.
- **Hall proxemics**: intimate / personal / social / public zones; territoriality.
- **Etiquette sub-domains**: table manners, queue etiquette, phone/digital
  etiquette, transit etiquette, noise norms, hygiene norms, reciprocity/gift norms.

**Cross-cultural**
- **Hofstede** 6 dimensions (norm violations are dimension-relative).
- **Gelfand tightness–looseness** (same act = violation in tight, fine in loose).
- **Honor / face / dignity** cultures.

**Applied / HRI / benchmarks**
- **EgoNormia 7**: safety, privacy, proxemics, politeness, cooperation,
  coordination/proactivity, communication/legibility (video-grounded — closest to us).
- HRI social-error taxonomies (physical vs psychological safety, legibility).

**NLP norm datasets** (each contributes a category system): NormBank (SCENE /
Goffman dramaturgy, expected/okay/unexpected), Social-Chemistry-101 (RoTs ×
moral foundations × cultural pressure × legality), Social-IQa, ETHICS (justice /
deontology / virtue / utilitarianism / commonsense), Moral Stories, Delphi /
Commonsense Norm Bank, Moral Integrity Corpus, ProsocialDialog (casual / needs-
caution / needs-intervention), NormSage, **NormAd** (75-country etiquette),
Social Bias Frames (offense / lewd / hate / target group), and **NormVio /
CPL-NoViD** for Reddit (incivility, harassment, spam, format, content,
off-topic, hate speech, trolling, meta).

---

## 4. The synthesized taxonomy → search terms

The 18 categories below are the keys in `config/norm_taxonomy.yaml`; each holds a
curated list of Dailymotion-style search phrases (with colloquial variants). The
crawler seeds every term tagged with its category.

| # | Category key | Label | Primary frameworks |
|---|---|---|---|
| 1 | `proxemics_personal_space` | Proxemics / personal space | Hall, EgoNormia-Proxemics, B&L negative face |
| 2 | `queue_turn_taking` | Queue / line-jumping / turn-taking | MFT-Fairness, EgoNormia-Coordination, Cialdini |
| 3 | `politeness_courtesy` | Politeness / basic courtesy | B&L positive face, EgoNormia-Politeness |
| 4 | `entitled_outburst` | Entitled outbursts / "Karen" confrontation | NormVio-incivility, MFT-Liberty, Schwartz-Power |
| 5 | `noise_phone_public` | Noise / phone in shared space | etiquette-noise, transit, Cialdini |
| 6 | `hygiene_disgust` | Hygiene / disgust / sanctity | MFT-Sanctity, Turiel-taboo |
| 7 | `table_manners_dining` | Table manners / dining | etiquette-table, Turiel-conventional |
| 8 | `safety_endangering` | Safety / endangering others / reckless | MFT-Care, EgoNormia-Safety, Schwartz-Security |
| 9 | `privacy_surveillance` | Privacy / filming / snooping | EgoNormia-Privacy, SBF, NormBank |
| 10 | `theft_property` | Theft / property damage / vandalism | MFT-Fairness, Turiel-moral |
| 11 | `cheating_freeriding` | Cheating / free-riding / not paying share | MFT-Fairness, Cialdini, reciprocity |
| 12 | `dishonesty_scam` | Dishonesty / lying / scamming | MFT-Fairness, ETHICS |
| 13 | `authority_ritual` | Disrespecting authority / sacred space | MFT-Authority/Sanctity, Schwartz-Tradition, Hofstede |
| 14 | `neglect_bullying` | Failure to help / bullying / mocking vulnerable | MFT-Care, Schwartz-Benevolence, ProsocialDialog |
| 15 | `discrimination_harassment` | Discrimination / harassment / hate | SBF, NormVio-hate/harassment, MFT-Liberty |
| 16 | `transit_flow_coordination` | Coordination / transit flow / blocking | EgoNormia-Coordination/Communication, transit |
| 17 | `reciprocity_ingratitude` | Reciprocity / ingratitude / broken promises | reciprocity, MFT-Loyalty/Fairness |
| 18 | `cross_cultural_etiquette` | Cross-cultural etiquette violations | NormAd, NormSage, Hofstede, Gelfand |

EgoNormia's 7 categories map onto ours as: Safety→#8, Privacy→#9, Proxemics→#1,
Politeness→#3, Cooperation→#11, Coordination→#16, Communication→#16. The
categories in-the-wild video adds *beyond* the egocentric benchmark are the loud
classes — #4 (entitled outburst), #10 (theft), #15 (discrimination/harassment) —
which dominate "people being rude" content but are absent from curated
egocentric data.

**Multi-label note:** a clip can belong to several categories (queue-jumping =
MFT-Fairness + EgoNormia-Coordination + etiquette). We store the single
*surfacing* category (the query that found it); richer multi-label tagging can be
done post-hoc from the transcript.

---

## 5. Provenance design (for the paper)

Every clip is traceable to a norm category through the existing + new plumbing:

- **video → query:** `data/metadata/<uid>.json` stores `found_by_query`, and
  `seen_videos.query` stores the surfacing query string. (pre-existing)
- **video → category:** `seen_videos.category` is stamped at enumeration time
  with the surfacing query's taxonomy key. (new column; migrated in-place)
- **query → category:** `queries.category` records each query's taxonomy key.
- **snowball inheritance:** when a Dailymotion HIT enqueues its `dmrelated`
  child, the child query is created **with the parent's category**, so a whole
  recommendation branch stays attributed to the norm class that seeded it. A
  road-rage hit → road-rage-tagged related videos, etc.

So for the paper, `SELECT category, COUNT(*) ... GROUP BY category` over
`seen_videos` (filtered to hits) gives the per-norm yield directly, and joining
`reactions` → `seen_videos` attributes each saved reaction to a norm class —
across both search-found and snowball-found clips.

---

## 6. Breadth-first crawling

- **Across terms:** the scheduler is platform round-robin + within-platform LRU
  (`state.next_query`), so the ~200 taxonomy terms are visited in rotation rather
  than one being exhausted first — breadth-first by construction. All terms share
  equal priority.
- **Snowball, breadth-first:** `snowball.enabled: true` with `priority: 1.0` (no
  depth boost) and `max_pages: 3` (`related_dailymotion` stops paginating a seed
  after 3 pages). Volume comes from *many distinct seeds*, not deep pagination of
  a few — addressing the observed tendency of an unconstrained snowball to drill
  one neighborhood (e.g. a single road-rage cluster).

---

## 7. Platforms & snowball mechanisms (current)

The taxonomy now drives **three** discovery platforms, each category-tagged, plus
two snowball mechanisms:

| Platform key | What it does | Snowball |
|---|---|---|
| `dailymotion` | free-text API search; 190 taxonomy terms @ prio 2.0 lead breadth-first | `dmrelated`: a HIT enqueues the video's related-videos (genuinely new clips), category inherited, depth-capped (`max_pages`) for breadth |
| `rumble` | no-auth HTML search (`/search/video?q=`), yt-dlp download; highest genre density | uses `dmrelated`-style not yet — relies on breadth of terms (future: Rumble related) |
| `rdsearch` | **anchored** full-text: Arctic Shift requires `query`+`subreddit`, so each term sweeps `reddit.search_subreddits` (cursor `subIdx:before`) — this is how norm categories inform Reddit | n/a |
| `reddit` | per-subreddit firehose (Arctic Shift, v.redd.it DASH) | **crosspost discovery**: a HIT's `crosspost_parent_id` reveals other subs carrying the same video → new subs added to the firehose (`source='related-sub'`, category inherited). Crossposts are the same video, so we harvest *subreddits*, not re-download. |

Arctic Shift finding (verified 2026-06-02): global free-text search is **not**
available — `query=`/`title=`/`selftext=` require an `author` or `subreddit`
anchor. The only unanchored text filter is `url=` (prefix). Hence the anchored
per-subreddit sweep for `rdsearch`, and crosspost-`subreddit`-discovery (not the
dead, OAuth-walled `/duplicates.json`) for the Reddit snowball.

Other platforms scoped but deferred: Internet Archive (easy public API, modest
yield), Rutube (Russian dashcam genre, verify US-IP reachability), PeerTube/
SepiaSearch (clean API, thin genre). Excluded: Bilibili (412-blocks US datacenter
IPs), Veoh (defunct), Facebook/Instagram (login wall), Vimeo/Twitch/Niconico
(wrong genre or OAuth), VK (genre fit but OAuth-gated).

## 8. Key references

- EgoNormia — arXiv 2502.20490 · https://opensocial.world/articles/egonormia
- NormVio — https://aclanthology.org/2021.findings-emnlp.288.pdf · code https://github.com/chan0park/NormVio
- NormBank — https://aclanthology.org/2023.acl-long.429/ · arXiv 2305.17008
- Social Chemistry 101 — https://aclanthology.org/2020.emnlp-main.48/ · arXiv 2011.00620
- Social-IQa — https://aclanthology.org/D19-1454/ · arXiv 1904.09728
- ETHICS — arXiv 2008.02275 · https://github.com/hendrycks/ethics
- Moral Stories — https://aclanthology.org/2021.emnlp-main.54/
- Delphi / Commonsense Norm Bank — arXiv 2110.07574
- Moral Integrity Corpus — https://aclanthology.org/2022.acl-long.261/
- ProsocialDialog — https://aclanthology.org/2022.emnlp-main.267/
- NormSage — https://aclanthology.org/2023.emnlp-main.941/
- NormAd — arXiv 2404.12464
- Social Bias Frames — https://aclanthology.org/2020.acl-main.486/
- MUStARD — https://aclanthology.org/P19-1455/
- SMILE — arXiv 2312.09818
- Cheng et al., social-norm-violation via social emotions — PNAS 2023, PMC10199061
- Bystander Affect Detection — arXiv 2303.04835
- Moral Foundations Theory; Schwartz Basic Values; Brown & Levinson politeness;
  Hall proxemics; Turiel social-domain theory; Cialdini focus theory;
  Hofstede dimensions; Gelfand tightness–looseness.
</content>
