"""
utils/sources.py — Actual scrapers for data ingestion.
arXiv API, Google Patents search, job postings scraping.
All free. All rate-limit safe. All return structured dicts.

IMPORTANT: These are real HTTP calls. They will fail gracefully and return [] on errors.
The LLM agents downstream handle empty inputs — they still run with their own knowledge.
"""
from __future__ import annotations
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Rate limiting ──────────────────────────────────────────────────────────────
_ARXIV_DELAY  = 3.0   # seconds between arXiv requests (their API requires this)
_REQUEST_TIMEOUT = 30  # seconds


def _safe_get(url: str, params: dict = None, headers: dict = None) -> Optional[requests.Response]:
    """Safe HTTP GET with timeout and error handling."""
    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers or {"User-Agent": "rd-engine-research/1.0 (research tool)"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp
    except requests.exceptions.Timeout:
        logger.warning(f"[Sources] Timeout fetching {url[:80]}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[Sources] Request failed {url[:80]}: {e}")
        return None


# ── arXiv API ─────────────────────────────────────────────────────────────────

def fetch_arxiv_papers(
    keywords: list[str],
    max_results: int = 15,
    days_back: int = 30,
) -> list[dict]:
    """
    Fetch recent papers from arXiv API matching keywords.
    Returns list of {title, abstract, url, authors, published}.
    arXiv API docs: https://info.arxiv.org/help/api/user-manual.html
    """
    results = []
    seen_urls = set()

    # Build query — combine keywords with OR
    # Short query — arXiv API breaks with long queries
    query_terms = " OR ".join(keywords[:4])
    cat_filter = "cat:cs.AR OR cat:cs.ET OR cat:eess.SP OR cat:cs.DC"
    full_query = f"({query_terms}) AND ({cat_filter})"

    params = {
        "search_query": full_query,
        "sortBy":       "lastUpdatedDate",
        "sortOrder":    "descending",
        "max_results":  max_results,
        "start":        0,
    }

    logger.info(f"[arXiv] Fetching papers for {len(keywords)} keywords")
    time.sleep(_ARXIV_DELAY)

    resp = _safe_get("https://export.arxiv.org/api/query", params=params)
    if not resp:
        return []

    # Parse Atom XML response
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {
            "atom":  "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        for entry in root.findall("atom:entry", ns):
            try:
                title    = entry.findtext("atom:title",    "", ns).strip().replace("\n", " ")
                abstract = entry.findtext("atom:summary",  "", ns).strip().replace("\n", " ")
                url      = entry.findtext("atom:id",       "", ns).strip()
                published_str = entry.findtext("atom:published", "", ns)

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Parse date
                try:
                    pub_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    if pub_date.replace(tzinfo=None) < cutoff_date:
                        continue
                except (ValueError, AttributeError):
                    pass

                authors = [
                    author.findtext("atom:name", "", ns)
                    for author in entry.findall("atom:author", ns)
                ]

                results.append({
                    "title":     title,
                    "abstract":  abstract[:800],
                    "url":       url,
                    "authors":   authors[:5],
                    "published": published_str,
                    "source":    "arxiv",
                })
            except Exception as e:
                logger.debug(f"[arXiv] Failed parsing entry: {e}")
                continue

    except Exception as e:
        logger.error(f"[arXiv] XML parsing failed: {e}")
        return []

    logger.info(f"[arXiv] Retrieved {len(results)} papers")
    return results


def fetch_semantic_scholar_papers(
    keywords: list[str],
    max_results: int = 10,
    year_from: int = None,
) -> list[dict]:
    """
    Fetch papers from Semantic Scholar API (free, no key needed for basic use).
    https://api.semanticscholar.org/graph/v1
    """
    results = []
    if year_from is None:
        year_from = datetime.now().year - 1

    query = " ".join(keywords[:5])
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,url,year,authors,externalIds,publicationTypes",
        "year":   f"{year_from}-",
    }

    logger.info(f"[SemanticScholar] Searching: {query[:60]}")
    resp = _safe_get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers={
            "User-Agent": "rd-engine-research/1.0",
        }
    )

    if not resp:
        return []

    try:
        data = resp.json()
        for paper in data.get("data", []):
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            url = (
                f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                else paper.get("url", "")
            )
            results.append({
                "title":     paper.get("title", ""),
                "abstract":  (paper.get("abstract") or "")[:800],
                "url":       url,
                "year":      paper.get("year"),
                "source":    "semantic_scholar",
            })
    except Exception as e:
        logger.error(f"[SemanticScholar] Parse failed: {e}")

    logger.info(f"[SemanticScholar] Retrieved {len(results)} papers")
    return results


# ── Google Patents ─────────────────────────────────────────────────────────────

def fetch_google_patents(
    keywords: list[str],
    assignees: list[str] = None,
    max_results: int = 10,
    status: str = "PENDING",  # PENDING = applications, GRANT = granted
) -> list[dict]:
    """
    Search Google Patents via their public search (no API key needed).
    Returns list of {title, abstract, url, assignee, filing_date}.

    Note: Google Patents doesn't have a formal free API.
    We use the public search URL which is rate-limited but functional.
    """
    results = []

    if assignees is None:
        assignees = ["NVIDIA", "TSMC", "AMD", "Intel Corporation", "Google LLC"]

    # Build assignee filter
    assignee_query = " OR ".join(f'(assignee:"{a}")' for a in assignees[:4])
    keyword_query  = " ".join(f'"{k}"' for k in keywords[:4])

    # Use CrossRef / USPTO patent search as fallback
    # Google Patents blocks automated scraping — use USPTO open API instead
    results = _fetch_uspto_patents(keywords, assignees, max_results)

    if not results:
        logger.info("[Patents] USPTO returned empty — LLM agent will use its own knowledge")

    return results


def _fetch_uspto_patents(
    keywords: list[str],
    assignees: list[str],
    max_results: int,
) -> list[dict]:
    """
    USPTO PatentsView API v1 — migrated from dead api.patentsview.org (410 Gone).
    New endpoint: https://search.patentsview.org/api/v1/patent/
    Free without API key; register at patentsview.org for higher rate limits.
    Optional env var: PATENTSVIEW_API_KEY
    """
    import os
    results = []

    keyword_str = " AND ".join(keywords[:4])
    two_years_ago = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    payload = {
        "q": {
            "_and": [
                {"_text_all": {"patent_title": keyword_str}},
                {"_gte": {"patent_date": two_years_ago}},
            ]
        },
        "f": [
            "patent_id", "patent_title", "patent_abstract",
            "patent_date", "assignees.assignee_organization",
        ],
        "s": [{"patent_date": "desc"}],
        "o": {"per_page": max_results},
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent":   "rd-engine-research/1.0",
    }
    api_key = os.environ.get("PATENTSVIEW_API_KEY", "")
    if api_key:
        headers["X-Api-Key"] = api_key

    try:
        resp = requests.post(
            "https://search.patentsview.org/api/v1/patent/",
            json=payload,
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        for p in data.get("patents", []):
            # v1 API nests assignees as a list of objects
            assignee_list = p.get("assignees") or []
            assignee = (
                assignee_list[0].get("assignee_organization", "")
                if assignee_list else ""
            )
            results.append({
                "title":       p.get("patent_title", ""),
                "abstract":    (p.get("patent_abstract") or "")[:600],
                "url":         f"https://patents.google.com/patent/US{p.get('patent_id', '')}",
                "assignee":    assignee,
                "filing_date": p.get("patent_date", ""),
                "source":      "patent",
            })

    except Exception as e:
        logger.warning(f"[Patents] USPTO API failed: {e}")

    return results


# ── Job Postings (intelligence signal) ────────────────────────────────────────

def fetch_job_postings_signals(
    companies: list[str],
    role_keywords: list[str],
) -> list[dict]:
    """
    Job posting signals from two free, open APIs:
      1. RemoteOK (https://remoteok.com/api) — JSON, no key, no scraping
      2. Hacker News "Who is Hiring" monthly thread via Algolia search API
         (https://hn.algolia.com/api/v1) — completely free, no key

    Indeed/LinkedIn are permanently blocked via Cloudflare. Dropped.
    """
    signals = []
    companies_lower = {c.lower(): c for c in companies}
    kw_lower        = {k.lower(): k for k in role_keywords}

    # ── Source 1: RemoteOK ────────────────────────────────────────────────────
    try:
        resp = _safe_get(
            "https://remoteok.com/api",
            headers={"User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
        )
        if resp:
            jobs = resp.json()
            if isinstance(jobs, list) and len(jobs) > 1:
                jobs = jobs[1:]  # first element is a metadata notice dict
                for job in jobs[:100]:
                    company_raw = (job.get("company") or "").lower()
                    position    = (job.get("position") or "").lower()
                    desc        = (job.get("description") or "").lower()
                    combined    = f"{company_raw} {position} {desc}"

                    matched_company = next(
                        (orig for low, orig in companies_lower.items() if low in company_raw), None
                    )
                    if not matched_company:
                        continue
                    matched_kw = next(
                        (orig for low, orig in kw_lower.items() if low in combined), None
                    )
                    if not matched_kw:
                        continue

                    signals.append({
                        "company":         matched_company,
                        "keyword":         matched_kw,
                        "signal_strength": 5,
                        "url":             job.get("url", "https://remoteok.com"),
                        "note":            f"RemoteOK: {job.get('position','')} @ {job.get('company','')}",
                    })
    except Exception as e:
        logger.debug(f"[Jobs] RemoteOK error: {e}")

    # ── Source 2: Hacker News "Who is Hiring" ─────────────────────────────────
    try:
        # Find latest monthly thread
        search_resp = _safe_get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "tags":        "story,ask_hn",
                "query":       "Ask HN: Who is hiring",
                "hitsPerPage": 1,
            },
        )
        if search_resp:
            hits = search_resp.json().get("hits", [])
            if hits:
                story_id = hits[0].get("objectID", "")
                comments_resp = _safe_get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "tags":        f"comment,story_{story_id}",
                        "hitsPerPage": 200,
                    },
                )
                if comments_resp:
                    for comment in comments_resp.json().get("hits", []):
                        text = (comment.get("comment_text") or "").lower()
                        for low_c, orig_c in companies_lower.items():
                            if low_c not in text:
                                continue
                            for low_k, orig_k in kw_lower.items():
                                if low_k in text:
                                    signals.append({
                                        "company":         orig_c,
                                        "keyword":         orig_k,
                                        "signal_strength": 7,  # HN signals are high-quality
                                        "url": f"https://news.ycombinator.com/item?id={comment.get('objectID','')}",
                                        "note":  f"HN Who Is Hiring: {orig_c} hiring for {orig_k}",
                                    })
                                    break  # one match per company per comment
    except Exception as e:
        logger.debug(f"[Jobs] HN error: {e}")

    if not signals:
        logger.info("[Jobs] No job posting signals retrieved — LLM will use its knowledge")

    return signals


# ── Orchestration helpers ──────────────────────────────────────────────────────

def fetch_all_for_cycle1(seed: dict) -> dict:
    """
    Fetch all data needed for Cycle 1 (Harvest).
    Returns dict with: papers, patents, job_signals, github_signals, rss_signals,
                       edgar_signals, darpa_signals.
    Called once at the start of Cycle 1.

    Sources (all free):
      1.  arXiv           — academic papers
      2.  Semantic Scholar — citations + broader coverage
      3.  USPTO Patents   — pending patents = unsolved problems
      4.  RemoteOK + HN   — job postings = where companies invest
      5.  GitHub Issues   — real engineer pain points (token optional)
      6.  HuggingFace     — daily trending AI papers
      7.  OpenReview      — NeurIPS/ICLR/ICML with reviewer criticism
      8.  OSTI/DOE        — DARPA-funded research, 2-3yr ahead of market
      9.  RSS feeds       — IEEE Spectrum, SemiAnalysis, EE Times, Tom's Hardware
      10. OCP GitHub      — Meta/Microsoft/Google/Amazon datacenter specs
      11. SEC EDGAR       — 10-K/10-Q Risk Factors = what giants admit is broken
      12. DARPA BAA       — gov funding signals = 2-3yr forward technology bets
    """
    import os
    keywords       = seed.get("seed_keywords", [])
    companies      = seed.get("target_companies", [])
    signals        = seed.get("intelligence_signals", [])
    github_token   = os.environ.get("GITHUB_TOKEN", "")

    logger.info("[Sources] Starting full fetch for Cycle 1 harvest — 12 sources")

    if not github_token:
        logger.warning(
            "[Sources] GITHUB_TOKEN not set — GitHub Issues and OCP signals will be empty. "
            "Add GITHUB_TOKEN secret to repository for full source coverage."
        )
    result = {
        "papers":         [],
        "patents":        [],
        "job_signals":    [],
        "github_signals": [],
        "rss_signals":    [],
        "edgar_signals":  [],   # NEW: SEC 10-K/10-Q risk factors
        "darpa_signals":  [],   # NEW: DARPA BAA funding opportunities
    }

    # 1. arXiv
    try:
        items = fetch_arxiv_papers(keywords, max_results=20, days_back=30)
        result["papers"].extend(items)
        logger.info(f"[Sources] arXiv: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] arXiv failed: {e}")

    # 2. Semantic Scholar
    try:
        items = fetch_semantic_scholar_papers(keywords, max_results=10)
        result["papers"].extend(items)
        logger.info(f"[Sources] Semantic Scholar: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] Semantic Scholar failed: {e}")

    # 3. HuggingFace daily papers
    try:
        items = fetch_huggingface_papers(max_results=15)
        result["papers"].extend(items)
        logger.info(f"[Sources] HuggingFace Papers: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] HuggingFace Papers failed: {e}")

    # 4. OpenReview (NeurIPS/ICLR)
    try:
        items = fetch_openreview_papers(keywords, max_results=10)
        result["papers"].extend(items)
        logger.info(f"[Sources] OpenReview: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] OpenReview failed: {e}")

    # 5. OSTI / DOE research
    try:
        items = fetch_osti_research(keywords, max_results=8)
        result["papers"].extend(items)
        logger.info(f"[Sources] OSTI/DOE: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] OSTI failed: {e}")

    # 6. Patents
    try:
        items = fetch_google_patents(keywords[:6], companies[:4], max_results=10)
        result["patents"] = items
        logger.info(f"[Sources] Patents: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] Patents failed: {e}")

    # 7. Job signals
    try:
        items = fetch_job_postings_signals(companies[:5], signals[:5])
        result["job_signals"] = items
        logger.info(f"[Sources] Job signals: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] Job signals failed: {e}")

    # 8. GitHub issues (real engineer pain points)
    try:
        items = fetch_github_signals(github_token=github_token, max_issues=30)
        result["github_signals"] = items
        logger.info(f"[Sources] GitHub: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] GitHub failed: {e}")

    # 9. RSS feeds (IEEE Spectrum, SemiAnalysis, EE Times, Tom's Hardware, TechPowerUp)
    try:
        items = fetch_rss_signals(max_per_feed=5)
        result["rss_signals"] = items
        logger.info(f"[Sources] RSS: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] RSS failed: {e}")

    # 10. Open Compute Project — real datacenter hardware specs (Meta/Microsoft/Google/Amazon)
    try:
        items = fetch_ocp_signals(github_token=github_token, max_items=20)
        result["github_signals"].extend(items)
        logger.info(f"[Sources] OCP: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] OCP failed: {e}")

    # 11. SEC EDGAR — 10-K/10-Q Risk Factors (what the giants admit is unsolved)
    try:
        items = fetch_sec_edgar_signals(keywords=keywords[:6], max_results=10, days_back=180)
        result["edgar_signals"] = items
        logger.info(f"[Sources] SEC EDGAR: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] SEC EDGAR failed: {e}")

    # 12. DARPA BAA — active funding solicitations (2-3yr forward signal)
    try:
        items = fetch_darpa_baa_signals(keywords=keywords[:6], max_results=8, days_back=365)
        result["darpa_signals"] = items
        logger.info(f"[Sources] DARPA BAA: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] DARPA failed: {e}")

    total = sum(len(v) for v in result.values())
    logger.info(f"[Sources] Total items fetched: {total} (12 sources)")
    return result


def fetch_papers_by_domain(domain: str, keywords: list[str], days_back: int = 60) -> list[dict]:
    """
    Domain-specific paper fetch. Used in focused cycles.
    """
    domain_keyword_map = {
        "thermal":      ["thermal resistance", "heat flux GPU", "junction temperature", "3D IC cooling"],
        "power":        ["power delivery network", "CMOS power density", "energy per operation", "VRM AI"],
        "data_movement": ["memory bandwidth", "roofline model", "HBM latency", "memory wall LLM"],
        "pdn":          ["PDN impedance", "IR drop chiplet", "decoupling capacitor AI", "power integrity"],
        "hardware":     ["AI accelerator", "chiplet integration", "advanced packaging", "3nm 2nm TSMC"],
    }

    all_keywords = list(set(keywords + domain_keyword_map.get(domain, [])))
    return fetch_arxiv_papers(all_keywords[:10], max_results=12, days_back=days_back)


# ══════════════════════════════════════════════════════════════════════════════
# NEW FREE SOURCES
# ══════════════════════════════════════════════════════════════════════════════

# ── GitHub API ────────────────────────────────────────────────────────────────

TARGET_REPOS = [
    "NVIDIA/cuda-samples", "NVIDIA/TensorRT", "NVIDIA/apex",
    "ROCm/ROCm", "AMD/amd-lab-notes",
    "openai/triton", "openai/transformer-debugger",
    "microsoft/DeepSpeed", "microsoft/onnxruntime",
    "intel/intel-extension-for-pytorch",
    "google/jax", "google-deepmind/gemma",
    "pytorch/pytorch", "vllm-project/vllm",
]

TARGET_GITHUB_ORGS = ["NVIDIA", "AMD", "intel", "google-deepmind", "openai", "microsoft"]


def fetch_github_signals(github_token: str = "", max_issues: int = 30) -> list[dict]:
    """
    Fetch GitHub signals: open issues + recent commits on target repos.
    github_token: optional, increases rate limit from 60→5000 req/hour.
    Returns list of {title, body, url, repo, signal_type, created_at}.
    """
    if not github_token:
        import os
        github_token = os.environ.get("GITHUB_TOKEN", "")

    headers = {"User-Agent": "rd-engine/1.0", "Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    results = []

    for repo in TARGET_REPOS[:8]:  # limit to avoid rate limits
        # Fetch open issues labeled as bugs or enhancements (= real pain points)
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": "open",
            "sort": "created",
            "direction": "desc",
            "per_page": 5,
            "labels": "bug,performance,enhancement",
        }
        resp = _safe_get(url, params=params, headers=headers)
        if not resp:
            continue
        try:
            issues = resp.json()
            if not isinstance(issues, list):
                continue
            for issue in issues[:3]:
                body = (issue.get("body") or "")[:400]
                if len(body) < 30:
                    continue  # skip empty issues
                results.append({
                    "title":       issue.get("title", ""),
                    "body":        body,
                    "url":         issue.get("html_url", ""),
                    "repo":        repo,
                    "signal_type": "github_issue",
                    "created_at":  issue.get("created_at", ""),
                    "comments":    issue.get("comments", 0),
                })
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[GitHub] Failed parsing {repo}: {e}")
            continue

    logger.info(f"[GitHub] Fetched {len(results)} issue signals")
    return results


# ── HuggingFace Daily Papers ──────────────────────────────────────────────────

def fetch_huggingface_papers(max_results: int = 15) -> list[dict]:
    """
    Fetch today's trending papers from HuggingFace Papers.
    No API key needed. These are community-curated — high signal.
    """
    resp = _safe_get(
        "https://huggingface.co/api/daily_papers",
        params={"limit": max_results},
    )
    if not resp:
        return []

    results = []
    try:
        data = resp.json()
        for item in data:
            paper = item.get("paper", {})
            title = paper.get("title", "")
            abstract = paper.get("summary", "")[:600]
            if not title or len(abstract) < 50:
                continue
            results.append({
                "title":     title,
                "abstract":  abstract,
                "url":       f"https://huggingface.co/papers/{paper.get('id', '')}",
                "authors":   [a.get("name", "") for a in paper.get("authors", [])[:3]],
                "published": item.get("publishedAt", ""),
                "source":    "huggingface_daily",
                "upvotes":   item.get("totalUpvotes", 0),
            })
    except Exception as e:
        logger.warning(f"[HuggingFace Papers] Parse error: {e}")

    logger.info(f"[HuggingFace Papers] Fetched {len(results)} papers")
    return results


# ── OpenReview API (NeurIPS / ICLR / ICML) ───────────────────────────────────

OPENREVIEW_VENUES = [
    "NeurIPS.cc/2024/Conference",
    "ICLR.cc/2025/Conference",
    "ICML.cc/2024/Conference",
]

def fetch_openreview_papers(keywords: list[str], max_results: int = 10) -> list[dict]:
    """
    Search OpenReview for papers matching keywords.
    Returns papers + reviewer criticism (= gaps = opportunities).
    No API key needed.
    """
    results = []
    query = " ".join(keywords[:5])

    for venue in OPENREVIEW_VENUES[:2]:
        resp = _safe_get(
            "https://api2.openreview.net/notes",
            params={
                "content.venue": venue,
                "term":          query,
                "limit":         max_results // 2,
                "offset":        0,
            },
        )
        if not resp:
            continue
        try:
            data = resp.json()
            notes = data.get("notes", [])
            for note in notes:
                content = note.get("content", {})
                title   = content.get("title", {})
                title   = title.get("value", title) if isinstance(title, dict) else title
                abstract = content.get("abstract", {})
                abstract = abstract.get("value", abstract) if isinstance(abstract, dict) else abstract
                abstract = str(abstract)[:500]
                if not title or len(abstract) < 30:
                    continue
                results.append({
                    "title":    str(title),
                    "abstract": abstract,
                    "url":      f"https://openreview.net/forum?id={note.get('id','')}",
                    "venue":    venue.split("/")[0],
                    "source":   "openreview",
                })
            time.sleep(1)
        except Exception as e:
            logger.warning(f"[OpenReview] Parse error for {venue}: {e}")

    logger.info(f"[OpenReview] Fetched {len(results)} papers")
    return results


# ── OSTI.gov (US Dept of Energy / DARPA-funded research) ─────────────────────

def fetch_osti_research(keywords: list[str], max_results: int = 10) -> list[dict]:
    """
    Fetch DOE/DARPA-funded research from OSTI.gov.
    Precedes commercial applications by 2-3 years. Free, no key.
    """
    query = " ".join(keywords[:6])
    resp = _safe_get(
        "https://www.osti.gov/api/v1/records",
        params={
            "q":          query,
            "sort":       "publication_date desc",
            "page_size":  max_results,
            "fields":     "title,description,doi,publication_date,site_url",
        },
        headers={"Accept": "application/json", "User-Agent": "rd-engine/1.0"},
    )
    if not resp:
        return []

    results = []
    try:
        data = resp.json()
        records = data if isinstance(data, list) else data.get("records", [])
        for rec in records[:max_results]:
            title = rec.get("title", "")
            desc  = (rec.get("description") or "")[:500]
            if not title:
                continue
            results.append({
                "title":     title,
                "abstract":  desc,
                "url":       rec.get("site_url", f"https://doi.org/{rec.get('doi','')}"),
                "published": rec.get("publication_date", ""),
                "source":    "osti_doe",
            })
    except Exception as e:
        logger.warning(f"[OSTI] Parse error: {e}")

    logger.info(f"[OSTI] Fetched {len(results)} DOE records")
    return results


# ── RSS Feeds (IEEE Spectrum, SemiAnalysis, EE Times) ────────────────────────

RSS_FEEDS = [
    ("https://spectrum.ieee.org/feeds/feed.rss",            "ieee_spectrum"),
    ("https://www.eetimes.com/feed/",                       "ee_times"),
    ("https://semianalysis.com/feed/",                      "semianalysis"),
    # AnandTech was archived/shut down in 2024 — replaced with active feeds:
    ("https://www.tomshardware.com/feeds/all",              "tomshardware"),
    ("https://www.techpowerup.com/rss/news.xml",            "techpowerup"),
]

RELEVANT_KEYWORDS = [
    "nvidia", "tsmc", "amd", "intel", "gpu", "hbm", "packaging",
    "thermal", "power", "chiplet", "ai accelerator", "semiconductor",
    "3nm", "2nm", "cowos", "nvlink", "pdn", "bandwidth",
]


def _parse_rss_items(xml_text: str, source_name: str) -> list:
    """
    Two-stage XML parser:
      1. stdlib xml.etree.ElementTree — fast, strict
      2. Fallback: re-encode bytes stripping illegal XML chars, try again
      3. Last resort: regex scrape of <title> and <link> tags (no deps)
    This handles malformed feeds (bare &, invalid chars, encoding issues)
    without requiring BeautifulSoup/lxml as a new dependency.
    """
    import xml.etree.ElementTree as ET
    import re

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    def _extract_items(root):
        return root.findall(".//item") or root.findall(".//atom:entry", ns)

    def _get_field(item, *tags):
        for tag in tags:
            if "atom:" in tag:
                val = item.findtext(tag, namespaces=ns)
            else:
                val = item.findtext(tag)
            if val:
                return val.strip()
        return ""

    def _get_link(item):
        lnk = _get_field(item, "link")
        if not lnk:
            atom_link = item.find("atom:link", ns)
            if atom_link is not None:
                lnk = atom_link.get("href", "")
        return lnk

    # Stage 1 — strict parse
    try:
        root = ET.fromstring(xml_text)
        return [(item, _get_field, _get_link) for item in _extract_items(root)]
    except ET.ParseError:
        pass

    # Stage 2 — strip illegal XML chars (control chars + bare & not in entity)
    try:
        # Replace bare & that aren't part of &amp; &lt; etc.
        cleaned = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', xml_text)
        # Strip non-XML characters (ASCII control chars except tab/LF/CR)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        root = ET.fromstring(cleaned)
        return [(item, _get_field, _get_link) for item in _extract_items(root)]
    except ET.ParseError as e:
        logger.debug(f"[RSS] {source_name} still invalid after cleanup: {e} — using regex fallback")

    # Stage 3 — regex scrape (zero extra deps, handles completely broken XML)
    items_out = []
    titles = re.findall(r'<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', xml_text, re.S)
    links  = re.findall(r'<link[^>]*>(https?://[^<]+)</link>', xml_text)
    descs  = re.findall(r'<description[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>', xml_text, re.S)
    titles = titles[1:]  # skip feed-level <title>
    for i, title in enumerate(titles):
        link = links[i] if i < len(links) else ""
        desc = descs[i] if i < len(descs) else ""
        items_out.append({
            "title": title.strip(),
            "desc":  desc.strip()[:400],
            "link":  link.strip(),
        })
    return items_out  # note: different tuple format — handled below


def fetch_rss_signals(max_per_feed: int = 5) -> list[dict]:
    """
    Parse RSS feeds. Robust against malformed XML (AnandTech-style invalid tokens).
    Falls back through three parsing stages before giving up on a feed.
    No new dependencies required.
    """
    results = []

    for feed_url, source_name in RSS_FEEDS:
        resp = _safe_get(feed_url)
        if not resp:
            continue
        try:
            parsed = _parse_rss_items(resp.text, source_name)
            count  = 0

            for item in parsed:
                # Handle both ET-tuple format and regex-dict format
                if isinstance(item, dict):
                    title    = item["title"]
                    desc     = item["desc"]
                    link     = item["link"]
                else:
                    et_item, _get_field, _get_link = item
                    title = _get_field(et_item, "title", "atom:title")
                    desc  = _get_field(
                        et_item, "description", "summary", "atom:summary"
                    )[:400]
                    link  = _get_link(et_item)

                combined = (title + " " + desc).lower()
                if not any(kw in combined for kw in RELEVANT_KEYWORDS):
                    continue

                results.append({
                    "title":       title,
                    "abstract":    desc,
                    "url":         link,
                    "source":      source_name,
                    "signal_type": "industry_news",
                })
                count += 1
                if count >= max_per_feed:
                    break

            logger.info(f"[RSS] {source_name}: {count} relevant items")
            time.sleep(1)

        except Exception as e:
            logger.warning(f"[RSS] Failed parsing {source_name}: {e}")

    logger.info(f"[RSS] Total: {len(results)} signals")
    return results


# ── Open Compute Project ──────────────────────────────────────────────────────
# OCP GitHub: github.com/opencomputeproject — specs from Meta/Microsoft/Google/Amazon
# Focus: hardware specs for AI datacenter — thermal, power, rack, networking

OCP_REPOS = [
    "opencomputeproject/OCP-Profiles",
    "opencomputeproject/Open-Rack",         # was: OpenRack — 404
    "opencomputeproject/Project_Olympus",
    "opencomputeproject/Cerberus",          # was: Project_Cerberus — 404
    "opencomputeproject/ocp-diag-core",
]

OCP_RELEVANT_KEYWORDS = [
    "thermal", "power", "cooling", "pdu", "rack", "voltage",
    "bandwidth", "accelerator", "gpu", "ai", "hbm", "packaging",
    "efficiency", "watt", "heat", "temperature",
]


def fetch_ocp_signals(github_token: str = "", max_items: int = 20) -> list[dict]:
    """
    Fetch Open Compute Project specs and issues from GitHub.
    OCP = Meta/Microsoft/Google/Amazon publish real datacenter hardware specs here.
    404s are silently skipped (private or renamed repos — not an error).
    """
    import os
    if not github_token:
        github_token = os.environ.get("GITHUB_TOKEN", "")

    headers = {"User-Agent": "rd-engine/1.0", "Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    results = []

    for repo in OCP_REPOS:
        # 1. Fetch recent issues — real spec discussions
        try:
            resp = _safe_get(
                f"https://api.github.com/repos/{repo}/issues",
                params={"state": "open", "sort": "created", "direction": "desc", "per_page": 5},
                headers=headers,
            )
            if resp and resp.status_code == 200:
                issues = resp.json()
                if isinstance(issues, list):
                    for issue in issues[:3]:
                        body = (issue.get("body") or "")[:400]
                        combined = (issue.get("title", "") + " " + body).lower()
                        if not any(kw in combined for kw in OCP_RELEVANT_KEYWORDS):
                            continue
                        results.append({
                            "title":       issue.get("title", ""),
                            "body":        body,
                            "url":         issue.get("html_url", ""),
                            "repo":        repo,
                            "signal_type": "ocp_spec_discussion",
                            "created_at":  issue.get("created_at", ""),
                        })
        except Exception as e:
            logger.debug(f"[OCP] Issues error {repo}: {e}")  # debug — not a warning

        time.sleep(0.5)

        # 2. Fetch recent releases — new hardware specs published
        try:
            resp = _safe_get(
                f"https://api.github.com/repos/{repo}/releases",
                params={"per_page": 3},
                headers=headers,
            )
            if resp and resp.status_code == 200:
                releases = resp.json()
                if isinstance(releases, list):
                    for rel in releases[:2]:
                        body = (rel.get("body") or "")[:400]
                        combined = (rel.get("name", "") + " " + body).lower()
                        if not any(kw in combined for kw in OCP_RELEVANT_KEYWORDS):
                            continue
                        results.append({
                            "title":       f"[OCP SPEC] {rel.get('name', '')}",
                            "body":        body,
                            "url":         rel.get("html_url", ""),
                            "repo":        repo,
                            "signal_type": "ocp_spec_release",
                            "created_at":  rel.get("published_at", ""),
                        })
        except Exception as e:
            logger.debug(f"[OCP] Releases error {repo}: {e}")  # debug — not a warning

        time.sleep(0.5)

    logger.info(f"[OCP] Fetched {len(results)} OCP signals")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR + DARPA BAA — אותות אינטליגנציה עסקית עמוקים
# ══════════════════════════════════════════════════════════════════════════════

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
# NVIDIA, TSMC, AMD, Intel חייבות לפרט בדוחות 10-K ו-10-Q שלהן בדיוק
# אילו בעיות טכניות הן מתמודדות איתן (Risk Factors + MD&A).
# זה מידע גולמי ישיר מהחברות עצמן — לא אנליסטים, לא עיתונות.
# API: EDGAR Full-Text Search — חינמי, ללא key, ממשלתי.

SEC_TARGET_COMPANIES = [
    ("NVIDIA CORPORATION",    "nvda"),
    ("TAIWAN SEMICONDUCTOR",  "tsm"),
    ("ADVANCED MICRO DEVICES","amd"),
    ("INTEL CORPORATION",     "intc"),
    ("GOOGLE LLC",            "googl"),
    ("MICROSOFT CORPORATION", "msft"),
]

SEC_RISK_KEYWORDS = [
    "thermal", "power consumption", "yield", "packaging",
    "bandwidth", "memory", "bottleneck", "constraint",
    "supply chain", "manufacturing", "chip", "semiconductor",
]


def fetch_sec_edgar_signals(
    keywords: list[str] = None,
    max_results: int = 10,
    days_back: int = 180,
) -> list[dict]:
    """
    מחפש ב-EDGAR Full-Text Search דוחות 10-K ו-10-Q של חברות יעד.
    מחזיר קטעים שמזכירים צווארי בקבוק טכניים — Risk Factors ו-MD&A.
    חינמי לחלוטין, ללא API key, ללא rate limit משמעותי.
    https://efts.sec.gov/LATEST/search-index (ממשלת ארה"ב)
    """
    results = []
    search_keywords = keywords or SEC_RISK_KEYWORDS
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # בנה query — שילוב של חברת יעד + מילת מפתח טכנית
    for company_name, ticker in SEC_TARGET_COMPANIES[:5]:
        for kw in search_keywords[:4]:
            query = f'"{ticker}" "{kw}"'
            try:
                resp = _safe_get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q":           query,
                        "forms":       "10-K,10-Q",
                        "dateRange":   "custom",
                        "startdt":     since_date,
                        "hits.hits.total.value": 1,
                    },
                    headers={
                        "User-Agent":  "rd-engine-research/1.0 research@example.com",
                        "Accept":      "application/json",
                    },
                )
                if not resp:
                    continue

                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])

                for hit in hits[:2]:  # מקסימום 2 דוחות לכל שילוב
                    src = hit.get("_source", {})
                    # חלץ את הקטע הרלוונטי מתוך הדוח
                    highlight = hit.get("highlight", {})
                    snippets  = []
                    for field_hits in highlight.values():
                        snippets.extend(field_hits[:2])
                    snippet = " [...] ".join(snippets)[:600] if snippets else ""

                    filing_date = src.get("file_date", "")
                    form_type   = src.get("form_type", "")
                    entity      = src.get("display_names", [{}])
                    entity_name = entity[0].get("name", company_name) if entity else company_name
                    accession   = src.get("accession_no", "").replace("-", "")
                    filing_url  = (
                        f"https://www.sec.gov/Archives/edgar/data/"
                        f"{src.get('entity_id','')}/{accession}/"
                        if accession else "https://www.sec.gov/cgi-bin/browse-edgar"
                    )

                    if not snippet and not form_type:
                        continue

                    results.append({
                        "title":       f"[{form_type}] {entity_name} — {kw}",
                        "abstract":    snippet or f"{entity_name} mentions '{kw}' in {form_type} filing.",
                        "url":         filing_url,
                        "company":     entity_name,
                        "keyword":     kw,
                        "filing_date": filing_date,
                        "form_type":   form_type,
                        "source":      "sec_edgar",
                        "signal_type": "regulatory_filing",
                    })

                time.sleep(0.3)  # EDGAR לא דורש delay, אבל נהיה מנומסים

            except Exception as e:
                logger.debug(f"[EDGAR] Query failed ({company_name} + {kw}): {e}")
                continue

            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    logger.info(f"[EDGAR] Fetched {len(results)} SEC filing signals")
    return results


# ── DARPA BAA ─────────────────────────────────────────────────────────────────
# DARPA מפרסמת Broad Agency Announcements — רשימת הבעיות הטכניות הכי קשות
# שהממשלה האמריקאית מוכנה לממן פתרונות עבורן.
# זה 2-3 שנים לפני השוק — מוקדם יותר מ-OSTI.
# API: SAM.gov — מערכת רכש פדרלית. DEMO_KEY חינמי (1000 req/יום).

SAM_DEMO_KEY = "DEMO_KEY"  # עובד ללא רישום עד 1000 בקשות ביום
                            # לרישום חינמי לקיבולת גבוהה יותר: sam.gov

DARPA_RELEVANT_KEYWORDS = [
    "semiconductor", "computing", "microelectronics", "photonics",
    "thermal", "power", "memory", "bandwidth", "processor",
    "packaging", "chiplet", "accelerator", "neural", "ai",
]


def fetch_darpa_baa_signals(
    keywords: list[str] = None,
    max_results: int = 10,
    days_back: int = 365,
) -> list[dict]:
    """
    מאחזר BAA (Broad Agency Announcements) של DARPA מ-SAM.gov.
    BAA = בעיות שהממשלה האמריקאית משלמת לפתור — אינטל 2-3 שנים קדימה.
    DEMO_KEY חינמי (sam.gov) — ללא רישום עד 1000 req/יום.
    לקיבולת גבוהה יותר: הירשם ב-sam.gov וקבל key חינמי אמיתי.
    Optional env var: SAM_GOV_API_KEY
    """
    import os
    results  = []
    api_key  = os.environ.get("SAM_GOV_API_KEY", SAM_DEMO_KEY)
    kw_filter = [k.lower() for k in (keywords or DARPA_RELEVANT_KEYWORDS)]
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%m/%d/%Y")
    today_str  = datetime.now().strftime("%m/%d/%Y")

    try:
        resp = _safe_get(
            "https://api.sam.gov/opportunities/v2/search",
            params={
                "api_key":      api_key,
                "ptype":        "o",           # o = solicitation (BAA נכלל כאן)
                "deptname":     "Defense Advanced Research Projects Agency",
                "postedFrom":   since_date,
                "postedTo":     today_str,
                "limit":        max_results * 2,  # נסנן אחר כך
                "offset":       0,
            },
            headers={"User-Agent": "rd-engine-research/1.0"},
        )

        if not resp:
            return []

        data = resp.json()
        opportunities = data.get("opportunitiesData", [])

        for opp in opportunities:
            title       = opp.get("title", "")
            description = (opp.get("description") or "")[:600]
            sol_number  = opp.get("solicitationNumber", "")
            posted_date = opp.get("postedDate", "")
            opp_type    = opp.get("typeOfSetAside", "") or opp.get("type", "")
            ui_link     = opp.get("uiLink", "")
            if not ui_link:
                ui_link = f"https://sam.gov/opp/{opp.get('noticeId','')}/view"

            # סנן רק הזדמנויות שרלוונטיות לתחום
            combined = (title + " " + description).lower()
            if not any(kw in combined for kw in kw_filter):
                continue

            results.append({
                "title":         f"[DARPA BAA] {title}",
                "abstract":      description or f"DARPA solicitation: {title}",
                "url":           ui_link,
                "sol_number":    sol_number,
                "posted_date":   posted_date,
                "opp_type":      opp_type,
                "source":        "darpa_baa",
                "signal_type":   "government_funding_signal",
            })

            if len(results) >= max_results:
                break

    except Exception as e:
        logger.warning(f"[DARPA] SAM.gov fetch failed: {e}")

    logger.info(f"[DARPA] Fetched {len(results)} DARPA BAA signals")
    return results
