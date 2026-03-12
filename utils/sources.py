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
    USPTO PatentsView API — actually has a free REST API.
    https://patentsview.org/apis/purpose
    """
    results = []

    keyword_str = " AND ".join(keywords[:4])
    assignee_queries = [{"_contains": {"assignee_organization": a}} for a in assignees[:3]]

    payload = {
        "q": {
            "_and": [
                {"_text_all": {"patent_abstract": keyword_str}},
                {"_gte": {"patent_date": (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")}},
            ]
        },
        "f": ["patent_number", "patent_title", "patent_abstract", "patent_date", "assignee_organization"],
        "o": {"per_page": max_results, "sort": [{"patent_date": "desc"}]},
    }

    resp = _safe_get(
        "https://api.patentsview.org/patents/query",
        params={"q": str(payload["q"]), "f": str(payload["f"]), "o": str(payload["o"])},
    )

    # PatentsView requires POST with JSON — use requests.post directly
    try:
        resp = requests.post(
            "https://api.patentsview.org/patents/query",
            json=payload,
            headers={"User-Agent": "rd-engine-research/1.0"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        for p in data.get("patents", []):
            results.append({
                "title":       p.get("patent_title", ""),
                "abstract":    (p.get("patent_abstract") or "")[:600],
                "url":         f"https://patents.google.com/patent/US{p.get('patent_number')}",
                "assignee":    p.get("assignee_organization", ""),
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
    Fetch job posting signals from LinkedIn/Indeed public search.

    Reality check: LinkedIn blocks scrapers aggressively.
    We use a lightweight approach: check company engineering blogs and public job APIs.
    Returns signals — not full job descriptions.
    """
    signals = []

    # Try Indeed's publicly accessible job search (limited but real)
    for company in companies[:5]:
        for keyword in role_keywords[:3]:
            query = f"{company} {keyword} engineer"
            resp = _safe_get(
                "https://www.indeed.com/jobs",
                params={"q": query, "l": "", "sort": "date"},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
                    "Accept": "text/html",
                }
            )

            if resp and resp.status_code == 200:
                # Simple count from HTML (not full parsing)
                count_signal = resp.text.count(keyword.lower())
                if count_signal > 5:
                    signals.append({
                        "company": company,
                        "keyword": keyword,
                        "signal_strength": min(count_signal // 5, 10),
                        "url": resp.url,
                        "note": f"Found {count_signal} mentions of '{keyword}' in {company} job listings",
                    })

            time.sleep(1.0)  # Be polite

    if not signals:
        logger.info("[Jobs] No job posting signals retrieved — LLM will use its knowledge")

    return signals


# ── Orchestration helpers ──────────────────────────────────────────────────────

def fetch_all_for_cycle1(seed: dict) -> dict:
    """
    Fetch all data needed for Cycle 1 (Harvest).
    Returns dict with: papers, patents, job_signals, github_signals, rss_signals.
    Called once at the start of Cycle 1.

    Sources (all free):
      1. arXiv          — academic papers
      2. Semantic Scholar — citations + broader coverage
      3. USPTO Patents  — pending patents = unsolved problems
      4. Indeed         — job postings = where companies invest
      5. GitHub Issues  — real engineer pain points (token optional)
      6. HuggingFace    — daily trending AI papers
      7. OpenReview     — NeurIPS/ICLR/ICML with reviewer criticism
      8. OSTI/DOE       — DARPA-funded research, 2-3yr ahead of market
      9. RSS feeds      — IEEE Spectrum, SemiAnalysis, EE Times, AnandTech
    """
    import os
    keywords       = seed.get("seed_keywords", [])
    companies      = seed.get("target_companies", [])
    signals        = seed.get("intelligence_signals", [])
    github_token   = os.environ.get("GITHUB_TOKEN", "")

    logger.info("[Sources] Starting full fetch for Cycle 1 harvest — 9 sources")

    # ── Warn explicitly about optional keys so failures are never silent ──────
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

    # 9. RSS feeds (IEEE Spectrum, SemiAnalysis, EE Times, AnandTech)
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

    total = sum(len(v) for v in result.values())
    logger.info(f"[Sources] Total items fetched: {total} (10 sources)")
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
    ("https://spectrum.ieee.org/feeds/feed.rss",        "ieee_spectrum"),
    ("https://www.eetimes.com/feed/",                   "ee_times"),
    ("https://semianalysis.com/feed/",                   "semianalysis"),
    ("https://www.anandtech.com/rss/",                   "anandtech"),
]

RELEVANT_KEYWORDS = [
    "nvidia", "tsmc", "amd", "intel", "gpu", "hbm", "packaging",
    "thermal", "power", "chiplet", "ai accelerator", "semiconductor",
    "3nm", "2nm", "cowos", "nvlink", "pdn", "bandwidth",
]

def fetch_rss_signals(max_per_feed: int = 5) -> list[dict]:
    """
    Parse RSS feeds from IEEE Spectrum, EE Times, SemiAnalysis, AnandTech.
    Filter for relevant keywords. No API key needed.
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        return []

    results = []

    for feed_url, source_name in RSS_FEEDS:
        resp = _safe_get(feed_url)
        if not resp:
            continue
        try:
            root = ET.fromstring(resp.text)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}

            # Handle both RSS 2.0 and Atom formats
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)

            count = 0
            for item in items:
                title = (
                    (item.findtext("title") or item.findtext("atom:title", namespaces=ns) or "")
                    .strip()
                )
                desc = (
                    (item.findtext("description") or
                     item.findtext("summary") or
                     item.findtext("atom:summary", namespaces=ns) or "")
                    [:400].strip()
                )
                link = (
                    item.findtext("link") or
                    (item.find("atom:link", ns).get("href", "") if item.find("atom:link", ns) is not None else "")
                )

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
    "opencomputeproject/OpenRack",
    "opencomputeproject/Project_Olympus",
    "opencomputeproject/Project_Cerberus",
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
    This is ground truth for what thermal/power constraints look like at scale.

    Returns list of {title, body, url, repo, signal_type}.
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
        resp = _safe_get(
            f"https://api.github.com/repos/{repo}/issues",
            params={"state": "open", "sort": "created", "direction": "desc", "per_page": 5},
            headers=headers,
        )
        if resp:
            try:
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
                logger.warning(f"[OCP] Issues parse error {repo}: {e}")

        time.sleep(0.5)

        # 2. Fetch recent releases — new hardware specs published
        resp = _safe_get(
            f"https://api.github.com/repos/{repo}/releases",
            params={"per_page": 3},
            headers=headers,
        )
        if resp:
            try:
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
                logger.warning(f"[OCP] Releases parse error {repo}: {e}")

        time.sleep(0.5)

    logger.info(f"[OCP] Fetched {len(results)} OCP signals")
    return results
