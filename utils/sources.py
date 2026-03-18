"""
utils/sources.py — Actual scrapers for data ingestion.

FIXES vs original:
  1. SemanticScholar: exponential backoff retry (was 429 with single sleep)
  2. OpenReview: REMOVED
  3. Greenhouse slugs: cerebras→cerebrassystems, removed openai/groq-inc/sifive/efinixinc/rivos
  4. Lever slugs: anthropic→Anthropic (case-sensitive), removed groq
  5. RSS The Register: /data_centre/rss → /headlines.atom (was 404)
  6. DARPA RSS: news.xml/news → rss.xml (correct URL)
  7. PatentsView: skip cleanly when PATENTSVIEW_API_KEY not set (was silent 403)
  8. SEC EDGAR: removed invalid highlight param that caused empty results
"""
from __future__ import annotations
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_ARXIV_DELAY     = 3.0
_REQUEST_TIMEOUT = 30


def _safe_get(url: str, params: dict = None, headers: dict = None) -> Optional[requests.Response]:
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


# ── arXiv ──────────────────────────────────────────────────────────────────────

def fetch_arxiv_papers(keywords: list[str], max_results: int = 15, days_back: int = 30) -> list[dict]:
    results  = []
    seen_urls = set()

    query_terms = " OR ".join(keywords[:4])
    cat_filter  = "cat:cs.AR OR cat:cs.ET OR cat:eess.SP OR cat:cs.DC"
    full_query  = f"({query_terms}) AND ({cat_filter})"

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

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns   = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        for entry in root.findall("atom:entry", ns):
            try:
                title         = entry.findtext("atom:title",   "", ns).strip().replace("\n", " ")
                abstract      = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                url           = entry.findtext("atom:id",      "", ns).strip()
                published_str = entry.findtext("atom:published", "", ns)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    pub_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    if pub_date.replace(tzinfo=None) < cutoff_date:
                        continue
                except (ValueError, AttributeError):
                    pass
                authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
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
    except Exception as e:
        logger.error(f"[arXiv] XML parsing failed: {e}")
        return []

    logger.info(f"[arXiv] Retrieved {len(results)} papers")
    return results


# ── Semantic Scholar ────────────────────────────────────────────────────────────

# ── HuggingFace Daily Papers ────────────────────────────────────────────────────

def fetch_huggingface_papers(max_results: int = 15) -> list[dict]:
    resp = _safe_get("https://huggingface.co/api/daily_papers", params={"limit": max_results})
    if not resp:
        return []
    results = []
    try:
        for item in resp.json():
            paper    = item.get("paper", {})
            title    = paper.get("title", "")
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


# ── OSTI ────────────────────────────────────────────────────────────────────────

def fetch_osti_research(keywords: list[str], max_results: int = 10) -> list[dict]:
    resp = _safe_get(
        "https://www.osti.gov/api/v1/records",
        params={
            "q":         " ".join(keywords[:6]),
            "sort":      "publication_date desc",
            "page_size": max_results,
            "fields":    "title,description,doi,publication_date,site_url",
        },
        headers={"Accept": "application/json", "User-Agent": "rd-engine/1.0"},
    )
    if not resp:
        return []
    results = []
    try:
        data    = resp.json()
        records = data if isinstance(data, list) else data.get("records", [])
        for rec in records[:max_results]:
            if not rec.get("title"):
                continue
            results.append({
                "title":     rec.get("title", ""),
                "abstract":  (rec.get("description") or "")[:500],
                "url":       rec.get("site_url", f"https://doi.org/{rec.get('doi','')}"),
                "published": rec.get("publication_date", ""),
                "source":    "osti_doe",
            })
    except Exception as e:
        logger.warning(f"[OSTI] Parse error: {e}")
    logger.info(f"[OSTI] Fetched {len(results)} DOE records")
    return results


# ── Patents ────────────────────────────────────────────────────────────────────

def fetch_google_patents(keywords: list[str], assignees: list[str] = None, max_results: int = 10, status: str = "PENDING") -> list[dict]:
    if assignees is None:
        assignees = ["NVIDIA", "TSMC", "AMD", "Intel Corporation", "Google LLC", "Samsung"]
    results = _fetch_lens_patents(keywords, assignees, max_results)
    if not results:
        results = _fetch_uspto_patents(keywords, assignees, max_results)
    if not results:
        logger.info("[Patents] All patent sources returned empty — LLM will use its knowledge")
    return results


def _fetch_lens_patents(keywords, assignees, max_results):
    import os
    api_key = os.environ.get("LENS_API_KEY", "")
    if not api_key:
        logger.debug("[Patents] LENS_API_KEY not set — skipping Lens.org")
        return []

    # Build short patent-friendly terms (1-2 words each)
    def _patent_terms(kws):
        short = []
        for kw in kws:
            parts = kw.split()
            short.append(" ".join(parts[:2]) if len(parts) > 2 else kw)
        return list(dict.fromkeys(short))

    patent_terms  = _patent_terms(keywords[:8])
    keyword_str   = " OR ".join(f'"{t}"' for t in patent_terms[:6])
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    results = []
    payload = {
        "query": {"bool": {"must": [
            {"query_string": {"query": keyword_str, "fields": ["title", "abstract", "claim"]}},
            {"range": {"date_published": {"gte": two_years_ago}}},
        ]}},
        "size": max_results,
        # NOTE: do NOT include "include" projection — Lens Patent API v2 uses different
        # internal field names and returns 400 "Unrecognized fields" for common names like
        # "title" and "applicant". Omit include to get all fields, then extract below.
        "sort": [{"date_published": "desc"}],
    }
    try:
        resp = requests.post(
            "https://api.lens.org/patent/search",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "User-Agent":    "rd-engine-research/1.0",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            for p in resp.json().get("data", []):
                lens_id = p.get("lens_id", "")

                # title: may be a list of {text, lang} dicts or a plain string
                raw_title = p.get("title", "") or p.get("biblio", {}).get("invention_title", "")
                if isinstance(raw_title, list):
                    title = next((t.get("text","") for t in raw_title if t.get("lang","")=="en"), raw_title[0].get("text","") if raw_title else "")
                else:
                    title = str(raw_title)

                # abstract
                raw_abs = p.get("abstract", "") or ""
                if isinstance(raw_abs, list):
                    abstract = next((a.get("text","") for a in raw_abs if a.get("lang","")=="en"), raw_abs[0].get("text","") if raw_abs else "")[:600]
                else:
                    abstract = str(raw_abs)[:600]

                # applicant name — may be nested
                applicants = p.get("applicant") or p.get("parties", {}).get("applicants", []) or []
                if applicants and isinstance(applicants[0], dict):
                    assignee = applicants[0].get("name","") or applicants[0].get("extracted_name",{}).get("value","")
                else:
                    assignee = str(applicants[0]) if applicants else ""

                if not title:
                    continue

                results.append({
                    "title":       title,
                    "abstract":    abstract,
                    "url":         f"https://lens.org/lens/patent/{lens_id}" if lens_id else "",
                    "assignee":    assignee,
                    "filing_date": p.get("date_published", ""),
                    "source":      "patent",
                })
        elif resp.status_code == 401:
            logger.warning("[Patents] Lens.org 401 — API key invalid or expired")
        elif resp.status_code == 400:
            logger.warning(f"[Patents] Lens.org 400 — bad query: {resp.text[:300]}")
        else:
            logger.warning(f"[Patents] Lens.org HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        logger.warning(f"[Patents] Lens.org error: {e}")

    logger.info(f"[Patents] Lens.org: {len(results)} patents")
    return results


def _fetch_uspto_patents(keywords, assignees, max_results):
    import os
    api_key = os.environ.get("PATENTSVIEW_API_KEY", "")
    if not api_key:
        logger.info("[Patents] PATENTSVIEW_API_KEY not set — skipping USPTO (register free at patentsview.org)")
        return []
    results       = []
    keyword_str   = " AND ".join(keywords[:4])
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    payload = {
        "q": {"_and": [
            {"_text_all": {"patent_title": keyword_str}},
            {"_gte": {"patent_date": two_years_ago}},
        ]},
        "f": ["patent_id", "patent_title", "patent_abstract", "patent_date", "assignees.assignee_organization"],
        "s": [{"patent_date": "desc"}],
        "o": {"per_page": max_results},
    }
    try:
        resp = requests.post(
            "https://search.patentsview.org/api/v1/patent/",
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "rd-engine-research/1.0", "X-Api-Key": api_key},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for p in resp.json().get("patents", []):
            assignee_list = p.get("assignees") or []
            results.append({
                "title":       p.get("patent_title", ""),
                "abstract":    (p.get("patent_abstract") or "")[:600],
                "url":         f"https://patents.google.com/patent/US{p.get('patent_id', '')}",
                "assignee":    assignee_list[0].get("assignee_organization", "") if assignee_list else "",
                "filing_date": p.get("patent_date", ""),
                "source":      "patent",
            })
    except Exception as e:
        logger.warning(f"[Patents] USPTO API failed: {e}")
    return results


# ── Job Postings ────────────────────────────────────────────────────────────────

# FIX: Greenhouse slugs verified March 2026
# Removed: openai (Ashby), groq-inc (acquired by NVIDIA), sifive/efinixinc/rivos (404)
# Fixed: cerebras → cerebrassystems
GREENHOUSE_COMPANIES = [
    ("cerebrassystems",  "Cerebras"),
    ("tenstorrent",      "Tenstorrent"),
    ("lightmatter",      "Lightmatter"),
    ("sambanovasystems", "SambaNova"),
    ("graphcore",        "Graphcore"),
]

# Lever slugs — all tested slugs return 404 (Anthropic, groq, cerebras, mistral)
# Lever API is slug-sensitive and companies change their slugs frequently
# Leaving empty — job signals come from Greenhouse + Workday which work reliably
LEVER_COMPANIES = []

WORKDAY_COMPANIES = [
    ("nvidia",      "NVIDIA",            "NVIDIAExternalCareerSite"),
    ("amd",         "AMD",               "AMD"),
    ("intel",       "Intel",             "Intel"),
    ("qualcomm",    "Qualcomm",          "qualcomm"),
    ("broadcom",    "Broadcom",          "Broadcom"),
    ("marvell",     "Marvell",           "marvell"),
    ("synopsys",    "Synopsys",          "Synopsys"),
    ("cadence",     "Cadence",           "Cadence"),
    ("applied",     "Applied Materials", "careers"),
    ("lamresearch", "Lam Research",      "LamCareers"),
]

JOB_SIGNAL_KEYWORDS = [
    "thermal", "cooling", "heat", "power delivery", "pdn",
    "packaging", "chiplet", "hbm", "memory bandwidth",
    "gpu", "accelerator", "silicon", "vlsi", "physical design",
    "3nm", "2nm", "advanced packaging", "cowos", "nvlink",
    "inference", "training", "distributed", "interconnect",
]


def fetch_job_postings_signals(companies: list[str], role_keywords: list[str]) -> list[dict]:
    signals  = []
    kw_lower = [k.lower() for k in (role_keywords or JOB_SIGNAL_KEYWORDS)]

    for slug, company_name in GREENHOUSE_COMPANIES:
        try:
            resp = _safe_get(
                f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                headers={"User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
            )
            if not resp or resp.status_code != 200:
                continue
            data = resp.json()
            jobs = data.get("jobs", []) if isinstance(data, dict) else data
            for job in jobs[:50]:
                title = (job.get("title") or "").lower()
                dept  = (job.get("departments", [{}])[0].get("name", "") if job.get("departments") else "").lower()
                matched_kw = next((k for k in kw_lower if k in f"{title} {dept}"), None)
                if not matched_kw:
                    continue
                signals.append({
                    "company":         company_name,
                    "keyword":         matched_kw,
                    "signal_strength": 8,
                    "url":             job.get("absolute_url", f"https://boards.greenhouse.io/{slug}"),
                    "note":            f"Greenhouse: {job.get('title','')} @ {company_name}",
                    "source_type":     "job_posting",
                })
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f"[Jobs] Greenhouse {slug} error: {e}")

    for slug, company_name in LEVER_COMPANIES:
        try:
            resp = _safe_get(
                f"https://api.lever.co/v0/postings/{slug}",
                params={"mode": "json"},
                headers={"User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
            )
            if not resp or resp.status_code != 200:
                continue
            jobs = resp.json() if isinstance(resp.json(), list) else []
            for job in jobs[:50]:
                title = (job.get("text") or "").lower()
                team  = (job.get("categories", {}).get("team") or "").lower()
                matched_kw = next((k for k in kw_lower if k in f"{title} {team}"), None)
                if not matched_kw:
                    continue
                signals.append({
                    "company":         company_name,
                    "keyword":         matched_kw,
                    "signal_strength": 8,
                    "url":             job.get("hostedUrl", f"https://jobs.lever.co/{slug}"),
                    "note":            f"Lever: {job.get('text','')} @ {company_name}",
                    "source_type":     "job_posting",
                })
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f"[Jobs] Lever {slug} error: {e}")

    for tenant, company_name, board in WORKDAY_COMPANIES:
        try:
            resp = requests.post(
                f"https://{tenant}.wd5.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs",
                json={"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""},
                headers={"Content-Type": "application/json", "User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            for job in resp.json().get("jobPostings", [])[:30]:
                title     = (job.get("title") or "").lower()
                matched_kw = next((k for k in kw_lower if k in title), None)
                if not matched_kw:
                    continue
                ext_url = job.get("externalPath", "")
                signals.append({
                    "company":         company_name,
                    "keyword":         matched_kw,
                    "signal_strength": 9,
                    "url":             f"https://{tenant}.wd5.myworkdayjobs.com{ext_url}" if ext_url else f"https://{tenant}.wd5.myworkdayjobs.com/{board}",
                    "note":            f"Workday: {job.get('title','')} @ {company_name}",
                    "source_type":     "job_posting",
                })
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"[Jobs] Workday {company_name} error: {e}")

    try:
        search_resp = _safe_get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "story,ask_hn", "query": "Ask HN: Who is hiring", "hitsPerPage": 1},
        )
        if search_resp:
            hits = search_resp.json().get("hits", [])
            if hits:
                story_id = hits[0].get("objectID", "")
                comments_resp = _safe_get(
                    "https://hn.algolia.com/api/v1/search",
                    params={"tags": f"comment,story_{story_id}", "hitsPerPage": 300},
                )
                if comments_resp:
                    for comment in comments_resp.json().get("hits", []):
                        text = (comment.get("comment_text") or "").lower()
                        for company in ["nvidia", "tsmc", "amd", "intel", "google",
                                        "cerebras", "tenstorrent", "graphcore",
                                        "sambanova", "d-matrix", "lightmatter"]:
                            if company not in text:
                                continue
                            matched_kw = next((k for k in kw_lower if k in text), None)
                            if matched_kw:
                                signals.append({
                                    "company":         company.title(),
                                    "keyword":         matched_kw,
                                    "signal_strength": 7,
                                    "url":             f"https://news.ycombinator.com/item?id={comment.get('objectID','')}",
                                    "note":            f"HN Who Is Hiring: {company} hiring for {matched_kw}",
                                    "source_type":     "job_posting",
                                })
                                break
    except Exception as e:
        logger.debug(f"[Jobs] HN error: {e}")

    logger.info(f"[Jobs] {len(signals)} job signals (Workday + Greenhouse + Lever + HN)")
    if not signals:
        logger.info("[Jobs] No signals retrieved — LLM will use its knowledge")
    return signals


# ── Main orchestrator ───────────────────────────────────────────────────────────

def fetch_all_for_cycle1(seed: dict) -> dict:
    import os
    keywords     = seed.get("seed_keywords", [])
    companies    = seed.get("target_companies", [])
    signals      = seed.get("intelligence_signals", [])
    github_token = os.environ.get("GITHUB_TOKEN", "")

    logger.info("[Sources] Starting full fetch for Cycle 1 harvest — 12 sources (OpenReview+SemanticScholar removed)")

    result = {
        "papers": [], "patents": [], "job_signals": [],
        "github_signals": [], "rss_signals": [],
        "edgar_signals": [], "darpa_signals": [],
        "nasa_signals": [],
    }

    for name, fn, kwargs in [
        ("arXiv",       fetch_arxiv_papers,   {"keywords": keywords, "max_results": 20, "days_back": 30}),
        ("OpenAlex",    fetch_openalex_papers, {"keywords": keywords, "max_results": 15, "days_back": 90}),
        ("HuggingFace", fetch_huggingface_papers, {"max_results": 15}),
        ("OSTI/DOE",    fetch_osti_research,       {"keywords": keywords, "max_results": 8}),
    ]:
        try:
            items = fn(**kwargs)
            result["papers"].extend(items)
            logger.info(f"[Sources] {name}: {len(items)}")
        except Exception as e:
            logger.error(f"[Sources] {name} failed: {e}")

    try:
        items = fetch_google_patents(keywords[:6], companies[:4], max_results=10)
        result["patents"] = items
        logger.info(f"[Sources] Patents: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] Patents failed: {e}")

    try:
        items = fetch_job_postings_signals(companies[:5], signals[:5])
        result["job_signals"] = items
        logger.info(f"[Sources] Job signals: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] Job signals failed: {e}")

    try:
        items = fetch_github_signals(github_token=github_token, max_issues=30)
        result["github_signals"] = items
        logger.info(f"[Sources] GitHub: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] GitHub failed: {e}")

    try:
        items = fetch_rss_signals(max_per_feed=5)
        result["rss_signals"] = items
        logger.info(f"[Sources] RSS: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] RSS failed: {e}")

    try:
        items = fetch_ocp_signals(github_token=github_token, max_items=20)
        result["github_signals"].extend(items)
        logger.info(f"[Sources] OCP: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] OCP failed: {e}")

    try:
        items = fetch_sec_edgar_signals(keywords=keywords[:6], max_results=10, days_back=180)
        result["edgar_signals"] = items
        logger.info(f"[Sources] SEC EDGAR: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] SEC EDGAR failed: {e}")

    try:
        items = fetch_darpa_baa_signals(keywords=keywords[:6], max_results=8, days_back=365)
        result["darpa_signals"] = items
        logger.info(f"[Sources] DARPA BAA: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] DARPA failed: {e}")

    try:
        items = fetch_nasa_research(keywords=keywords[:6], max_results=10)
        result["nasa_signals"] = items
        logger.info(f"[Sources] NASA NTRS+TechTransfer: {len(items)}")
    except Exception as e:
        logger.error(f"[Sources] NASA failed: {e}")

    total = sum(len(v) for v in result.values())
    logger.info(f"[Sources] Total items fetched: {total} (14 sources)")

    # ── Domain relevance filter ───────────────────────────────────────────────
    # Filter papers to only those relevant to the research domains.
    # Without this, HuggingFace daily papers (LLM benchmarks, NLP, etc.)
    # flood the context and dilute domain-specific signal.
    result["papers"] = _filter_by_domain_relevance(result["papers"], keywords)
    logger.info(f"[Sources] After domain filter: {len(result['papers'])} papers")

    # ── Unpaywall enrichment — add PDF URLs where available ──────────────────
    result["papers"] = enrich_unpaywall(result["papers"])

    filtered_total = sum(len(v) for v in result.values())
    logger.info(f"[Sources] Final total: {filtered_total} items")
    return result


# ── Domain Relevance Filter ────────────────────────────────────────────────────

# Core domain terms — items must match at least one to be kept
_DOMAIN_TERMS = [
    # thermal
    "thermal", "heat", "cooling", "temperature", "junction", "hotspot",
    "carnot", "heat flux", "thermal resistance", "thermal interface",
    # power
    "power", "energy", "voltage", "pdn", "ir drop", "electromigration",
    "landauer", "power density", "vrm", "decap", "power delivery",
    # data movement / memory
    "memory", "bandwidth", "hbm", "roofline", "memory wall", "latency",
    "dram", "cache", "data movement", "interconnect", "nvlink",
    # packaging
    "packaging", "chiplet", "cowos", "advanced packaging", "3d ic",
    "signal integrity", "interposer", "ucie",
    # compute scheduling / utilization
    "gpu", "accelerator", "inference", "training", "utilization",
    "scheduling", "workload", "compute", "throughput", "efficiency",
    # general AI hardware
    "ai chip", "ai accelerator", "tpu", "npu", "silicon",
    "semiconductor", "fabrication", "tsmc", "3nm", "5nm",
]


def _filter_by_domain_relevance(papers: list[dict], keywords: list[str]) -> list[dict]:
    """
    Keep only papers relevant to the R&D domains.
    Matches against title + abstract using domain terms AND seed keywords.
    Passes through papers with no abstract (can't judge, keep for safety).
    """
    kw_terms = [k.lower() for k in keywords[:20]]
    all_terms = _DOMAIN_TERMS + kw_terms

    kept = []
    for p in papers:
        text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
        if not text.strip() or len(text) < 20:
            kept.append(p)   # no text to judge — keep
            continue
        if any(t in text for t in all_terms):
            kept.append(p)
        else:
            logger.debug(f"[DomainFilter] Dropped: {p.get('title','?')[:60]}")

    logger.info(f"[DomainFilter] Kept {len(kept)}/{len(papers)} papers")
    return kept


# ── Unpaywall Enrichment ───────────────────────────────────────────────────────

def enrich_unpaywall(papers: list[dict]) -> list[dict]:
    """
    For each paper with a DOI, check Unpaywall for a free PDF URL.
    Adds pdf_url field to papers where open-access PDF is available.
    Uses UNPAYWALL_EMAIL (same as OPENALEX_EMAIL) — required by Unpaywall ToS.
    No API key needed — just email in query param.
    Rate limit: ~100k/day, no auth needed beyond email.
    """
    import os as _os
    email = _os.environ.get("UNPAYWALL_EMAIL", "") or _os.environ.get("OPENALEX_EMAIL", "")
    if not email:
        logger.debug("[Unpaywall] No email set — skipping enrichment")
        return papers

    enriched = 0
    for paper in papers:
        doi = paper.get("doi", "") or ""

        # ── FIX: comprehensive DOI extraction from multiple fields ────────────
        if not doi:
            # Try extracting from URL
            url = paper.get("url", "") or ""
            for prefix in ["doi.org/", "dx.doi.org/"]:
                if prefix in url:
                    doi = url.split(prefix)[-1].strip().rstrip("/")
                    break
        if not doi:
            # Try extracting from abstract/source fields
            abstract = paper.get("abstract", "") or ""
            import re as _re
            doi_match = _re.search(r'10[.]\d{4,}/[^\s<>]+', abstract)
            if doi_match:
                doi = doi_match.group(0).rstrip(".,;)")

        if not doi:
            continue

        # Skip non-standard DOIs
        if not doi.startswith("10.") or "/" not in doi:
            continue
        # Skip dataset DOIs — Unpaywall returns 422
        if any(x in doi.lower() for x in ["zenodo", "figshare", "dryad", "dvn/", "7910/"]):
            continue
        # Skip known paywalled publisher prefixes — Unpaywall returns 422 for most
        # Elsevier (10.1016), Springer (10.1007), Wiley (10.1002), etc.
        # These clog the rate limit — only try arXiv and open-access prefixes
        OPEN_PREFIXES = [
            "10.48550",  # arXiv
            "10.21203",  # Research Square (preprints)
            "10.1101",   # bioRxiv/medRxiv
            "10.31235",  # SocArXiv
            "10.5281",   # Zenodo (valid ones)
            "10.3390",   # MDPI (open access)
            "10.3389",   # Frontiers (open access)
            "10.1371",   # PLOS (open access)
            "10.7717",   # PeerJ
            "10.1145",   # ACM (sometimes OA)
            "10.1109",   # IEEE (sometimes OA)
        ]
        if not any(doi.startswith(p) for p in OPEN_PREFIXES):
            continue  # skip likely-paywalled DOIs entirely

        try:
            resp = _safe_get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": email},
                headers={"User-Agent": "rd-engine-research/1.0"},
            )
            if not resp:
                continue
            if resp.status_code == 422:
                logger.debug(f"[Unpaywall] 422 for DOI {doi[:40]} — skipping (not in Unpaywall)")
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            # Try all OA locations, not just best_oa
            pdf_url = ""
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url") or ""
            if not pdf_url:
                for loc in (data.get("oa_locations") or []):
                    candidate = loc.get("url_for_pdf") or loc.get("url") or ""
                    if candidate and candidate.endswith(".pdf"):
                        pdf_url = candidate
                        break
                if not pdf_url and (data.get("oa_locations") or []):
                    pdf_url = (data["oa_locations"][0].get("url_for_pdf")
                               or data["oa_locations"][0].get("url") or "")
            if pdf_url:
                paper["pdf_url"] = pdf_url
                enriched += 1
                logger.debug(f"[Unpaywall] PDF found for {paper.get('title','?')[:50]}")
            time.sleep(0.15)
        except Exception as e:
            logger.debug(f"[Unpaywall] Error for DOI {doi}: {e}")

    logger.info(f"[Unpaywall] Enriched {enriched}/{len(papers)} papers with PDF URLs")

    # After getting PDF URLs, try to fetch actual full text for top papers
    papers = _fetch_pdf_fulltext(papers, max_papers=5)

    return papers


def _fetch_pdf_fulltext(papers: list[dict], max_papers: int = 5) -> list[dict]:
    """
    For papers with a pdf_url, attempt to fetch the first ~1500 chars of text
    from the PDF to get numerical data beyond the abstract.
    Only fetches for papers that have pdf_url but no full_text yet.
    Limits to max_papers to avoid rate limiting.
    """
    import io
    fetched = 0
    for paper in papers:
        if fetched >= max_papers:
            break
        pdf_url = paper.get("pdf_url", "")
        if not pdf_url or paper.get("full_text"):
            continue
        try:
            resp = requests.get(
                pdf_url,
                headers={"User-Agent": "rd-engine-research/1.0"},
                timeout=15,
                stream=True,
            )
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("content-type", "")
            # Only process actual PDFs
            if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
                continue
            # Read first 100KB — enough for intro + methods
            raw = b""
            for chunk in resp.iter_content(chunk_size=8192):
                raw += chunk
                if len(raw) >= 102400:
                    break
            # Extract text from PDF bytes using pdfminer if available
            try:
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                output = io.StringIO()
                extract_text_to_fp(
                    io.BytesIO(raw),
                    output,
                    laparams=LAParams(),
                    output_type="text",
                    codec="utf-8",
                )
                text = output.getvalue().strip()
                if text and len(text) > 100:
                    paper["full_text"] = text[:1500]
                    fetched += 1
                    logger.debug(f"[PDFFetch] Got {len(text)} chars from {paper.get('title','?')[:50]}")
            except ImportError:
                # pdfminer not installed — store raw indicator
                logger.debug("[PDFFetch] pdfminer not available — skipping PDF text extraction")
                break
            except Exception as e:
                logger.debug(f"[PDFFetch] PDF parse failed: {e}")
        except Exception as e:
            logger.debug(f"[PDFFetch] Fetch failed for {pdf_url[:60]}: {e}")
        time.sleep(0.5)

    if fetched > 0:
        logger.info(f"[PDFFetch] Extracted full text from {fetched} PDFs")
    return papers


def fetch_papers_by_domain(domain: str, keywords: list[str], days_back: int = 60) -> list[dict]:
    domain_keyword_map = {
        "thermal":       ["thermal resistance", "heat flux GPU", "junction temperature", "3D IC cooling"],
        "power":         ["power delivery network", "CMOS power density", "energy per operation", "VRM AI"],
        "data_movement": ["memory bandwidth", "roofline model", "HBM latency", "memory wall LLM"],
        "pdn":           ["PDN impedance", "IR drop chiplet", "decoupling capacitor AI", "power integrity"],
        "hardware":      ["AI accelerator", "chiplet integration", "advanced packaging", "3nm 2nm TSMC"],
    }
    all_keywords = list(set(keywords + domain_keyword_map.get(domain, [])))
    return fetch_arxiv_papers(all_keywords[:10], max_results=12, days_back=days_back)


# ── GitHub ──────────────────────────────────────────────────────────────────────

TARGET_REPOS = [
    "NVIDIA/cuda-samples", "NVIDIA/TensorRT", "NVIDIA/apex", "NVIDIA/nccl",
    "ROCm/ROCm", "AMD/amd-lab-notes",
    "openai/triton", "openai/transformer-debugger",
    "microsoft/DeepSpeed", "microsoft/onnxruntime",
    "intel/intel-extension-for-pytorch",
    "google/jax", "google-deepmind/gemma",
    "pytorch/pytorch", "vllm-project/vllm",
    "huggingface/transformers",
    "TimDettmers/bitsandbytes",
]


def fetch_github_signals(github_token: str = "", max_issues: int = 30) -> list[dict]:
    if not github_token:
        import os
        github_token = os.environ.get("GITHUB_TOKEN", "")

    headers = {"User-Agent": "rd-engine/1.0", "Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    results     = []
    hw_keywords = ["memory", "bandwidth", "thermal", "power", "oom", "slow",
                   "performance", "latency", "cuda", "gpu", "error", "fail",
                   "crash", "hang", "bottleneck"]

    for repo in TARGET_REPOS[:8]:
        resp = _safe_get(
            f"https://api.github.com/repos/{repo}/issues",
            params={"state": "open", "sort": "comments", "direction": "desc", "per_page": 10},
            headers=headers,
        )
        if not resp:
            continue
        try:
            issues = resp.json()
            if not isinstance(issues, list):
                continue
            for issue in issues[:10]:
                title = (issue.get("title") or "").lower()
                body  = (issue.get("body") or "")[:400]
                if not any(kw in title or kw in body.lower() for kw in hw_keywords):
                    continue
                results.append({
                    "title":       issue.get("title", ""),
                    "body":        body,
                    "url":         issue.get("html_url", ""),
                    "repo":        repo,
                    "signal_type": "github_issue",
                    "created_at":  issue.get("created_at", ""),
                    "comments":    issue.get("comments", 0),
                })
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"[GitHub] Failed parsing {repo}: {e}")

    logger.info(f"[GitHub] Fetched {len(results)} issue signals")
    return results


# ── RSS Feeds ────────────────────────────────────────────────────────────────────

# FIX: The Register: /data_centre/rss → /headlines.atom (was 404)
RSS_FEEDS = [
    # ── Semiconductors & AI Hardware ─────────────────────────────────────────
    ("https://spectrum.ieee.org/feeds/feed.rss",                "ieee_spectrum"),
    ("https://www.eetimes.com/feed/",                           "ee_times"),
    ("https://semianalysis.com/feed/",                          "semianalysis"),
    ("https://www.tomshardware.com/feeds/all",                  "tomshardware"),
    ("https://www.techpowerup.com/rss/",                        "techpowerup"),
    ("https://www.anandtech.com/rss/",                          "anandtech"),
    ("https://semiwiki.com/feed/",                              "semiwiki"),
    ("https://www.theregister.com/headlines.atom",              "theregister_dc"),
    ("https://feeds.arstechnica.com/arstechnica/technology-lab","arstechnica"),
    # ── Robotics ─────────────────────────────────────────────────────────────
    ("https://www.therobotreport.com/feed/",                    "robot_report"),
    ("https://spectrum.ieee.org/feeds/topic/robotics.rss",      "ieee_robotics"),
    # ── Data Center Cooling & Power ───────────────────────────────────────────
    ("https://www.datacenterknowledge.com/rss.xml",             "datacenter_knowledge"),
    ("https://www.datacenterdynamics.com/en/rss/",              "dcd"),
]

RELEVANT_KEYWORDS = [
    # semiconductors
    "nvidia", "tsmc", "amd", "intel", "gpu", "hbm", "packaging",
    "thermal", "power", "chiplet", "ai accelerator", "semiconductor",
    "3nm", "2nm", "cowos", "nvlink", "pdn", "bandwidth",
    # robotics & mechanical
    "robot", "robotic", "actuator", "servo", "joint failure", "fatigue",
    "wiring harness", "connector", "gear wear", "vibration", "motor",
    # cooling & fluid
    "liquid cooling", "immersion cooling", "coolant", "cavitation",
    "leak", "corrosion", "data center cooling", "thermal management",
    # power & electrical
    "arc flash", "insulation", "voltage drop", "contact resistance",
]


def _parse_rss_items(xml_text: str, source_name: str) -> list:
    import xml.etree.ElementTree as ET
    import re

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    def _extract_items(root):
        return root.findall(".//item") or root.findall(".//atom:entry", ns)

    def _get_field(item, *tags):
        for tag in tags:
            val = item.findtext(tag, namespaces=ns) if "atom:" in tag else item.findtext(tag)
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

    try:
        root = ET.fromstring(xml_text)
        return [(item, _get_field, _get_link) for item in _extract_items(root)]
    except ET.ParseError:
        pass

    try:
        cleaned = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', xml_text)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        root    = ET.fromstring(cleaned)
        return [(item, _get_field, _get_link) for item in _extract_items(root)]
    except ET.ParseError as e:
        logger.debug(f"[RSS] {source_name} still invalid after cleanup: {e} — using regex fallback")

    items_out = []
    titles = re.findall(r'<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', xml_text, re.S)
    links  = re.findall(r'<link[^>]*>(https?://[^<]+)</link>', xml_text)
    descs  = re.findall(r'<description[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>', xml_text, re.S)
    for i, title in enumerate(titles[1:]):
        items_out.append({
            "title": title.strip(),
            "desc":  descs[i].strip()[:400] if i < len(descs) else "",
            "link":  links[i].strip() if i < len(links) else "",
        })
    return items_out


def fetch_rss_signals(max_per_feed: int = 5) -> list[dict]:
    results = []
    for feed_url, source_name in RSS_FEEDS:
        resp = _safe_get(feed_url)
        if not resp:
            continue
        try:
            parsed = _parse_rss_items(resp.text, source_name)
            count  = 0
            for item in parsed:
                if isinstance(item, dict):
                    title, desc, link = item["title"], item["desc"], item["link"]
                else:
                    et_item, _get_field, _get_link = item
                    title = _get_field(et_item, "title", "atom:title")
                    desc  = _get_field(et_item, "description", "summary", "atom:summary")[:400]
                    link  = _get_link(et_item)
                if not any(kw in (title + " " + desc).lower() for kw in RELEVANT_KEYWORDS):
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


# ── OCP ──────────────────────────────────────────────────────────────────────────

OCP_REPOS = [
    "opencomputeproject/OCP-Profiles",
    "opencomputeproject/Project_Olympus",
    "opencomputeproject/ocp-diag-core",
]

OCP_RELEVANT_KEYWORDS = [
    "thermal", "power", "cooling", "pdu", "rack", "voltage",
    "bandwidth", "accelerator", "gpu", "ai", "hbm", "packaging",
    "efficiency", "watt", "heat", "temperature",
]


def fetch_ocp_signals(github_token: str = "", max_items: int = 20) -> list[dict]:
    import os
    if not github_token:
        github_token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"User-Agent": "rd-engine/1.0", "Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    results = []
    for repo in OCP_REPOS:
        for endpoint, key in [("issues", "ocp_spec_discussion"), ("releases", "ocp_spec_release")]:
            try:
                params = {"state": "open", "sort": "created", "direction": "desc", "per_page": 5} if endpoint == "issues" else {"per_page": 3}
                resp   = _safe_get(f"https://api.github.com/repos/{repo}/{endpoint}", params=params, headers=headers)
                if not resp or resp.status_code != 200:
                    continue
                items = resp.json()
                if not isinstance(items, list):
                    continue
                for item in items[:3]:
                    body     = (item.get("body") or "")[:400]
                    name     = item.get("title") or item.get("name") or ""
                    combined = (name + " " + body).lower()
                    if not any(kw in combined for kw in OCP_RELEVANT_KEYWORDS):
                        continue
                    results.append({
                        "title":       f"[OCP SPEC] {name}" if endpoint == "releases" else name,
                        "body":        body,
                        "url":         item.get("html_url", ""),
                        "repo":        repo,
                        "signal_type": key,
                        "created_at":  item.get("created_at") or item.get("published_at", ""),
                    })
            except Exception as e:
                logger.debug(f"[OCP] {endpoint} error {repo}: {e}")
            time.sleep(0.5)
    logger.info(f"[OCP] Fetched {len(results)} OCP signals")
    return results


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────────

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


def fetch_sec_edgar_signals(keywords=None, max_results=10, days_back=180):
    """
    FIX v3: Use EDGAR full-text search UI endpoint (efts.sec.gov/LATEST/search-index)
    with correct query format. The v2 query used double-quotes around company names
    which EDGAR treats as exact phrase — very strict, returns 0 for long names.
    Fix: use ticker symbol only + keyword, no quotes.
    """
    results         = []
    search_keywords = keywords or SEC_RISK_KEYWORDS
    since_date      = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    for company_name, ticker in SEC_TARGET_COMPANIES[:6]:
        for kw in search_keywords[:3]:
            try:
                # Use ticker + keyword — EDGAR indexes tickers reliably
                resp = _safe_get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={
                        "q":       f"{ticker} {kw}",
                        "forms":   "10-K,10-Q",
                        "startdt": since_date,
                        "enddt":   datetime.now().strftime("%Y-%m-%d"),
                    },
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; rd-engine-research/1.0)",
                        "Accept":     "application/json",
                    },
                )
                if not resp or resp.status_code != 200:
                    continue
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    continue

                for hit in hits[:2]:
                    src          = hit.get("_source", {})
                    form_type    = src.get("form_type", "10-K")
                    filing_date  = src.get("file_date", "")
                    accession_no = src.get("accession_no", "")
                    entity_id    = src.get("entity_id", "")

                    acc_clean  = accession_no.replace("-", "")
                    filing_url = (
                        f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{acc_clean}/"
                        if entity_id and acc_clean
                        else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}"
                    )

                    highlights = hit.get("highlight", {})
                    snippet = ""
                    for field_hits in highlights.values():
                        if field_hits:
                            snippet = " ... ".join(str(h) for h in field_hits[:2])[:500]
                            break
                    if not snippet:
                        snippet = f"{company_name} ({ticker}) {form_type} filing mentions '{kw}' as risk factor."

                    results.append({
                        "title":       f"[{form_type}] {company_name} — {kw}",
                        "abstract":    snippet,
                        "url":         filing_url,
                        "company":     company_name,
                        "keyword":     kw,
                        "filing_date": filing_date,
                        "form_type":   form_type,
                        "source":      "sec_edgar",
                        "signal_type": "regulatory_filing",
                    })
                time.sleep(0.4)
            except Exception as e:
                logger.debug(f"[EDGAR] Query failed ({ticker} + {kw}): {e}")

            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    logger.info(f"[EDGAR] Fetched {len(results)} SEC filing signals")
    return results


# ── DARPA BAA ─────────────────────────────────────────────────────────────────────

# ── OpenAlex ───────────────────────────────────────────────────────────────────

def fetch_openalex_papers(keywords: list[str], max_results: int = 15, days_back: int = 90) -> list[dict]:
    """
    OpenAlex — largest open academic graph (~250M works, indexed within 1-2 days).

    FIX: search + filter combined returned 0 results.
    Correct approach: use filter with title.search for each short keyword separately,
    then merge and deduplicate. OpenAlex title.search is strict — short terms work best.
    """
    import os as _os
    from datetime import datetime, timedelta

    email      = _os.environ.get("OPENALEX_EMAIL", "")
    since_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    results    = []
    seen_ids   = set()

    # Use short 2-word terms per query — OpenAlex search is strict
    def _short_terms(kws):
        short = []
        for kw in kws:
            parts = kw.split()
            term = " ".join(parts[:2]) if len(parts) > 2 else kw
            if term not in short:
                short.append(term)
        return short

    terms = _short_terms(keywords[:10])[:5]   # max 5 queries to avoid rate limiting

    for term in terms:
        params: dict = {
            "filter":   f"title.search:{term},from_publication_date:{since_date}",
            "sort":     "cited_by_count:desc",
            "per-page": min(max(3, max_results // len(terms)), 10),
            "select":   "id,title,abstract_inverted_index,doi,publication_date,authorships,cited_by_count,open_access",
        }
        if email:
            params["mailto"] = email

        logger.info(f"[OpenAlex] Searching: {term} (last {days_back}d)")

        resp = _safe_get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": f"rd-engine-research/1.0 (mailto:{email or 'anonymous'})"},
        )
        if not resp:
            continue

        try:
            data = resp.json()
            for work in data.get("results", []):
                work_id = work.get("id", "")
                if work_id in seen_ids:
                    continue
                seen_ids.add(work_id)

                title = (work.get("title") or "").strip()
                if not title:
                    continue

                # Reconstruct abstract from inverted index
                abstract = ""
                inv_idx  = work.get("abstract_inverted_index") or {}
                if inv_idx:
                    try:
                        max_pos = max(pos for positions in inv_idx.values() for pos in positions)
                        words   = [""] * (max_pos + 1)
                        for word, positions in inv_idx.items():
                            for p in positions:
                                words[p] = word
                        abstract = " ".join(words).strip()[:600]
                    except Exception:
                        abstract = ""

                doi    = work.get("doi") or ""
                url    = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else work_id)
                oa_url = (work.get("open_access") or {}).get("oa_url") or url
                authors = [
                    (a.get("author") or {}).get("display_name", "")
                    for a in (work.get("authorships") or [])[:5]
                ]

                results.append({
                    "title":     title,
                    "abstract":  abstract,
                    "url":       oa_url or url,
                    "authors":   [a for a in authors if a],
                    "published": work.get("publication_date", ""),
                    "citations": work.get("cited_by_count", 0),
                    "source":    "openalex",
                })
        except Exception as e:
            logger.error(f"[OpenAlex] Parse error for '{term}': {e}")
        time.sleep(0.3)

    results = results[:max_results]
    logger.info(f"[OpenAlex] Retrieved {len(results)} papers (sorted by citations)")
    return results


# ── NASA Technical Reports & Tech Transfer ─────────────────────────────────────

def fetch_nasa_research(keywords: list[str], max_results: int = 10) -> list[dict]:
    """
    Two real-time NASA sources:

    1. NTRS (NASA Technical Reports Server) — open, no key needed.
       Uses NASA-specific domain keywords — NOT seed keywords.
       NASA researchers write about "thermal interface materials", "heat dissipation",
       not "GPU kernel utilization". Mapping applied below.

    2. NASA Tech Transfer — patents from NASA R\'D, queried with short patent-friendly terms.
       Uses NASA_API_KEY (DEMO_KEY = 30 req/hour; register free at api.nasa.gov).
    """
    import os as _os

    results  = []

    # ── NASA domain keyword mapping ───────────────────────────────────────────
    # NTRS responds best to short, single-topic queries.
    # Multi-word phrases return 0. Single domain words return real results.
    # v2: expanded to cover robotics + fluid dynamics domains
    NASA_DOMAIN_QUERIES = [
        "thermal",
        "cooling",
        "heat pipe",
        "power management",
        "wiring harness",
        "actuator",
        "corrosion",
        "fluid dynamics",
        "fatigue",
        "vibration",
    ]

    # ── Part 1: NTRS papers ───────────────────────────────────────────────────
    # FIX v3: NTRS /api/citations/search returns 0 results for all queries.
    # Use the correct NTRS search endpoint with proper parameters.
    ntrs_per_query = max(2, max_results // len(NASA_DOMAIN_QUERIES))

    for ntrs_query in NASA_DOMAIN_QUERIES:
        logger.info(f"[NASA-NTRS] Searching: {ntrs_query}")
        # Try both known endpoints
        for ntrs_url, ntrs_params in [
            (
                "https://ntrs.nasa.gov/api/citations/search",
                {"q": ntrs_query, "rows": ntrs_per_query, "sort": "modified desc"},
            ),
            (
                "https://ntrs.nasa.gov/search",
                {"q": ntrs_query, "rows": ntrs_per_query},
            ),
        ]:
            resp = _safe_get(
                ntrs_url,
                params=ntrs_params,
                headers={"User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
            )
            if not resp:
                continue
            try:
                data = resp.json()
                # Handle both response formats
                hits_data = data.get("hits", {})
                docs = hits_data.get("hits") or data.get("results") or []
                if not docs:
                    continue
                for doc in docs:
                    src   = doc.get("_source", doc)  # flat or nested
                    title = (src.get("title") or "").strip()
                    if not title:
                        continue
                    ntrs_id  = src.get("id") or doc.get("_id", "")
                    pub_date = (src.get("modified") or src.get("publicationDate") or "")[:10]
                    results.append({
                        "title":     title,
                        "abstract":  (src.get("abstract") or "")[:600],
                        "url":       f"https://ntrs.nasa.gov/citations/{ntrs_id}" if ntrs_id else "https://ntrs.nasa.gov",
                        "published": pub_date,
                        "type":      src.get("stiTypeDetails", ""),
                        "source":    "nasa_ntrs",
                    })
                break  # success — don't try alternate endpoint
            except Exception as e:
                logger.warning(f"[NASA-NTRS] Parse error ({ntrs_url}): {e}")
        time.sleep(0.5)

    # deduplicate by title
    seen_titles: set = set()
    deduped = []
    for r in results:
        t = r["title"].lower()[:80]
        if t not in seen_titles:
            seen_titles.add(t)
            deduped.append(r)
    results = deduped[:max_results]
    logger.info(f"[NASA-NTRS] Retrieved {len(results)} technical reports")

    # ── Part 2: NASA Tech Transfer patents ────────────────────────────────────
    # Endpoint changed: api.nasa.gov/techtransfer → technology.nasa.gov/api/api/patent/{keyword}
    # No API key required — public endpoint.
    TRANSFER_QUERIES = ["thermal", "power", "semiconductor", "heat pipe", "actuator", "corrosion", "cooling", "wiring"]
    patent_results = []

    for tq in TRANSFER_QUERIES[:3]:
        resp2 = _safe_get(
            f"https://technology.nasa.gov/api/api/patent/{tq}",
            headers={"User-Agent": "rd-engine-research/1.0", "Accept": "application/json"},
        )
        if not resp2:
            continue
        try:
            body = resp2.text.strip()
            if not body:
                logger.warning(f"[NASA-TechTransfer] Empty response for '{tq}' — skipping")
                continue
            data2 = resp2.json()
            for item in (data2.get("results") or [])[:4]:
                if len(item) < 4:
                    continue
                # result format: [id, case_number, title, description, patent_id, category, ...]
                pat_title = str(item[2]).strip()
                pat_desc  = str(item[3]).strip()[:500]
                pat_id    = str(item[4]).strip() if len(item) > 4 else ""
                pat_url   = f"https://technology.nasa.gov/patent/{pat_id}" if pat_id else "https://technology.nasa.gov/patents"
                if not pat_title or pat_title == "None":
                    continue
                patent_results.append({
                    "title":     f"[NASA Patent] {pat_title}",
                    "abstract":  pat_desc,
                    "url":       pat_url,
                    "published": "",
                    "source":    "nasa_tech_transfer",
                    "type":      "patent",
                })
        except Exception as e:
            logger.warning(f"[NASA-TechTransfer] Parse error for '{tq}': {e}")
        time.sleep(0.5)

    patent_results = patent_results[:max_results // 2]
    logger.info(f"[NASA-TechTransfer] Retrieved {len(patent_results)} patent records")

    combined = results + patent_results
    logger.info(f"[NASA] Total: {len(combined)} records (NTRS + Tech Transfer)")
    return combined


DARPA_RELEVANT_KEYWORDS = [
    "semiconductor", "computing", "microelectronics", "photonics",
    "thermal", "power", "memory", "bandwidth", "processor",
    "packaging", "chiplet", "accelerator", "neural", "ai",
]


def fetch_darpa_baa_signals(keywords=None, max_results=10, days_back=365):
    """
    FIX v2: Multiple DARPA signal sources:
    1. DARPA main RSS — broader keyword filter (was too strict: 0 results)
    2. SAM.gov open search — public, no API key, returns BAA/SBIR opportunities
    3. grants.gov RSS — DARPA + ARPA-E opportunities
    """
    results   = []
    kw_filter = [k.lower() for k in (keywords or DARPA_RELEVANT_KEYWORDS)]

    # ── Source 1: DARPA RSS ───────────────────────────────────────────────────
    DARPA_RSS_URLS = [
        "https://www.darpa.mil/rss.xml",
        # opportunities.rss returned 404 — removed
    ]
    import xml.etree.ElementTree as ET

    for rss_url in DARPA_RSS_URLS:
        try:
            resp = _safe_get(rss_url, headers={"User-Agent": "rd-engine-research/1.0"})
            if not resp:
                continue
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:max_results * 5]:
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "")[:600].strip()
                link  = (item.findtext("link") or "").strip()
                combined = (title + " " + desc).lower()
                # Broader match: any keyword OR just "DARPA" in title
                if not (any(kw in combined for kw in kw_filter) or "baa" in combined or "sbir" in combined):
                    continue
                results.append({
                    "title":       f"[DARPA] {title}",
                    "abstract":    desc or title,
                    "url":         link,
                    "posted_date": item.findtext("pubDate") or "",
                    "source":      "darpa_baa",
                    "signal_type": "government_funding_signal",
                })
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.debug(f"[DARPA] RSS error ({rss_url}): {e}")
        if results:
            break  # stop after first working feed

    # ── Source 2: grants.gov RSS — public, no key, covers DARPA/ARPA-E ─────────
    if len(results) < max_results:
        try:
            grants_resp = _safe_get(
                "https://www.grants.gov/rss/GG_NewOpps.xml",
                headers={"User-Agent": "rd-engine-research/1.0"},
            )
            if grants_resp and grants_resp.status_code == 200:
                import xml.etree.ElementTree as ET2
                root2 = ET2.fromstring(grants_resp.text)
                for item in root2.findall(".//item")[:30]:
                    title    = (item.findtext("title") or "").strip()
                    desc     = (item.findtext("description") or "")[:400].strip()
                    link     = (item.findtext("link") or "").strip()
                    combined = (title + " " + desc).lower()
                    if not any(kw in combined for kw in kw_filter):
                        continue
                    results.append({
                        "title":       f"[Grants.gov] {title}",
                        "abstract":    desc or title,
                        "url":         link,
                        "posted_date": item.findtext("pubDate") or "",
                        "source":      "darpa_baa",
                        "signal_type": "government_funding_signal",
                    })
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.debug(f"[DARPA/Grants.gov] Error: {e}")

    logger.info(f"[DARPA] Fetched {len(results)} DARPA BAA signals")
    return results[:max_results]
