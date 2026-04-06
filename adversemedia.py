#!/usr/bin/env python3
"""
adverse_media_scanner.py
Flag potential wrongdoing news about a target entity using Google News RSS, GDELT, (opt) NewsAPI,
and regulator sources (SEC/MAS). Outputs a scored CSV and console table.

Usage:
  python adverse_media_scanner.py "JPMorgan" --aliases "JPM,JP Morgan,JP Morgan Chase" \
      --tickers JPM --days 30 --newsapi_key YOUR_KEY

Author: (you)
"""

import os, re, sys, json, time, math, html, logging, argparse, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import requests
import feedparser
import pandas as pd
from dateutil import parser as dtp
from bs4 import BeautifulSoup
import tldextract
from rapidfuzz import fuzz
from tabulate import tabulate

# -------------------------- Config & Taxonomy --------------------------

WRONGDOING_LABELS = [
    "fraud / securities fraud",
    "accounting irregularities / restatement",
    "bribery / corruption / FCPA",
    "money laundering / AML / CFT",
    "sanctions violation / export control",
    "insider trading / market manipulation",
    "antitrust / competition",
    "tax evasion",
    "environmental violation",
    "labor / human rights / safety",
    "cybersecurity breach / data privacy",
    "regulatory investigation / probe",
    "lawsuit / class action / settlement",
]

# Lightweight keyword fallback; extend as needed.
KEYWORDS = {
    "fraud / securities fraud": [
        r"\bfraud(ulent|)?\b", r"\bponzi\b", r"\bmisrepresent(ed|ation)\b",
        r"\bsecurities\s+fraud\b", r"\bwire\s+fraud\b"
    ],
    "accounting irregularities / restatement": [
        r"\brestat(e|ement)\b", r"\baccounting\s+irregularit(y|ies)\b",
        r"\bmaterial\s+weakness(es)?\b"
    ],
    "bribery / corruption / FCPA": [
        r"\bbrib(ery|e|ed)\b", r"\bcorruption\b", r"\bFCPA\b", r"\bkickback(s)?\b"
    ],
    "money laundering / AML / CFT": [
        r"\bmoney\s+launder(ing|ed)\b", r"\bAML\b", r"\bCFT\b",
        r"\bKYC\b", r"\bterrorist\s+financ(ing|e)\b"
    ],
    "sanctions violation / export control": [
        r"\bsanction(s|ed)?\b", r"\bOFAC\b", r"\bexport\s+control(s)?\b",
        r"\bSDN\b", r"\bConsolidated\s+Screening\s+List\b"
    ],
    "insider trading / market manipulation": [
        r"\binsider\s+trad(ing|e)\b", r"\bfront\-running\b", r"\bspoof(ing|ed)\b",
        r"\bmanipulation\b", r"\bcorner\b", r"\bsqueeze\b"
    ],
    "antitrust / competition": [
        r"\banti\-?trust\b", r"\bcompetition\s+law\b", r"\bprice\s+fix(ing|)\b",
        r"\bcartel\b"
    ],
    "tax evasion": [
        r"\btax\s+evasion\b", r"\bunderreport(ing|ed)\b", r"\bshell\s+company\b"
    ],
    "environmental violation": [
        r"\benvironmental\s+(violation|offen[cs]e)\b", r"\bEPA\b",
        r"\bpollut(ion|e[ds])\b", r"\bspill\b"
    ],
    "labor / human rights / safety": [
        r"\bchild\s+labor\b", r"\bforced\s+labor\b", r"\bosha\b",
        r"\bsafety\s+violation\b", r"\bhuman\s+rights\b", r"\bharass(ment|)\b"
    ],
    "cybersecurity breach / data privacy": [
        r"\bdata\s+breach\b", r"\bransomware\b", r"\bhack(ed|)\b",
        r"\bprivacy\s+violation\b", r"\bGDPR\b"
    ],
    "regulatory investigation / probe": [
        r"\bregulator(y|)\s+(investigation|probe)\b", r"\bshow\-cause\b",
        r"\benforcement\s+action\b"
    ],
    "lawsuit / class action / settlement": [
        r"\blawsuit\b", r"\bclass\s+action\b", r"\bsettlement\b", r"\blitigation\b"
    ],
}
NEGATION_HINTS = [r"\ballege(d|s|ly)\b", r"\bden(y|ies|ied)\b", r"\bno\s+wrongdoing\b"]

TRUST_WEIGHTS = {  # simple priors; tune with your own lists
    "reuters.com": 1.00, "ft.com": 0.95, "bloomberg.com": 0.95, "wsj.com": 0.95,
    "nytimes.com": 0.92, "apnews.com": 0.92, "bbc.com": 0.92, "cnbc.com": 0.9,
    "str.sg": 0.9, "businesstimes.com.sg": 0.9, "channelnewsasia.com": 0.9,
}
DEFAULT_TRUST = 0.75

HEADERS = {
    # Friendly UA to avoid 403 on some gov sites (SEC, etc.)
    "User-Agent": "AdverseMediaScanner/1.0 (+https://example.org; for compliance OSINT)",
    "Accept": "*/*",
    "Accept-Language": "en;q=0.9",
    "Connection": "keep-alive",
}

# -------------------------- Utilities --------------------------

def extract_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    except Exception:
        return ""

def parse_date(value: str) -> Optional[datetime]:
    try:
        dt = dtp.parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def recency_weight(published: Optional[datetime], now: datetime) -> float:
    if not published:
        return 0.6
    days = (now - published).total_seconds() / 86400.0
    if days <= 3: return 1.0
    if days <= 7: return 0.9
    if days <= 30: return 0.75
    if days <= 90: return 0.6
    return 0.45

def first(s: Optional[str]) -> str:
    return (s or "").strip()

# -------------------------- Zero-shot (optional) --------------------------

_ZS = None
def get_zero_shot():
    """Load a zero-shot classifier if transformers is installed."""
    global _ZS
    if _ZS is not None:
        return _ZS
    try:
        from transformers import pipeline
        _ZS = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return _ZS
    except Exception:
        return None

# -------------------------- Data model --------------------------

@dataclass
class Hit:
    source: str            # domain or source label
    source_type: str       # 'news', 'rss', 'regulator', 'gdelt'
    title: str
    url: str
    published: Optional[datetime]
    excerpt: str
    matched_entity: bool
    labels: List[str]
    label_scores: Dict[str, float]
    negation_hint: bool
    risk_score: float

# -------------------------- Fetchers --------------------------

def gdelt_doc(query: str, timespan: str = "1m", maxrecords: int = 150) -> List[dict]:
    """
    GDELT DOC 2.0 API: query article list.
    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ and data docs.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "timespan": timespan,
        "format": "json",
        "sort": "DateDesc",
    }
    r = requests.get(url, params=params, timeout=20, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])

def gdelt_context(query, timespan="7d", maxrecords=250):
    """Fetch from GDELT Context API."""
    url = "https://api.gdeltproject.org/api/v2/context/context"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": maxrecords
    }
    
    try:
        r = requests.get(url, params=params, timeout=20, headers=HEADERS)
        r.raise_for_status()
        
        # Check if response is actually JSON
        if not r.text.strip():
            logging.warning("GDELT Context returned empty response")
            return []
        
        try:
            data = r.json()
            return data.get("articles", [])
        except ValueError as e:
            logging.warning(f"GDELT Context returned non-JSON response: {r.text[:200]}")
            return []
            
    except requests.exceptions.RequestException as e:
        logging.warning(f"GDELT Context request failed: {e}")
        return []

def google_news_rss(query: str) -> List[dict]:
    """
    Google News RSS search (undocumented; works well in practice).
    Helpful guide to parameters: see Newscatcher blog.
    """
    url = "https://news.google.com/rss/search"
    # Tip: intitle:, allintext:, inurl: can focus results; avoid site: to stay broad.
    params = {"q": query, "hl": "en-SG", "gl": "SG", "ceid": "SG:en"}
    feed = feedparser.parse(requests.get(url, params=params, headers=HEADERS, timeout=20).text)
    out = []
    for e in feed.entries:
        out.append({
            "title": e.get("title"),
            "url": e.get("link"),
            "seendate": e.get("published") or e.get("updated"),
            "domain": extract_domain(e.get("link", "")),
            "excerpt": html.unescape(re.sub("<.*?>", "", e.get("summary", "") or "")),
            "source": "google_news",
        })
    return out

def newsapi_everything(query: str, from_dt: datetime, api_key: str, page_size: int = 100, pages: int = 1) -> List[dict]:
    """
    NewsAPI Everything endpoint (requires API key).
    Docs: https://newsapi.org/docs/endpoints/everything
    """
    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": api_key, **HEADERS}
    results = []
    for page in range(1, pages + 1):
        params = {
            "q": query,
            "from": from_dt.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "page": page,
        }
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 429:
            break
        r.raise_for_status()
        payload = r.json()
        for a in payload.get("articles", []):
            results.append({
                "title": a.get("title"),
                "url": a.get("url"),
                "seendate": a.get("publishedAt"),
                "domain": extract_domain(a.get("url", "")),
                "excerpt": a.get("description") or "",
                "source": a.get("source", {}).get("name") or "newsapi",
            })
    return results

def sec_litigation_releases_rss() -> List[dict]:
    """SEC Litigation Releases RSS (official)."""
    rss_url = "https://www.sec.gov/enforcement-litigation/litigation-releases/rss"
    text = requests.get(rss_url, headers=HEADERS, timeout=20).text
    feed = feedparser.parse(text)
    out = []
    for e in feed.entries:
        out.append({
            "title": e.get("title"),
            "url": e.get("link"),
            "seendate": e.get("published"),
            "domain": "sec.gov",
            "excerpt": re.sub("<.*?>", "", e.get("summary", "") or ""),
            "source": "SEC Litigation Releases",
        })
    return out

def sec_press_releases_rss() -> List[dict]:
    """SEC Newsroom Press Releases RSS."""
    rss_url = "https://www.sec.gov/news/pressreleases.rss"
    text = requests.get(rss_url, headers=HEADERS, timeout=20).text
    feed = feedparser.parse(text)
    out = []
    for e in feed.entries:
        out.append({
            "title": e.get("title"),
            "url": e.get("link"),
            "seendate": e.get("published"),
            "domain": "sec.gov",
            "excerpt": re.sub("<.*?>", "", e.get("summary", "") or ""),
            "source": "SEC Press Releases",
        })
    return out

def mas_enforcement_list() -> List[dict]:
    """
    MAS Enforcement Actions landing page scrape (simple list items and cards).
    https://www.mas.gov.sg/regulation/enforcement/enforcement-actions
    """
    url = "https://www.mas.gov.sg/regulation/enforcement/enforcement-actions"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    # Titles typically within 'a' tags beneath list/card containers.
    for a in soup.select("a"):
        href = a.get("href", "")
        text = (a.get_text() or "").strip()
        if not href or not text:
            continue
        if "/regulation/enforcement/enforcement-actions" in href and href != url:
            full = href if href.startswith("http") else "https://www.mas.gov.sg" + href
            out.append({
                "title": text,
                "url": full,
                "seendate": None,
                "domain": "mas.gov.sg",
                "excerpt": "",
                "source": "MAS Enforcement",
            })
    # Deduplicate by URL
    seen = set()
    dedup = []
    for item in out:
        if item["url"] in seen: continue
        seen.add(item["url"])
        dedup.append(item)
    return dedup

# -------------------------- Matching & Scoring --------------------------

def entity_string_match(entity: str, aliases: List[str], text: str, threshold: int = 85) -> bool:
    """Fuzzy name match — catches minor variants. Tweak threshold for strictness."""
    hay = (text or "").lower()
    for cand in [entity] + aliases:
        c = cand.lower().strip()
        if not c: continue
        if c in hay: return True
        # fuzzy partial ratio on small windows (fallback)
        if fuzz.partial_ratio(c, hay) >= threshold:
            return True
    return False

def keyword_labels(text: str) -> Tuple[List[str], Dict[str, float], bool]:
    """Heuristic label assignment with negation hints."""
    text = (text or "")
    hits, scores = [], {}
    for label, patterns in KEYWORDS.items():
        score = 0.0
        for pat in patterns:
            if re.search(pat, text, flags=re.I):
                score += 0.5
        if score > 0:
            hits.append(label)
            scores[label] = min(1.0, score)
    neg = any(re.search(p, text, flags=re.I) for p in NEGATION_HINTS)
    return hits, scores, neg

def zero_shot_labels(text: str) -> Tuple[List[str], Dict[str, float]]:
    zs = get_zero_shot()
    if not zs:
        return [], {}
    res = zs(text[:2000], WRONGDOING_LABELS, multi_label=True)
    labels, scores = [], {}
    for lbl, sc in zip(res["labels"], res["scores"]):
        if sc >= 0.35:  # threshold for relevance
            labels.append(lbl)
            scores[lbl] = float(sc)
    return labels, scores

def severity_weight(labels: List[str]) -> float:
    if not labels:
        return 0.5
    sev_map = {
        "sanctions violation / export control": 1.0,
        "money laundering / AML / CFT": 0.95,
        "insider trading / market manipulation": 0.9,
        "fraud / securities fraud": 0.9,
        "bribery / corruption / FCPA": 0.9,
        "cybersecurity breach / data privacy": 0.8,
        "accounting irregularities / restatement": 0.8,
        "antitrust / competition": 0.75,
        "tax evasion": 0.75,
        "environmental violation": 0.7,
        "labor / human rights / safety": 0.7,
        "regulatory investigation / probe": 0.65,
        "lawsuit / class action / settlement": 0.6,
    }
    return max(sev_map.get(l, 0.6) for l in labels)

def trust_weight(domain: str) -> float:
    return TRUST_WEIGHTS.get(domain, DEFAULT_TRUST)

def score_hit(domain: str, published: Optional[datetime], labels: List[str], negation_hint: bool, now: datetime) -> float:
    base = trust_weight(domain)
    timew = recency_weight(published, now)
    sev = severity_weight(labels)
    score = base * timew * sev
    if negation_hint:
        score *= 0.85  # discount if strongly hedged
    return round(score, 4)

# -------------------------- Pipeline --------------------------

def normalize_article(raw: dict) -> dict:
    """Map various sources to a common schema."""
    return {
        "title": first(raw.get("title")),
        "url": first(raw.get("url")),
        "excerpt": first(raw.get("excerpt") or raw.get("context") or raw.get("snippet") or raw.get("summary")),
        "domain": extract_domain(first(raw.get("url"))),
        "seendate": first(raw.get("seendate")),
        "source": first(raw.get("source") or raw.get("domain")),
    }

def unique_id(item: dict) -> str:
    key = f"{item.get('title','')}|{item.get('url','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def collect(entity: str, aliases: List[str], days: int, newsapi_key: Optional[str]) -> List[Hit]:
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(days=days)

    search_query = f"\"{entity}\" (fraud OR bribery OR corruption OR AML OR investigation OR probe OR sanctions OR settlement OR lawsuit OR insider OR manipulation OR 'money laundering' OR 'regulatory action')"

    # --- Gather
    blobs = []

    # Google News RSS (broad media)
    blobs += google_news_rss(query=search_query)

    # GDELT DOC + Context
    blobs += gdelt_doc(query=search_query, timespan=f"{max(1, days)}d" if days <= 60 else "2m")
    blobs += gdelt_context(query=search_query, timespan=f"{max(1, min(days, 60))}d")

    # Regulator feeds
    try:
        blobs += sec_litigation_releases_rss()
    except Exception as e:
        logging.warning("SEC Litigation RSS fetch failed: %s", e)
    try:
        blobs += sec_press_releases_rss()
    except Exception as e:
        logging.warning("SEC Press RSS fetch failed: %s", e)
    try:
        blobs += mas_enforcement_list()
    except Exception as e:
        logging.warning("MAS enforcement scrape failed: %s", e)

    # NewsAPI (optional)
    if newsapi_key:
        blobs += newsapi_everything(query=search_query, from_dt=since_dt, api_key=newsapi_key, pages=2)

    # --- Normalize, filter, classify
    seen = set()
    hits: List[Hit] = []
    for b in blobs:
        n = normalize_article(b)
        if not n["title"] or not n["url"]:
            continue
        uid = unique_id(n)
        if uid in seen:  # de-dup by title+URL
            continue
        seen.add(uid)

        pub = parse_date(n["seendate"]) if n["seendate"] else None
        if pub and pub < since_dt:
            continue

        text_for_match = " ".join([n["title"], n["excerpt"]])
        matched = entity_string_match(entity, aliases, text_for_match)

        # Quick path: regulators – we still require match or alias
        is_regulator = n["domain"] in {"sec.gov", "mas.gov.sg"}

        if not matched and not is_regulator:
            # Ignore general articles that merely mention the search words without the specific entity
            continue

        # Labels via zero-shot then fallback to keywords
        labels_zs, scores_zs = zero_shot_labels(text_for_match)
        labels_kw, scores_kw, neg = keyword_labels(text_for_match)

        # Merge labels/scores
        label_set = list({*labels_zs, *labels_kw})
        scores = {**scores_kw, **scores_zs}  # zero-shot overwrites if present

        risk = score_hit(n["domain"], pub, label_set, neg, now)

        hits.append(Hit(
            source=n["source"] or n["domain"],
            source_type=("regulator" if is_regulator else ("gdelt" if "gdelt" in n["source"].lower() else "news")),
            title=n["title"],
            url=n["url"],
            published=pub,
            excerpt=n["excerpt"],
            matched_entity=matched or is_regulator,
            labels=label_set,
            label_scores=scores,
            negation_hint=neg,
            risk_score=risk
        ))

    # Rank
    hits.sort(key=lambda h: h.risk_score, reverse=True)
    return hits

def to_dataframe(hits: List[Hit]) -> pd.DataFrame:
    rows = []
    for h in hits:
        rows.append({
            "risk_score": h.risk_score,
            "labels": "; ".join(h.labels),
            "title": h.title,
            "source": h.source,
            "domain": extract_domain(h.url),
            "published": h.published.isoformat() if h.published else "",
            "url": h.url,
            "matched_entity": h.matched_entity,
            "negation_hint": h.negation_hint,
            "excerpt": h.excerpt,
            "label_scores": json.dumps(h.label_scores, ensure_ascii=False),
        })
    return pd.DataFrame(rows)

# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser(description="Adverse media / wrongdoing screener")
    ap.add_argument("entity", help="Target entity name, e.g., 'JPMorgan Chase & Co.'")
    ap.add_argument("--aliases", default="", help="Comma-separated aliases / brand names")
    ap.add_argument("--tickers", default="", help="Comma-separated tickers (for matching in titles)")
    ap.add_argument("--days", type=int, default=30, help="Lookback window in days (default 30)")
    ap.add_argument("--newsapi_key", default=os.getenv("NEWSAPI_KEY"), help="NewsAPI key (optional)")
    ap.add_argument("--out", default="adverse_media_results.csv", help="Output CSV path")
    args = ap.parse_args()

    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    if args.tickers:
        aliases += [t.strip() for t in args.tickers.split(",") if t.strip()]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Scanning for potential wrongdoing about: %s (aliases: %s)", args.entity, aliases)

    hits = collect(args.entity, aliases, args.days, args.newsapi_key)
    df = to_dataframe(hits)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} records -> {args.out}\n")

    # Pretty console view (top 12)
    preview = df.head(12)[["risk_score", "labels", "title", "domain", "published", "url"]]
    print(tabulate(preview, headers="keys", tablefmt="github", showindex=False))

if __name__ == "__main__":
    main()
