import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import feedparser
import requests


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    link: str
    published: str


@dataclass
class SentimentResult:
    action: str
    score: float
    confidence: int
    matched_themes: List[str]
    source_count: int
    summary: str


class NewsFetcher:
    def __init__(self, sources: Optional[List[str]] = None) -> None:
        self.sources = sources or [
            # Core energy feeds
            "https://www.investing.com/rss/news_25.rss",
            "https://www.oilprice.com/rss/main",
            "https://www.cnbc.com/id/19836768/device/rss/rss.html",
            "https://www.eia.gov/rss/press_rss.xml",
            "https://www.spglobal.com/commodityinsights/en/rss-feed/news",
            # Google News queries (international English sources aggregation)
            "https://news.google.com/rss/search?q=Brent+oil+Iran+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=US+crude+inventories+EIA+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Trump+Iran+sanctions+oil+when:2d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Middle+East+war+oil+supply+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Reuters+Brent+oil+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Bloomberg+oil+market+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Financial+Times+oil+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Wall+Street+Journal+oil+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=AP+News+Middle+East+oil+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=OPEC+meeting+oil+output+when:2d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=IEA+oil+demand+forecast+when:3d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Russia+oil+exports+sanctions+when:2d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=China+oil+demand+refinery+when:2d&hl=en-US&gl=US&ceid=US:en",
        ]

    def fetch_latest_news(self, max_items_per_source: int = 3) -> List[NewsItem]:
        collected: List[NewsItem] = []
        seen = set()

        for source_url in self.sources:
            feed = feedparser.parse(source_url)
            for entry in feed.entries[:max_items_per_source]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                key = (title.strip().lower(), link.strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                collected.append(
                    NewsItem(
                        title=title,
                        summary=getattr(entry, "summary", ""),
                        source=source_url,
                        link=link,
                        published=getattr(entry, "published", ""),
                    )
                )
        return collected


def _domain_from_link(link: str) -> str:
    try:
        return urlparse(link).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def _canonical_title(title: str) -> str:
    text = title.lower().strip()
    for ch in [",", ".", ":", ";", "-", "_", "|", "/", "\\", "(", ")", "[", "]", "{", "}", '"', "'"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _parse_published_ts(text: str) -> Optional[datetime]:
    if not text:
        return None
    cleaned = text.strip()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _recency_multiplier(published: str) -> float:
    ts = _parse_published_ts(published)
    if ts is None:
        return 1.0
    hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if hours <= 6:
        return 1.25
    if hours <= 24:
        return 1.10
    if hours <= 48:
        return 1.0
    return 0.85


def _source_weight(domain: str) -> float:
    weights = {
        "reuters.com": 1.35,
        "bloomberg.com": 1.30,
        "ft.com": 1.28,
        "wsj.com": 1.25,
        "apnews.com": 1.2,
        "eia.gov": 1.4,
        "spglobal.com": 1.25,
        "cnbc.com": 1.15,
        "investing.com": 1.08,
        "oilprice.com": 1.05,
    }
    for key, val in weights.items():
        if key in domain:
            return val
    if "news.google.com" in domain:
        return 1.0
    return 0.95


def _rule_based_scoring(news_items: List[NewsItem]) -> SentimentResult:
    theme_rules: Dict[str, Dict[str, Any]] = {
        "iran_tension": {
            "weight": 12,
            "positive_for_oil": ["iran attack", "iran tension", "strait of hormuz", "sanctions on iran"],
            "negative_for_oil": ["iran deal", "sanction relief", "nuclear deal progress"],
        },
        "us_inventory": {
            "weight": 10,
            "positive_for_oil": ["inventory draw", "stockpile draw", "crude draw"],
            "negative_for_oil": ["inventory build", "stockpile build", "crude build"],
        },
        "opec_supply": {
            "weight": 10,
            "positive_for_oil": ["opec cut", "production cut", "supply disruption"],
            "negative_for_oil": ["increase output", "output hike", "oversupply"],
        },
        "demand_growth": {
            "weight": 8,
            "positive_for_oil": ["demand rises", "strong demand", "economic expansion"],
            "negative_for_oil": ["recession risk", "demand weak", "demand slowdown"],
        },
        "trump_policy": {
            "weight": 7,
            "positive_for_oil": ["trump sanctions", "hardline iran policy", "tariff shock"],
            "negative_for_oil": ["trump calls lower oil", "pressure for cheaper energy"],
        },
    }

    raw_score = 0.0
    matched_themes: List[str] = []
    source_domains = set()
    theme_source_map: Dict[str, set] = {}
    title_domain_map: Dict[str, set] = {}
    duplicate_clusters = 0

    for item in news_items[:24]:
        text = f"{item.title} {item.summary}".lower()
        domain = _domain_from_link(item.link) or _domain_from_link(item.source)
        source_domains.add(domain)
        recency_mul = _recency_multiplier(item.published)
        src_mul = _source_weight(domain)

        canon = _canonical_title(item.title)
        if canon:
            title_domain_map.setdefault(canon, set()).add(domain)

        for theme, rule in theme_rules.items():
            pos_hits = sum(1 for kw in rule["positive_for_oil"] if kw in text)
            neg_hits = sum(1 for kw in rule["negative_for_oil"] if kw in text)
            if pos_hits == 0 and neg_hits == 0:
                continue

            delta = (pos_hits - neg_hits) * rule["weight"] * recency_mul * src_mul
            raw_score += delta
            matched_themes.append(theme)
            theme_source_map.setdefault(theme, set()).add(domain)

    # Multi-source confirmation bonus.
    confirmation_bonus = 0.0
    for theme, srcs in theme_source_map.items():
        if len(srcs) >= 2:
            confirmation_bonus += 4.0
        if len(srcs) >= 3:
            confirmation_bonus += 3.0

    # If same/similar headline appears across many domains, increase confidence and score mildly.
    for domains in title_domain_map.values():
        if len(domains) >= 2:
            duplicate_clusters += 1
            confirmation_bonus += 1.8
        if len(domains) >= 3:
            confirmation_bonus += 1.6
    raw_score += confirmation_bonus

    score = max(-100.0, min(100.0, raw_score))
    unique_themes = sorted(set(matched_themes))
    confidence = min(95, 35 + len(unique_themes) * 9 + len(source_domains) * 3 + duplicate_clusters * 2)

    if score >= 18:
        action = "AL"
    elif score <= -18:
        action = "SAT"
    else:
        action = "TUT"

    return SentimentResult(
        action=action,
        score=score,
        confidence=int(confidence),
        matched_themes=unique_themes,
        source_count=len(source_domains),
        summary=(
            f"rule_score={score:.1f}, confirmation_bonus={confirmation_bonus:.1f}, "
            f"clusters={duplicate_clusters}"
        ),
    )


def _ollama_adjustment(news_items: List[NewsItem], base_result: SentimentResult) -> Optional[SentimentResult]:
    if not news_items:
        return None

    prompt_parts = []
    for item in news_items[:10]:
        prompt_parts.append(f"- {item.title} | {item.summary}")
    combined_news = "\n".join(prompt_parts)

    prompt = (
        "You are an oil strategist. Evaluate Brent direction from these headlines. "
        "Return STRICT JSON only with keys: action(AL|SAT|TUT), score(-100..100), "
        "confidence(0..100), summary(max 1 sentence).\n\n"
        f"Rule-based baseline: {asdict(base_result)}\n\n"
        f"Headlines:\n{combined_news}"
    )

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    try:
        response = requests.post(
            ollama_url,
            json={"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=25,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        upper = raw.upper()

        action = "TUT"
        if '"ACTION":"AL"' in upper or " ACTION: AL" in upper:
            action = "AL"
        elif '"ACTION":"SAT"' in upper or " ACTION: SAT" in upper:
            action = "SAT"

        score = base_result.score
        confidence = base_result.confidence
        return SentimentResult(
            action=action,
            score=max(-100.0, min(100.0, score)),
            confidence=max(1, min(99, confidence)),
            matched_themes=base_result.matched_themes,
            source_count=base_result.source_count,
            summary="llm_checked",
        )
    except Exception:
        return None


def analyze_sentiment_detailed(news_items: List[NewsItem]) -> SentimentResult:
    if not news_items:
        return SentimentResult(
            action="TUT",
            score=0.0,
            confidence=10,
            matched_themes=[],
            source_count=0,
            summary="no_news",
        )

    base = _rule_based_scoring(news_items)
    llm = _ollama_adjustment(news_items, base)
    if not llm:
        return base

    combined_score = (base.score * 0.75) + (llm.score * 0.25)
    combined_conf = int((base.confidence * 0.7) + (llm.confidence * 0.3))
    action = llm.action if llm.action != "TUT" else base.action

    if combined_score >= 18:
        action = "AL"
    elif combined_score <= -18:
        action = "SAT"
    elif action not in ("AL", "SAT"):
        action = "TUT"

    return SentimentResult(
        action=action,
        score=combined_score,
        confidence=combined_conf,
        matched_themes=base.matched_themes,
        source_count=base.source_count,
        summary=f"{base.summary}; llm_blend",
    )


def analyze_sentiment(news_items: List[NewsItem]) -> str:
    return analyze_sentiment_detailed(news_items).action

