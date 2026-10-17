"""Research for the forecasting bot, built from sources that need no paid search API.

The brief has up to three parts:
1. A web-search brief from Gemini with Google Search grounding (only when its small free quota allows; skipped otherwise).
2. Recent news headlines from Google News RSS (queries written by a cheap DeepSeek call).
3. Related prediction-market prices (Polymarket, Manifold).

Checking similar questions/markets and web research correlated best with score in Metaculus' Spring 2026 bot
survey, so markets are always included when found. No probability is produced here.
"""

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

GROUNDED_MODEL = os.getenv("RESEARCH_MODEL", "gemini-2.5-flash")
_UA = {"User-Agent": "Mozilla/5.0 (cariuang-bot research; contact via github.com/neimasilk)"}


class ResearchError(RuntimeError):
    pass


def _get(url: str, timeout: int = 25) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout).read()


# ----------------------------------------------------------------------------- queries

def _deepseek_json(prompt: str, max_tokens: int = 300) -> dict:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        raise ResearchError("DEEPSEEK_API_KEY is not set")
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    data = json.load(urllib.request.urlopen(req, timeout=60))
    text = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ResearchError("no JSON in query-writer reply")
    return json.loads(m.group(0))


def _fallback_queries(question_text: str) -> dict:
    words = re.sub(r"[^A-Za-z0-9 ]", " ", question_text).split()
    short = " ".join(words[:10])
    return {"news": [short], "markets": [short]}


async def write_queries(question_text: str, background: str) -> dict:
    prompt = (
        "Write search queries for a forecaster researching this question. Reply with JSON only, in the form "
        '{"news": ["...", "...", "..."], "markets": ["...", "..."]}. '
        '"news" are 3 short news-search queries (2 to 6 words each, no quotes, name the key entities). '
        '"markets" are 2 short queries to find related prediction markets (Polymarket, Manifold).\n\n'
        f"Question: {question_text}\nBackground: {(background or '')[:600]}"
    )
    try:
        out = await asyncio.to_thread(_deepseek_json, prompt)
        news = [str(q)[:120] for q in out.get("news", [])][:3]
        markets = [str(q)[:120] for q in out.get("markets", [])][:2]
        if news:
            return {"news": news, "markets": markets or news[:1]}
    except Exception as e:  # noqa: BLE001 - research must degrade, not fail
        logger.warning("query writer failed: %s", e)
    return _fallback_queries(question_text)


# ----------------------------------------------------------------------------- news

def _google_news(query: str, days: int = 30, limit: int = 6) -> list[str]:
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(f"{query} when:{days}d")
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    root = ET.fromstring(_get(url))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for it in root.findall(".//item"):
        try:
            when = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if when < cutoff:
            continue
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        rows.append((when, f"- {when:%Y-%m-%d} {source}: {title}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [r[1] for r in rows[:limit]]


async def news_block(queries: list[str]) -> str:
    seen, lines = set(), []
    for q in queries:
        try:
            for line in await asyncio.to_thread(_google_news, q):
                key = line.split(": ", 1)[-1][:60].lower()
                if key not in seen:
                    seen.add(key)
                    lines.append(line)
        except Exception as e:  # noqa: BLE001
            logger.warning("news query %r failed: %s", q, e)
    return "\n".join(lines[:14])


# ----------------------------------------------------------------------------- markets

def _polymarket(query: str) -> list[str]:
    url = "https://gamma-api.polymarket.com/public-search?q=" + urllib.parse.quote(query) + "&limit_per_type=4&events_status=active"
    data = json.loads(_get(url))
    out = []
    for ev in (data.get("events") or [])[:4]:
        for m in (ev.get("markets") or [])[:3]:
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                outcomes = json.loads(m.get("outcomes") or "[]")
            except (TypeError, ValueError):
                continue
            if not prices or m.get("closed"):
                continue
            pairs = ", ".join(f"{o} {float(p) * 100:.0f}%" for o, p in list(zip(outcomes, prices))[:3])
            end = (m.get("endDate") or "")[:10]
            out.append(f'- Polymarket: "{(m.get("question") or ev.get("title") or "")[:130]}" (ends {end}): {pairs}')
    return out[:6]


def _manifold(query: str) -> list[str]:
    url = "https://api.manifold.markets/v0/search-markets?term=" + urllib.parse.quote(query) + "&limit=4&sort=liquidity&filter=open"
    data = json.loads(_get(url))
    out = []
    for m in data[:4]:
        if m.get("outcomeType") == "BINARY" and m.get("probability") is not None:
            out.append(f'- Manifold: "{m.get("question", "")[:130]}": {float(m["probability"]) * 100:.0f}% (traders: {m.get("uniqueBettorCount", "?")})')
    return out


async def markets_block(queries: list[str]) -> str:
    lines = []
    for q in queries:
        for fn in (_polymarket, _manifold):
            try:
                lines += await asyncio.to_thread(fn, q)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s query %r failed: %s", fn.__name__, q, e)
    seen, uniq = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            uniq.append(line)
    return "\n".join(uniq[:10])


# ----------------------------------------------------------------------------- grounded brief (optional)

def _grounded_prompt(question_text: str, background: str, criteria: str, fine_print: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        "You are a research assistant to a professional forecaster. You never give a probability yourself.\n"
        f"Today is {today} (UTC). Search the web for the most recent and most relevant information for the question below.\n\n"
        f"Question: {question_text}\n\nBackground: {background}\n\nResolution criteria: {criteria}\n\nFine print: {fine_print}\n\n"
        "Write a brief of at most 350 words: (1) current status with dates, (2) scheduled events before the question closes, "
        "(3) base rates or comparable cases with numbers, (4) any market or forecaster estimates for this or a closely related "
        "event, with the number and date, (5) what would push the outcome toward Yes or No. Flag conflicts and uncertainty."
    )


def _call_grounded(prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GROUNDED_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2500},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": api_key}
    )
    data = json.load(urllib.request.urlopen(req, timeout=150))
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or []).strip()
    if not text:
        raise ResearchError("empty grounded research")
    return text


async def grounded_brief(question_text: str, background: str, criteria: str, fine_print: str) -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return ""
    try:
        return await asyncio.to_thread(_call_grounded, _grounded_prompt(question_text, background, criteria, fine_print), key)
    except urllib.error.HTTPError as e:
        logger.info("grounded research skipped (HTTP %s)", e.code)  # usually the small free daily quota
    except Exception as e:  # noqa: BLE001
        logger.warning("grounded research failed: %s", e)
    return ""


# ----------------------------------------------------------------------------- public entry point

async def build_brief(question_text: str, background: str, criteria: str, fine_print: str) -> str:
    queries = await write_queries(question_text, background)
    grounded, news, markets = await asyncio.gather(
        grounded_brief(question_text, background, criteria, fine_print),
        news_block(queries["news"]),
        markets_block(queries["markets"]),
    )
    parts = []
    if grounded:
        parts.append("WEB-SEARCH BRIEF (Gemini with Google Search):\n" + grounded)
    if news:
        parts.append("RECENT NEWS HEADLINES (Google News, newest first; headlines only, unverified):\n" + news)
    if markets:
        parts.append("RELATED PREDICTION MARKETS (may differ from this question's exact resolution criteria):\n" + markets)
    if not parts:
        raise ResearchError("no research sources returned anything")
    note = (
        "NOTE: this research was gathered automatically. Some items may be irrelevant to the question; ignore those. "
        "Headlines are unverified snippets, not established facts."
    )
    return note + "\n\n" + "\n\n".join(parts)
