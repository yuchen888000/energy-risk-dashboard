"""Headlines for the Market page from the GDELT DOC 2.0 API (https://www.gdeltproject.org/).

Why GDELT: its datasets are "available for unlimited and unrestricted use for any academic, commercial, or
governmental use of any kind without fee", provided that "any use or redistribution of the data must include a
citation to the GDELT Project and a link to this website (https://www.gdeltproject.org/)" (GDELT "About" page,
read 2026-10-23). The page cites it under each news section. Google News and Bing News RSS feeds carry a
copyright line that allows them only in a personal feed reader for personal, non-commercial use, and the BBC
terms (section 15) do not allow taking metadata out of its feeds, so none of them is used.

GDELT allows one request every 5 seconds per address and answers HTTP 429 above that. A request waits for the
gap, a 429 is retried twice, and every failure raises NewsError with the reason, so the page can say why no
headline came back and a failed query is not cached.

GDELT also answers HTTP 429 to shared cloud addresses that have not sent a request for some time (13 and 14 Nov
2026: the hosted app, and a test from another cloud server after 30 seconds without a request). So a GitHub
Actions job (.github/scripts/save_news.py) saves GDELT's answer to the four commodity searches in
data/news_saved.json, and the page falls back on that saved copy, or on the last answer the app itself got,
when GDELT refuses.
"""
import html
import json
import os
import re
import threading
import time

import pandas as pd
import requests

# GDELT search for each commodity's headline list: any of these phrases, which the page's own keyword filter then
# narrows (gdelt_query). Single words under three letters are rejected by GDELT. Kept here, not in common.py, so
# the saving job can use them without Streamlit.
NEWS_QUERIES = {
    "gas": '("natural gas" OR LNG OR "gas price" OR "gas storage" OR TTF) (Europe OR European OR Dutch OR Germany OR Britain)',
    # US benchmark terms (Cushing, "US crude", shale) so the list is not the same as Brent's.
    "wti": '("crude oil" OR "oil price" OR WTI OR "oil market" OR OPEC OR Cushing OR "US crude" OR shale)',
    "brent": '(Brent OR "crude oil" OR "oil price" OR "oil market" OR OPEC)',
    "carbon": '("EU ETS" OR "carbon price" OR "carbon market" OR "emissions trading" OR "carbon allowance")',
}
NEWS_DAYS = 7              # the page's headlines cover the last 7 days; the saving job asks for the same
SAVED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "news_saved.json")
SAVED_MAX_AGE = pd.Timedelta(days=2)   # an older saved answer is not shown

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_CAP = 250            # the API returns at most this many articles per query (MAXRECORDS)
MIN_GAP_S = 5.5            # GDELT: "Please limit requests to one every 5 seconds"
RETRY_WAITS_S = (6.0, 10.0)   # waits before the second and third try after HTTP 429
TIMEOUT_S = 30

_lock = threading.Lock()
_last_call = [0.0]


class NewsError(RuntimeError):
    """No headlines from GDELT this time; the message says why, in words for the page."""


def term(t):
    """A search term as GDELT expects it: a phrase of several words in quotes, one word as it is."""
    t = t.strip()
    return f'"{t}"' if " " in t else t


def build_query(*terms):
    """All terms must appear (GDELT joins terms with AND); English-language sources only.
    GDELT rejects a single search word of fewer than three letters, so pass "EU ETS", not "EU"."""
    return " ".join(term(t) for t in terms if t.strip()) + " sourcelang:english"


def gdelt_query(expr):
    """A ready GDELT expression (OR groups in brackets, groups joined by AND), English-language sources only."""
    return f"{expr.strip()} sourcelang:english"


def _clean_title(t):
    """GDELT tokenises titles: it puts spaces around punctuation ("$30 . 01", "25 %", "year - end", "Pact : Algeria")
    and keeps the site name the publisher appends ("... | Hellenic Shipping News Worldwide") or puts in front
    ("Baystreet . ca - ..."). Those are undone, so the headline reads as published and the same story from two
    sites is recognised as one."""
    t = re.sub(r"\s+", " ", html.unescape(t or "")).strip()
    t = re.sub(r"(\d) \. (\d)", r"\1.\2", t)                # 30 . 01 -> 30.01
    t = re.sub(r"\b([A-Z]) \. (?=[A-Z] \.|[A-Z]\b)", r"\1.", t)  # "U . S . Crude" -> "U.S. Crude"
    t = re.sub(r"\b([A-Z]) \. (?=[A-Z][a-z])", r"\1. ", t)
    t = re.sub(r"\b([A-Z]\.[A-Z]) \.", r"\1.", t)              # "U.S . naval" (left by the rules above) -> "U.S. naval"
    t = re.sub(r"(\d) %", r"\1%", t)                         # 25 % -> 25%
    t = re.sub(r" ([,:;!?])", r"\1", t)                      # "Natgas , Henry" -> "Natgas, Henry"
    t = re.sub(r"\b(\d{1,3}), (\d{3})(?!\d)", r"\1,\2", t)    # "Rs 48 , 056" -> "Rs 48,056" (thousands)
    t = re.sub(r"(\w) \. (?=[A-Z])", r"\1. ", t)               # "$85,000 . Here" -> "$85,000. Here"
    t = re.sub(r" \.$", ".", t)                              # a full stop at the end
    t = re.sub(r"\( ", "(", t)                               # "( LON: UKOG )" -> "(LON: UKOG)"
    t = re.sub(r" \)", ")", t)
    t = re.sub(r"(\d) / ([A-Za-z])", r"\1/\2", t)             # "USD 112 / tonne" -> "USD 112/tonne"
    # "year - end", "low - carbon", "war - driven" -> one hyphenated word. Only when one side is a part that
    # forms compounds, so a dash between two clauses ("prices rise - analysts warn") keeps its spaces.
    _c = ("low|high|year|half|short|long|mid|non|pre|post|re|co|multi|cross|gas|coal|oil|state|market|export"
          "|driven|led|backed|based|related|linked|owned|run|fired|term|end|start|over|under|sized|scale|free"
          "|wide|intensive|heavy|light|day|ahead|month|week|time|hour|front|side|cycle|carbon|energy|price"
          "|barrel|tonne|bbl|cooled|cooling|powered|fuelled|fueled|rich|poor|only|per|era|style|like|plus")
    t = re.sub(rf"(?<=[A-Za-z]) - (?=(?:{_c})\b)", "-", t, flags=re.I)
    # Not before the word that names who said it ("... to Turkmen gas - expert", "... - analysts", "... - minister"):
    # that dash separates the headline from its source, so it keeps its spaces.
    _who = r"(?:experts?|analysts?|officials?|ministers?|minister|sources?|reports?|study|survey|poll|data|CEO|says?)\b"
    t = re.sub(rf"\b({_c}) - (?=[A-Za-z])(?!{_who})", r"\1-", t, flags=re.I)
    # "4 - month high", "1 - week rally": a number, a period word, then a noun.
    t = re.sub(r"(\d) - (?=(?:month|week|year|day|hour|per|barrel|tonne)\b)", r"\1-", t, flags=re.I)
    t = re.sub(r"^[A-Za-z0-9-]+ \. [a-z]{2,6} - ", "", t)    # "Baystreet . ca - Japan ..." -> "Japan ..."
    if " | " in t:                                          # "... | Site Name": keep the headline part
        head, tail = t.rsplit(" | ", 1)
        if len(tail.split()) <= 6 and len(head) >= 20:
            t = head
    return t.strip()


def _parse(text):
    """JSON of an artlist answer. GDELT sometimes leaves raw control characters inside titles, which strict
    JSON rejects: they are replaced by spaces and the text parsed again."""
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(re.sub(r"[\x00-\x1f]", " ", text))


def fetch(query, days, get=requests.get, sleep=time.sleep, clock=time.monotonic):
    """Articles matching `query` that GDELT saw in the last `days` days, newest first, at most GDELT_CAP.
    Returns a list of dicts: title, url, domain (the publisher's site), country (GDELT's source country),
    seen (pandas Timestamp, UTC: when GDELT first saw the article, close to but not always its publication
    time). An empty list means GDELT answered and found nothing. Raises NewsError when GDELT does not answer,
    is still rate-limited after two retries, or answers with an error text."""
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": str(GDELT_CAP),
              "timespan": f"{int(days)}d", "sort": "datedesc"}
    resp = None
    for attempt in range(len(RETRY_WAITS_S) + 1):
        with _lock:
            wait = MIN_GAP_S - (clock() - _last_call[0])
            if wait > 0:
                sleep(wait)
            try:
                resp = get(GDELT_URL, params=params, timeout=TIMEOUT_S)
            except requests.RequestException as e:
                raise NewsError(f"GDELT did not answer ({type(e).__name__})") from e
            finally:
                _last_call[0] = clock()
        if resp.status_code != 429:
            break
        if attempt < len(RETRY_WAITS_S):
            sleep(RETRY_WAITS_S[attempt])
    if resp.status_code == 429:
        raise NewsError("GDELT limited the request rate (HTTP 429)")
    if resp.status_code != 200:
        raise NewsError(f"GDELT answered HTTP {resp.status_code}")
    body = resp.text.strip()
    if body in ("", "{}"):
        return []          # GDELT found nothing
    if not body.startswith("{"):
        # Errors come back as plain text with status 200 ("The specified phrase is too short.").
        raise NewsError(f"GDELT returned an error message: {body[:120]}")
    try:
        data = _parse(body)
    except ValueError as e:
        raise NewsError("GDELT returned text that is not valid JSON") from e
    out = []
    for a in data.get("articles") or []:
        title = _clean_title(a.get("title"))
        try:
            seen = pd.to_datetime(a.get("seendate", ""), format="%Y%m%dT%H%M%SZ").tz_localize("UTC")
        except (ValueError, TypeError):
            continue
        if not title:
            continue
        out.append({"title": title, "url": a.get("url", ""), "domain": a.get("domain", ""),
                    "country": a.get("sourcecountry", ""), "seen": seen})
    return out


# ─── Saved copy: GDELT's last answer to each commodity search (data/news_saved.json) ───
def articles_to_json(articles):
    """Articles as fetch returns them, with `seen` as text, for the saved file."""
    return [{**a, "seen": a["seen"].strftime("%Y-%m-%dT%H:%M:%SZ")} for a in articles]


def articles_from_json(rows):
    """The saved file's articles back as fetch returns them; a row without a readable time is left out."""
    out = []
    for a in rows or []:
        try:
            seen = pd.Timestamp(a["seen"])
            seen = seen.tz_localize("UTC") if seen.tzinfo is None else seen.tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        if a.get("title"):
            out.append({"title": a["title"], "url": a.get("url", ""), "domain": a.get("domain", ""),
                        "country": a.get("country", ""), "seen": seen})
    return out


def load_saved(path=SAVED_PATH):
    """{query expression: (saved at, UTC Timestamp, articles)} from the saved file; {} if it is missing or broken."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for q, entry in (data.get("queries") or {}).items():
        try:
            at = pd.Timestamp(entry["saved_at"])
            at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        out[q] = (at, articles_from_json(entry.get("articles")))
    return out


def fallback(query, days, sources, now):
    """The newest earlier answer to `query` among `sources` (each {query: (time, articles)}) that is at most
    SAVED_MAX_AGE old at `now`, with articles older than `days` days at `now` left out: (time, articles), or None."""
    best = None
    for src in sources:
        hit = src.get(query)
        if hit and now - hit[0] <= SAVED_MAX_AGE and (best is None or hit[0] > best[0]):
            best = hit
    if best is None:
        return None
    cut = now - pd.Timedelta(days=days)
    return best[0], [a for a in best[1] if a["seen"] >= cut]
