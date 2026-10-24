"""Headlines for the Market page from the GDELT DOC 2.0 API (https://www.gdeltproject.org/).

Why GDELT: its datasets are "available for unlimited and unrestricted use for any academic, commercial, or
governmental use of any kind without fee", provided that "any use or redistribution of the data must include a
citation to the GDELT Project and a link to this website (https://www.gdeltproject.org/)" (GDELT "About" page,
read 2026-10-23). The page cites it under each news section. Google News and Bing News RSS feeds carry a
copyright line that allows them only in a personal feed reader for personal, non-commercial use, and the BBC
terms (section 15) do not allow taking metadata out of its feeds, so none of them is used.

GDELT allows one request every 5 seconds per address and answers HTTP 429 above that. A request waits for the
gap, a 429 is retried once, and every failure raises NewsError with the reason, so the page can say why no
headline came back and a failed query is not cached.
"""
import html
import json
import re
import threading
import time

import pandas as pd
import requests

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
    t = re.sub(r"(\d) %", r"\1%", t)                         # 25 % -> 25%
    t = re.sub(r" ([,:;!?])", r"\1", t)                      # "Natgas , Henry" -> "Natgas, Henry"
    t = re.sub(r"(\d) / ([A-Za-z])", r"\1/\2", t)             # "USD 112 / tonne" -> "USD 112/tonne"
    # "year - end", "low - carbon", "war - driven" -> one hyphenated word. Only when one side is a part that
    # forms compounds, so a dash between two clauses ("prices rise - analysts warn") keeps its spaces.
    _c = ("low|high|year|half|short|long|mid|non|pre|post|re|co|multi|cross|gas|coal|oil|state|market|export"
          "|driven|led|backed|based|related|linked|owned|run|fired|term|end|start|over|under|sized|scale|free"
          "|wide|intensive|heavy|light|day|ahead|month|week|time|hour|front|side|cycle|carbon|energy|price"
          "|barrel|tonne|bbl|cooled|cooling|powered|fuelled|fueled|rich|poor|only|per|era|style|like|plus")
    t = re.sub(rf"(?<=[A-Za-z]) - (?=(?:{_c})\b)", "-", t, flags=re.I)
    t = re.sub(rf"\b({_c}) - (?=[A-Za-z])", r"\1-", t, flags=re.I)
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
