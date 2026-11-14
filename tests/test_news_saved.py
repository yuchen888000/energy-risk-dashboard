"""Offline tests of the saved GDELT answers: the file round trip, the fallback rule and the saving job."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import news_data  # noqa: E402

_path = os.path.join(os.path.dirname(__file__), "..", ".github", "scripts", "save_news.py")
_spec = importlib.util.spec_from_file_location("save_news", _path)
save_news = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(save_news)

NOW = pd.Timestamp("2026-11-14 12:00", tz="UTC")


def _art(title, hours_ago):
    return {"title": title, "url": "https://x.org/a", "domain": "x.org", "country": "UK",
            "seen": NOW - pd.Timedelta(hours=hours_ago)}


def test_round_trip(tmp_path):
    q = news_data.gdelt_query(news_data.NEWS_QUERIES["gas"])
    arts = [_art("Gas storage falls", 3)]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(save_news.update({}, {q: arts}, NOW)))
    saved = news_data.load_saved(str(p))
    assert saved[q][0] == NOW
    assert saved[q][1] == arts


def test_missing_or_broken_file(tmp_path):
    assert news_data.load_saved(str(tmp_path / "none.json")) == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert news_data.load_saved(str(tmp_path / "bad.json")) == {}


def test_fallback_takes_newest_and_drops_old_articles():
    q = "q"
    older = {q: (NOW - pd.Timedelta(hours=10), [_art("a", 12)])}
    newer = {q: (NOW - pd.Timedelta(hours=2), [_art("b", 3), _art("c", 24 * 8)])}
    at, arts = news_data.fallback(q, 7, [older, newer], NOW)
    assert at == NOW - pd.Timedelta(hours=2)
    assert [a["title"] for a in arts] == ["b"]


def test_fallback_too_old_or_missing():
    q = "q"
    old = {q: (NOW - pd.Timedelta(days=3), [_art("a", 80)])}
    assert news_data.fallback(q, 7, [old], NOW) is None
    assert news_data.fallback("other", 7, [old], NOW) is None


def test_job_keeps_refused_searches(tmp_path, monkeypatch):
    qs = [news_data.gdelt_query(e) for e in news_data.NEWS_QUERIES.values()]
    earlier = NOW - pd.Timedelta(hours=6)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(save_news.update({}, {q: [_art("old", 7)] for q in qs}, earlier)))
    monkeypatch.setattr(save_news, "PATH", str(p))

    def fetch(q, days):
        if q == qs[0]:
            return [_art("new", 1)]
        raise news_data.NewsError("GDELT limited the request rate (HTTP 429)")

    assert save_news.main(fetch=fetch, now=NOW, sleep=lambda s: None) == 1
    saved = news_data.load_saved(str(p))
    assert saved[qs[0]][0] == NOW and saved[qs[0]][1][0]["title"] == "new"
    for q in qs[1:]:
        assert saved[q][0] == earlier and saved[q][1][0]["title"] == "old"


def test_job_all_refused_leaves_file(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text("{}")
    monkeypatch.setattr(save_news, "PATH", str(p))

    def fetch(q, days):
        raise news_data.NewsError("GDELT limited the request rate (HTTP 429)")

    assert save_news.main(fetch=fetch, now=NOW, sleep=lambda s: None) == 0
    assert p.read_text() == "{}"
