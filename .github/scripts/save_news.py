"""Update data/news_saved.json, the saved copy of GDELT's answer to the four commodity headline searches, which the
Market page shows when GDELT answers the hosted app with HTTP 429. Run by .github/workflows/save-news.yml.

Each search is saved with its own time. A search GDELT refuses keeps its earlier saved answer, so a refused run never
empties the file and never fails the job; when every search is refused the run ends with a warning in the Actions log.
"""
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import news_data  # noqa: E402

PATH = news_data.SAVED_PATH
ROUNDS = 3            # a refused search is asked again in the next round
ROUND_WAIT_S = 60     # 14 Nov 2026: GDELT refused one of four searches and answered the others


def update(old, results, now):
    """The new file content: `old` (the file's dict) with each query in `results` ({query: articles}) replaced and
    stamped with `now`; queries not in `results` keep their earlier entry."""
    queries = dict((old or {}).get("queries") or {})
    for q, arts in results.items():
        queries[q] = {"saved_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "articles": news_data.articles_to_json(arts)}
    return {"source": "GDELT Project DOC 2.0 API (https://www.gdeltproject.org/)", "days": news_data.NEWS_DAYS,
            "queries": queries}


def main(fetch=news_data.fetch, now=None, sleep=time.sleep):
    now = now or pd.Timestamp.now(tz="UTC")
    try:
        with open(PATH, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    results = {}
    todo = dict(news_data.NEWS_QUERIES)
    for attempt in range(ROUNDS):
        if attempt:
            sleep(ROUND_WAIT_S)
        for name, expr in list(todo.items()):
            q = news_data.gdelt_query(expr)
            try:
                results[q] = fetch(q, news_data.NEWS_DAYS)
                print(f"{name}: {len(results[q])} articles")
                del todo[name]
            except news_data.NewsError as e:
                print(f"{name}: refused, round {attempt + 1} of {ROUNDS} ({e})")
        if not todo:
            break
    if not results:
        print("::warning::GDELT refused every search; the saved copy is unchanged.")
        return 0
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(update(old, results, now), f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    return len(results)


if __name__ == "__main__":
    main()
