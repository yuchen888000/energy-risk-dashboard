"""Market page: country exposure, recent headlines, statistical checks and an AI summary.

Designed with data and analytics firms in mind.
"""
import html
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as req
import streamlit as st

import common
import news_data
from country_data import (EU27, MODES, available_years, dependency_key, is_provisional, load_country_data,
                          measure_column, mentions_country)

ctx = common.context()
selected_commodity = ctx.selected_commodity
commodity = ctx.commodity
compare_label = ctx.compare_label
start_date, end_date = ctx.start_date, ctx.end_date
CARBON_TICKER = ctx.carbon["ticker"]

NEWS_DAYS = 7          # headlines older than this are not "current"
PER_SITE_CAP = 3       # at most this many headlines from one site, so no single publisher dominates
MAX_HEADLINES = 15     # the recent headlines list shows at most this many
CHECK_DAYS = 252       # base rates of the statistical checks: the last 252 trading days
VOL_LO_PCT, VOL_HI_PCT = 5, 95       # volatility and correlation checks: percentiles of the last 3 years
GARCH_HI, GARCH_LO = 1.30, 0.70      # GARCH forecast vs the 3-year average 30-day volatility
TAIL_MULT = 2.0                      # tail check: loss beyond 2x the 3-year 99% VaR
ROLLING_FUTURES = ('TTF=F', 'CL=F', 'BZ=F')
AI_MODEL = "claude-sonnet-5-5"       # Anthropic model ID of the AI interpretation
with st.sidebar.expander("Methodology: Market page"):
    st.markdown("""
    - **Country exposure**: Eurostat data only (nrg_ind_id, nrg_bal_c, nrg_ind_ren,
      env_air_gge, nama_10_gdp), no hand-entered value and no composite score. For gas
      and oil, countries are ranked by net imports of that fuel as a share of all their
      energy: the fuel's share of gross available energy times its import dependency.
      For carbon, by greenhouse-gas emissions per euro of GDP at 2020 prices. Values Eurostat
      flags as provisional are marked (p). Storage, supplier concentration and LNG capacity
      are not included.
    - **Recent headlines**: from the GDELT Project's DOC 2.0 API (gdeltproject.org), which
      allows any use with a citation and a link. One search over the last 7 days in
      English-language sources (the commodity's own phrases: for gas "natural gas", LNG, TTF
      and similar with a European place, for carbon "EU ETS", "carbon price" and similar),
      then the headline filter below; at most 15 headlines and 3 per site, duplicates
      removed. The page says how many of the headlines returned it kept. The time shown is when
      GDELT first saw the article. Headlines are listed, not scored: a tone score of the
      wording is not a price direction ("gas prices surge" reads as negative). Google News
      and Bing News feeds are not used: their terms allow them only in a personal feed reader.
    - **Which headlines count**: a commodity keyword in the headline ("barrel" only as a price
      unit: "a barrel", "per barrel", "barrels a day"); for TTF an explicit natural-gas term
      (natural gas, LNG, TTF, gas storage, gas supply, Gazprom, Equinor and similar) or "gas"
      with a European place name, and on the TTF list a European place or a European gas name
      (TTF, NBP, Gazprom, Equinor, Nord Stream, Groningen) in the headline itself. Dropped: US retail fuel stories ("gas prices" at the pump),
      share tips ("stocks to watch", "investors should", shares rising or falling, and "stocks"
      with a signal such as "stocks for", "dividend", "to buy" or "top 5 ... stocks" unless a
      storage word such as storage, inventories, injection or full is there too), pages that
      are not news (exchange product pages, bare report titles), stock-market, index, fund and
      crypto stories (Wall Street, Sensex, ETFs, bitcoin, share-price technicals, foreign portfolio
      flows), and, except for WTI,
      US-domestic stories with no link to Europe. WTI is a NYMEX contract, so US inventory
      and production stories are kept for it.
    - **Statistical checks**: volatility above the 95th or below the 5th percentile of the
      last 750 trading days up to the end date (3 years; longer where stale prices of a
      suspension are excluded, shorter where the data start later, and the page then gives
      the dates), GARCH day-10 daily volatility forecast 30%
      above or below the average 30-day volatility of that window (the GARCH model for this
      check is fitted on the returns of that window), 30-day correlation outside its 5th to
      95th percentile range over its last 750 values, and daily losses beyond 2x the window's
      99% VaR (Brent roll days excluded).
      Nothing after the end date is used. Each check shows on how many of the last 252 trading
      days with a price it fired, and the dates those 252 days run from and to. With fewer than 20 returns in the latest
      30-day window (after a suspension) the volatility checks give no value.
    - **AI interpretation** (optional): the statistics on this page summarised in three
      sentences by Claude via the Anthropic API. It sees no fundamentals. A fixed rule removes
      a sentence that asks the reader to watch whether a value moves past a level it is
      already past, when the level is written as a number as on this page (annualised or
      daily volatility, or a correlation with its sign, to the page's rounding, also at the
      end of a sentence); other wordings are not caught. It is given no headlines, and with an
      end date in the past no event after that date.
    """)

st.title("Market Intelligence")
st.markdown(f"Country exposure, recent headlines and statistical checks for **{selected_commodity}**")

core = common.compute_core(commodity['ticker'], ctx.compare_ticker, start_date, end_date)
if core is None:
    st.warning("Please select a longer time range (at least 30 days of data required).")
    st.stop()

df_analysis = core['df_analysis']
latest_vol, avg_vol = core['latest_vol'], core['avg_vol']
returns_clean = core['returns_clean']
var_95 = core['var_95']              # last 250 trading days
# The 3 years (common.CHECK_WINDOW trading days) up to the end date, from compute_core's download: the window
# of this page's checks. Nothing after the end date is used, so a past end date is judged as it stood then.
ref3y = core['ref3y']
# The end date is in the past when it is before today in Berlin (the default end date is Berlin's today).
_past_end = end_date < common.berlin_today()

# Reference level for the GARCH check: the 3-year average 30-day volatility, the same window
# as the volatility check, so it does not move with the sidebar start date. When a suspension pushed the window
# back beyond 3 years (CARB.L across 2020-21), or when the history up to the end date holds fewer than 750 values
# (TTF to 30 Jun 2020: 656), the page names it by its trading days instead ("750-day", "656-day").
_ref_named = ref3y['long'] or ref3y['short']
ref_short = f"{ref3y['n']}-day" if _ref_named else "3-year"
garch_ref = ref3y['mean']
garch_ref_name = f"{ref_short} average 30-day volatility"
# The GARCH model for the check is fitted on the returns of the same 3 years. Fitted on the whole
# sidebar range (from 2020, with the 2021-22 crisis), the model's long-run level sits well above
# the 3-year average (TTF on 16 Oct 2026: 6.8% against 3.5%), so the forecast would be pulled
# towards a level from a different period than its reference.
garch_returns = ref3y['rets']
garch = common.fit_garch(garch_returns, garch_ref)
garch_forecast_10d = garch['forecast_10d'] if garch is not None else None

# ─── Escaping headline text ───
# Headlines are scraped text. Two dollar signs in one headline ("erase $1 trillion ... to $110")
# are read as a math formula by both matplotlib and st.markdown, so "$" is escaped before display.
def md_text(s):
    """Headline text safe for st.markdown: "$" shown literally, no stray link brackets."""
    return (str(s).replace("\\", "\\\\").replace("$", "\\$")
            .replace("[", "\\[").replace("]", "\\]"))


# ─── Headline relevance ───
def _word_pattern(words, flags=re.IGNORECASE):
    """Whole-word match for any of `words`, allowing a plural ending ("emission" to "emissions").
    Substring matching would let "ETS" match "markets" and "oil" match "turmoil"."""
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z])(?:{alt})(?:s|es)?(?![A-Za-z])", flags)


# A headline about the United States with no link to Europe (or to the global market) is
# dropped: US federal/state policy, pump prices, US production and inventory data.
_US = [_word_pattern(["US", "U.S.", "USA"], flags=0),
       _word_pattern(["America", "American", "Trump", "Biden", "White House", "Congress", "Senate",
                      "Republican", "Democrat", "EPA", "Interior Department", "Energy Department",
                      "Energy Secretary", "federal", "EIA", "Strategic Petroleum Reserve", "shale",
                      "Permian", "gasoline", "at the pump", "governor", "Texas", "California", "Alaska",
                      "Louisiana", "New York", "Pennsylvania", "North Dakota", "New Mexico", "Oklahoma",
                      "Colorado", "Gulf of Mexico", "Wall Street"])]
_EUROPE_OR_GLOBAL = _word_pattern([
    "Europe", "European", "EU", "Brussels", "eurozone", "euro area", "UK", "Britain", "British",
    "Germany", "German", "France", "French", "Italy", "Italian", "Spain", "Spanish", "Netherlands",
    "Dutch", "Norway", "Norwegian", "Poland", "Polish", "Austria", "Belgium", "Russia", "Russian",
    "Ukraine", "TTF", "ETS", "Nord Stream", "North Sea", "Brent", "OPEC", "Hormuz", "Suez",
    "Red Sea", "global", "world"])
# US LNG exports are Europe's largest source of LNG, so they are kept even without "Europe".
_LNG_EXPORT = (_word_pattern(["LNG"]),
               _word_pattern(["export", "exporter", "cargo", "cargoe", "shipment", "terminal"]))


def is_us_domestic(headline):
    if not any(p.search(headline) for p in _US):
        return False
    if _EUROPE_OR_GLOBAL.search(headline):
        return False
    return not all(p.search(headline) for p in _LNG_EXPORT)


# Gas (TTF): "gas" alone is not enough, because in American English it also means petrol
# ("gas prices" at the pump). A TTF headline needs an explicit natural-gas term, or the word
# "gas" together with a European place or market name.
_GAS_EXPLICIT = _word_pattern([
    "natural gas", "nat-gas", "natgas", "LNG", "TTF", "NBP", "gas storage", "gas reserve",
    "gas inventory", "gas inventories", "gas stock", "gas supply", "gas supplies", "gas shortage",
    "gas production", "gas output", "gas field", "gas pipeline",
    "gas flow", "gas market", "gas import", "gas export", "gas demand", "gas transit", "gas hub",
    "regasification", "Gazprom", "Equinor", "Nord Stream", "Groningen"])
_GAS_WORD = _word_pattern(["gas"])
_EUROPE_GEO = _word_pattern([
    "Europe", "European", "EU", "Brussels", "eurozone", "euro area", "UK", "Britain", "British",
    "Germany", "German", "France", "French", "Italy", "Italian", "Spain", "Spanish", "Netherlands",
    "Dutch", "Norway", "Norwegian", "Poland", "Polish", "Austria", "Austrian", "Belgium", "Belgian",
    "Ukraine", "Ukrainian", "Russia", "Russian", "Moscow", "Kremlin", "North Sea"])

# The TTF list itself (not the per-country lists, which already need the country) also needs a European subject in
# the headline: a European place, or a name that only belongs to the European gas market. GDELT's search matches
# the European words anywhere in the article, so without this the list fills with LNG stories from Canada or Asia.
_EUROPE_GAS = _word_pattern(["TTF", "NBP", "Gazprom", "Equinor", "Nord Stream", "Groningen"])


def is_european_gas(headline):
    return bool(_EUROPE_GEO.search(headline) or _EUROPE_GAS.search(headline))


# US retail fuel stories: "gas prices" meaning petrol at the pump. Dropped for every commodity.
_GAS_PRICES = _word_pattern(["gas price", "gasoline price"])
_RETAIL_STRONG = _word_pattern(["gallon", "AAA", "GasBuddy", "motorist", "Energy Secretary", "Wright"])
_RETAIL_WEAK = _word_pattern(["gasoline", "pump", "at the pump"])

# Pages that are not news: exchange product pages ("Brent Last Day Financial Overview" from CME,
# TradingView "Trade Ideas" pages) and bare report titles ("Oil Market Report - October 2026").
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December"
           "|Q[1-4]")
_NON_NEWS = [
    re.compile(r"\b(?:Financial Overview|Contract Specs?|Contract Specifications|Trade Ideas)\b", re.I),
    re.compile(r"(?:Quotes|Margins|Settlements|Volume (?:&|and) Open Interest)\s*$", re.I),
    re.compile(r"^[^.!?]{0,60}\b(?:Report|Outlook|Review|Bulletin|Monitor|Factsheet|Fact Sheet)"
               r"\s*[-:|,\u2013\u2014]\s*(?:" + _MONTHS + r")\s*[-\u2013]?\s*\d{4}\s*$", re.I),
]


def is_non_news(headline):
    return any(p.search(headline) for p in _NON_NEWS)


# Share tips and stock picks: the tone of "3 European Gas Stocks Investors Should Watch" is about equities,
# not about the commodity. A headline about shares rising or falling is a company story too. "Gas stocks",
# "crude stocks" and "oil stocks" also mean inventories ("European gas stocks are falling faster than expected"),
# so a headline with "stocks" is a stock pick only when it has a recommendation signal ("stocks for", "dividend",
# "to buy", "to watch", "top 5 ... stocks", "10 best ... stocks") and no storage signal (storage, inventories,
# injection, withdrawal, full, bcm, TWh, draw, build). A tip phrase ("stocks to watch", "investors should") and
# shares rising or falling are dropped in any case.
_SHARE_MOVE = ("rise|rises|rose|fall|falls|fell|jump|jumps|jumped|slide|slides|slid|drop|drops|dropped"
               "|surge|surges|surged|climb|climbs|climbed|tumble|tumbles|tumbled|gain|gains|gained"
               "|sink|sinks|sank|soar|soars|soared|slump|slumps|slumped|plunge|plunges|plunged")
_STOCK_TIP = [
    re.compile(r"\b(?:stocks?|shares?|equities)\s+to\s+(?:watch|buy|own|consider|sell)\b", re.I),
    re.compile(r"\binvestors?\s+should\b", re.I),
    re.compile(r"\bshares\s+(?:" + _SHARE_MOVE + r")\b", re.I),
]
_STOCKS_WORD = re.compile(r"\b(?:stocks|shares|equities)\b", re.I)
_PICK_SIGNAL = [
    re.compile(r"\bstocks\s+for\b", re.I),
    re.compile(r"\bdividends?\b", re.I),
    re.compile(r"\bto\s+(?:buy|watch)\b", re.I),
    # "Top 5 Gas Stocks", "Top 10 oil stocks": at most two words between the number and "stocks", so
    # "Top 3 reasons European gas stocks are falling" is not a list of shares.
    re.compile(r"\btop\s+(?:\d+|three|four|five|ten)\s+(?:[A-Za-z-]+\s+){0,2}stocks\b", re.I),
    # "5 dividend stocks", "10 best energy stocks", "top energy stocks": a number, top, best, cheap or dividend
    # before the word, "gas", "crude" or "oil" not directly before it.
    re.compile(r"\b(?:\d+|three|four|five|top|best|cheap|dividend)\s+(?:[A-Za-z-]+\s+){0,3}"
               r"(?<!\bgas )(?<!\bcrude )(?<!\boil )stocks\b", re.I),
]
_STORAGE_SIGNAL = re.compile(r"(?<![A-Za-z])(?:storage|inventory|inventories|injections?|withdrawals?|full|bcm|TWh"
                             r"|draws?|builds?)(?![A-Za-z])", re.I)


# Equity-market stories that mention oil in passing ("Sharp drops for AI stocks overwhelm markets worldwide as oil
# prices jump", "Sensex, Nifty extend losses as crude surges"), and fund or index stories ("Defiance Oil Enhanced
# Options Income ETF to go ex-dividend"): the subject is a stock market or a fund, not the commodity.
_EQUITY_MARKET = re.compile(r"\b(?:Wall Street|Nasdaq|S&P 500|Dow Jones|Big Tech|stock markets?|equity markets?"
                            r"|Sensex|Nifty|FTSE|DAX|CAC 40|Hang Seng|Nikkei|bullion"
                            r"|ETFs?|ex-dividend|dividend yield|options income"
                            r"|(?:AI|tech|bank|chip) stocks"
                            # Crypto, single-share technicals and equity flows ("With oil prices rising, Bitcoin ...",
                            # "UKOG share price crosses above fifty day moving average", "FPIs withdraw Rs 48,056 cr
                            # from equities", "Fed policy, Q2 earnings, crude oil prices among key market triggers").
                            r"|bitcoin|crypto(?:currency|currencies)?|share price|moving average|equities"
                            r"|FPIs?|foreign portfolio investors?|market triggers)\b", re.I)


# Index and equity moves ("European stocks edge lower as oil prices rise", "Indian markets snap a 1-week rally"):
# the subject is the stock market. "gas stocks", "crude stocks" and "oil stocks" mean inventories, so a move right
# after one of those words, or any storage signal in the headline, does not count.
_EQUITY_MOVE = re.compile(r"(?<!\bgas )(?<!\bcrude )(?<!\boil )\b(?:stocks|equities|markets)\s+"
                          r"(?:" + _SHARE_MOVE + r"|edge|edges|edged|snap|snaps|snapped|extend|extends|extended"
                          r"|close|closes|closed|open|opens|opened|rally|rallies|rallied)\b", re.I)


def is_stock_tip(headline):
    if any(p.search(headline) for p in _STOCK_TIP) or _EQUITY_MARKET.search(headline):
        return True
    if _EQUITY_MOVE.search(headline) and not _STORAGE_SIGNAL.search(headline):
        return True
    return bool(_STOCKS_WORD.search(headline) and any(p.search(headline) for p in _PICK_SIGNAL)
                and not _STORAGE_SIGNAL.search(headline))


def is_us_retail_fuel(headline):
    """"Gas prices" with a US pump-price marker. "Gasoline" or "pump" alone does not count when
    the headline is about Europe ("gas prices, but also gasoline and diesel ... across the EU")."""
    if not _GAS_PRICES.search(headline):
        return False
    if _RETAIL_STRONG.search(headline):
        return True
    return bool(_RETAIL_WEAK.search(headline)) and not _EUROPE_GEO.search(headline)


def is_gas_headline(headline):
    return bool(_GAS_EXPLICIT.search(headline)
                or (_GAS_WORD.search(headline) and _EUROPE_GEO.search(headline)))


def drop_reason(headline, keyword_re, gas=False, keep_us=False):
    """Why a headline is not used ("" if it is kept). The same rules for every feed. `keep_us` (WTI) keeps
    US-domestic stories: WTI is a NYMEX contract, so US inventory and production news bears on it."""
    if is_non_news(headline):
        return "not a news story (product or report-title page)"
    if is_us_retail_fuel(headline):
        return "US retail fuel prices"
    if is_stock_tip(headline):
        return "share tip or stock story"
    if gas:
        if not is_gas_headline(headline):
            return "no explicit natural-gas term"
    elif not keyword_re.search(headline):
        return "no commodity keyword"
    if not keep_us and is_us_domestic(headline):
        return "US-domestic, no link to Europe"
    return ""


def is_relevant(headline, keyword_re, gas=False, keep_us=False):
    """Keep a headline only if drop_reason finds nothing against it."""
    return drop_reason(headline, keyword_re, gas, keep_us) == ""


# With an end date in the past, every headline on the page is still today's: said once, under each news heading.
NEWS_DATE_NOTE = (f"Headlines are today's ({common.berlin_today():%d %b %Y}), not as of {end_date:%d %b %Y}."
                  if _past_end else "")


def news_date_note():
    if NEWS_DATE_NOTE:
        st.caption(NEWS_DATE_NOTE)


commodity_kw_re = _word_pattern(commodity['keywords'])
IS_GAS = commodity['ticker'] == 'TTF=F'
KEEP_US = commodity['ticker'] == 'CL=F'   # WTI: US-domestic stories are kept


def dedup_key(text):
    """Headline reduced to lower-case words, so the same story from two searches or sites counts once. A label before
    a colon is left out when at least five words follow it, so "B.C. Election Analysis: Questions raised about ..."
    and "B.C. Election: Questions raised about ..." (one story, two sites) count once."""
    head, sep, tail = text.rpartition(": ")
    if sep and len(tail.split()) >= 5:
        text = tail
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


# GDELT asks for a citation and a link wherever its data are used (news_data.py).
GDELT_NOTE = ("Source: the GDELT Project (https://www.gdeltproject.org/), English-language sources; the time is "
              "when GDELT first saw the article (UTC).")


@st.cache_data(ttl=7200, show_spinner="Fetching headlines from GDELT...")
def gdelt_articles(query, days):
    """news_data.fetch, cached for 2 hours. A NewsError is not cached, so the next run asks again."""
    return news_data.fetch(query, days)


def gdelt_try(query, days):
    """(articles, "") or ([], the reason in words) when GDELT gives nothing back this time."""
    try:
        return gdelt_articles(query, days), ""
    except news_data.NewsError as e:
        return [], str(e)


def headline_list_md(rows):
    """A markdown list: the headline linked to the article, then the site and the time GDELT saw it."""
    out = []
    for a in rows:
        url = a['url'].replace('(', '%28').replace(')', '%29').replace(' ', '%20')
        head = f"[{md_text(a['title'])}]({url})" if url.startswith('http') else md_text(a['title'])
        out.append(f"- {head} · *{md_text(a['domain'])}, {a['seen']:%d %b %H:%M} UTC*")
    return "\n".join(out)


# ─── Section 1: European country exposure ───
country_mode = dependency_key(selected_commodity)
M = MODES[country_mode]
mcol = measure_column(country_mode)
country_df, country_failed, country_updated = load_country_data()
country_years = available_years(country_df, country_mode)

st.subheader("European Country Exposure")
st.write(f"**{M['question']}**")
if country_failed:
    st.warning("Eurostat could not be reached for: " + ", ".join(country_failed)
               + ". Countries without those inputs are not ranked.")

cr_df = pd.DataFrame()
if not country_years:
    st.warning("Eurostat has no complete year for this measure right now, so the country table is not shown.")
else:
    if len(country_years) > 1:
        selected_year = st.slider("Select Year", min_value=country_years[0], max_value=country_years[-1],
                                  value=country_years[-1], step=1)
    else:
        selected_year = country_years[0]
    st.caption(f"Years {country_years[0]} to {country_years[-1]}: every year from {country_years[0]} for which "
               "Eurostat publishes the EU-27 value of this measure. Last Eurostat updates: "
               + ", ".join(f"{k} {v}" for k, v in country_updated.items() if v) + ". (p): Eurostat flags the "
               "value as provisional; a measure computed from a provisional input is marked too, and a country "
               "marked (p) has a provisional ranking measure.")
    if _past_end:
        st.caption("Eurostat annual data; the year is chosen above and does not follow the end date."
                   if len(country_years) > 1 else
                   f"Eurostat annual data for {selected_year}, the only year available; it does not follow the end "
                   "date.")

    yr_df = country_df[country_df['Year'] == selected_year]
    eu = yr_df[yr_df['Country'] == EU27].iloc[0]
    countries_yr = yr_df[yr_df['Country'] != EU27]
    ranked = countries_yr[countries_yr[mcol].notna()].sort_values(mcol, ascending=False).reset_index(drop=True)
    not_ranked = countries_yr.loc[countries_yr[mcol].isna(), 'Country'].tolist()
    n_ranked = len(ranked)

    if country_mode in ('gas', 'oil'):
        fuel = country_mode
        cols = {
            'Country': ranked['Country'],
            M['measure']: ranked[f'{fuel}_exposure'],
            M['share']: ranked[f'{fuel}_share'],
            M['dep']: ranked[f'dep_{fuel}'],
            'Total energy import dependency (%)': ranked['dep_total'],
            'Renewable share (%)': ranked['ren'],
        }
        st.markdown(
            f"Ranked by **net {fuel} imports as a share of all energy** = {fuel} share of gross available "
            f"energy × {fuel} import dependency. EU-27 {selected_year}: {fuel} is "
            f"**{eu[f'{fuel}_share']:.1f}%** of gross available energy, **{eu[f'dep_{fuel}']:.1f}%** of it "
            f"is imported (net), so net {fuel} imports are **{eu[f'{fuel}_exposure']:.1f}%** of all energy.")
        st.caption(
            f"Gross available energy is all energy a country has available (production + imports − exports "
            f"+ stock changes), from Eurostat nrg_bal_c in ktoe; import dependency is Eurostat nrg_ind_id "
            f"(net imports ÷ gross available energy of that fuel; Eurostat's fuel label: \"{M['fuel']}\"). "
            f"The product of share and dependency tells how much of a country's total "
            f"energy supply is {fuel} bought abroad, so a country that imports all its {fuel} but uses little "
            f"of it ranks low. Dependency is shown as published: below 0% means a net exporter, above 100% "
            f"means imports also refilled stocks. In the product a net exporter counts as 0.")
        if fuel == 'gas':
            st.caption("**Not included:** gas storage levels (EU filling targets under Regulation (EU) 2022/1032 as amended), how "
                       "concentrated imports are by supplier, LNG import capacity, pipeline routes, and how "
                       "gas use splits between power, industry and heating. A country can rank high here and "
                       "still be well protected by storage or diverse suppliers.")
        else:
            st.caption("**Not included:** emergency oil stocks (EU Directive 2009/119/EC; net-importing IEA members "
                       "also hold 90 days of net imports), supplier "
                       "concentration and refinery capacity. Gross available energy includes fuel for "
                       "international shipping (maritime bunkers), which raises the oil share of bunkering "
                       "hubs such as Malta, the Netherlands and Belgium.")
    else:
        cols = {
            'Country': ranked['Country'],
            M['measure']: ranked['ghg_int'],
            'Renewable share (%)': ranked['ren'],
        }
        st.markdown(
            f"Ranked by **greenhouse-gas emissions per million euro of GDP at 2020 prices**. "
            f"EU-27 {selected_year}: **{eu['ghg_int']:.0f} tCO2e per M€**.")
        st.caption(
            "What it measures: total greenhouse-gas emissions excluding land use and the memo items "
            "(international aviation and maritime bunkers) (Eurostat env_air_gge, TOTX4_MEMO) divided by GDP in chain-linked volumes (nama_10_gdp, CLV20_MEUR), so a fall over "
            "time is a fall in emissions per unit of real output, not an effect of inflation. Levels across "
            "countries are converted at 2020 exchange rates, not purchasing power, which makes lower-income "
            "economies look more intensive. Renewable share of gross final energy consumption: nrg_ind_ren. "
            "**Not included:** which emissions are covered by the EU ETS and so priced by EUAs (power, industry, "
            "aviation, maritime), free allocation and carbon leakage. A broad indicator of how carbon-intensive "
            "an economy is, not the cost of EUAs to it.")

    # Display column -> load_country_data column, for the provisional flags.
    _col_source = {M['measure']: mcol, 'Renewable share (%)': 'ren',
                   'Total energy import dependency (%)': 'dep_total'}
    if country_mode in ('gas', 'oil'):
        _col_source.update({M['share']: f'{country_mode}_share', M['dep']: f'dep_{country_mode}'})
    cr_df = pd.DataFrame(cols)
    cr_df.index = cr_df.index + 1   # rank
    cr_df.index.name = 'Rank'
    if not_ranked:
        st.caption(f"Not ranked for {selected_year} (Eurostat has no value for at least one input of the "
                   "measure): " + ", ".join(not_ranked) + ".")

    if cr_df.empty:
        st.warning("No country has the Eurostat inputs for this year.")
    else:
        cr_col1, cr_col2 = st.columns([2, 1])
        with cr_col1:
            st.write(f"**{M['measure']}, {selected_year}: all {n_ranked} countries with data**")
            col_cfg = {M['measure']: st.column_config.ProgressColumn(
                M['measure'], format="%.1f", min_value=0.0, max_value=float(cr_df[M['measure']].max()))}
            # A value Eurostat does not publish (e.g. Turkey's renewable share) shows as "n/a", as in
            # the detail table; the columns stay numeric, so sorting still works.
            _num_cols = [c for c in cr_df.columns if c not in ('Country', M['measure'])]
            cr_df[_num_cols] = cr_df[_num_cols].apply(pd.to_numeric, errors='coerce')
            cr_view = cr_df.style.format("{:.1f}", subset=_num_cols + [M['measure']], na_rep="n/a")
            # Provisional values (Eurostat flag "p") get "(p)" in the display only: the cells stay numeric,
            # so sorting, the ranking and the CSV are unchanged. The ranking column is a progress bar, whose
            # number format overrides the styler, so its flag is shown next to the country name.
            for _c in _num_cols + ['Country']:
                _src = _col_source[M['measure'] if _c == 'Country' else _c]
                _rows = [i for i, (_, r) in zip(cr_df.index, ranked.iterrows()) if is_provisional(r, _src)]
                if _rows:
                    cr_view = cr_view.format("{} (p)" if _c == 'Country' else "{:.1f} (p)",
                                             subset=pd.IndexSlice[_rows, [_c]], na_rep="n/a")
            # Tall enough to show every country without scrolling (35 px per row plus the header).
            st.dataframe(cr_view, width="stretch", height=(len(cr_df) + 1) * 35 + 3, column_config=col_cfg)
            if ctx.is_carbon:
                # Which countries EUAs cover (European Commission, About the EU ETS, read 2026-10-23).
                _outside = [c for c in ("Turkey", "Switzerland", "United Kingdom") if c in set(cr_df['Country'])]
                if _outside:
                    st.caption("The EU ETS covers the EU-27, Iceland, Liechtenstein and Norway; "
                               + (", ".join(_outside[:-1]) + " and " + _outside[-1] if len(_outside) > 1 else _outside[0])
                               + (" is" if len(_outside) == 1 else " are")
                               + " not in it (Switzerland has its own ETS, linked to the EU ETS since 2020).")

        with cr_col2:
            # Alphabetical options and a default taken from the latest year, so moving the year
            # slider does not reset the chosen country.
            _options = sorted(cr_df['Country'])
            _latest = country_df[(country_df['Year'] == country_years[-1]) & (country_df['Country'] != EU27)]
            _ranked_latest = _latest.sort_values(mcol, ascending=False)['Country']
            # On the carbon page the default is the highest-ranked country inside the EU ETS (Turkey ranks first
            # but EUAs do not cover it).
            if ctx.is_carbon:
                _ranked_latest = _ranked_latest[~_ranked_latest.isin(["Turkey", "Switzerland", "United Kingdom"])]
            _default = _ranked_latest.iloc[0]
            selected_country = st.selectbox("Select Country for Detail", _options,
                                            index=_options.index(_default) if _default in _options else 0)
            row = ranked[ranked['Country'] == selected_country].iloc[0]
            rank = int(cr_df.index[cr_df['Country'] == selected_country][0])
            st.markdown(f"### {selected_country} ({selected_year})")
            _p = " (p)" if is_provisional(row, mcol) else ""
            st.metric(M['measure'], f"{row[mcol]:.1f}{_p}", help=f"Rank {rank} of {n_ranked}; EU-27 "
                      f"{eu[mcol]:.1f}{' (p)' if is_provisional(eu, mcol) else ''}."
                      + (" (p): provisional in Eurostat." if _p else ""))
            st.caption(f"Rank {rank} of {n_ranked} countries with data.")

            def _fmt(col, f="{:.1f}"):
                v = row[col]
                if v is None or pd.isna(v):
                    return "n/a"
                return f.format(v) + (" (p)" if is_provisional(row, col) else "")
            if country_mode in ('gas', 'oil'):
                fuel = country_mode
                detail_rows = [
                    (M['share'], _fmt(f'{fuel}_share')),
                    (M['dep'], _fmt(f'dep_{fuel}')),
                    (f"Gross available {fuel} (ktoe)", _fmt(f'gae_{fuel}', "{:,.0f}")),
                    ("Gross available energy, all (ktoe)", _fmt('gae_total', "{:,.0f}")),
                    ("Total energy import dependency (%)", _fmt('dep_total')),
                    ("Renewable share (%)", _fmt('ren')),
                ]
            else:
                detail_rows = [
                    ("GHG emissions excl. LULUCF (kt CO2e)", _fmt('ghg', "{:,.0f}")),
                    ("GDP, 2020 prices (M€)", _fmt('gdp_clv', "{:,.0f}")),
                    ("Renewable share (%)", _fmt('ren')),
                ]
            # A two-column table wraps inside the narrow panel, where side-by-side metrics overlap.
            st.markdown("| Eurostat input | Value |\n|---|---:|\n"
                        + "\n".join(f"| {k} | {v} |" for k, v in detail_rows))

        # Bar chart of every ranked country, shaded by the measure itself.
        fig_cr, ax_cr = plt.subplots(figsize=(14, max(4, 0.28 * n_ranked + 1)))
        vals = cr_df[M['measure']].astype(float)
        shade = plt.cm.Blues(0.3 + 0.7 * vals / vals.max()) if vals.max() > 0 else 'steelblue'
        ax_cr.barh(range(n_ranked), vals, color=shade, height=0.7)
        ax_cr.set_yticks(range(n_ranked))
        ax_cr.set_yticklabels([c + (" (p)" if is_provisional(r, mcol) else "")
                               for c, (_, r) in zip(cr_df['Country'], ranked.iterrows())], fontsize=9)
        ax_cr.axvline(x=eu[mcol], color='black', linewidth=1, linestyle='--', alpha=0.6,
                      label=f'EU-27: {eu[mcol]:.1f}')
        ax_cr.set_xlabel(M['measure'])
        ax_cr.set_title(f'{M["measure"]}, {selected_year} (Eurostat, all {n_ranked} countries with data)')
        ax_cr.legend(fontsize=8, loc='lower right')
        ax_cr.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_cr)

        # The same Eurostat inputs for the selected country over every available year.
        trend = (country_df[(country_df['Country'] == selected_country)
                            & country_df['Year'].isin(country_years)]
                 .sort_values('Year').set_index('Year'))
        trend = trend.drop(columns=['Country']).apply(pd.to_numeric, errors='coerce')
        st.write(f"**{selected_country}: inputs over time, {country_years[0]} to {country_years[-1]}**")

        def _has(*cols):
            return any(pd.to_numeric(trend[c], errors='coerce').notna().any() for c in cols)

        def _draw_mix(ax):
            fuel = country_mode
            ax.plot(trend.index, trend[f'{fuel}_share'], 'o-', color='steelblue', label=M['share'])
            ax.plot(trend.index, trend[f'{fuel}_exposure'], 's-', color='navy', label=M['measure'])
            ax.set_ylabel('% of gross available energy')
            ax.set_title(f'{selected_country}: {fuel} in the energy mix')
            ax.legend(fontsize=8)

        def _draw_dep(ax):
            fuel = country_mode
            ax.plot(trend.index, trend[f'dep_{fuel}'], 'o-', color='gray', label=M['dep'])
            ax.axhline(100, color='black', linewidth=0.5, linestyle=':')
            ax.set_ylabel(f'% of gross available {fuel}')
            ax.set_title(f'{selected_country}: {fuel} import dependency (as published)')

        def _draw_int(ax):
            ax.bar(trend.index, trend['ghg_int'], color='gray', alpha=0.7)
            ax.set_ylabel('tCO2e per M€ GDP (2020 prices)')
            ax.set_title(f'{selected_country}: greenhouse-gas intensity of GDP')

        def _draw_ren(ax):
            ax.plot(trend.index, trend['ren'], 'o-', color='steelblue')
            ax.set_ylabel('% of gross final energy consumption')
            ax.set_title(f'{selected_country}: renewable share')

        # (draw function, columns it needs, what is missing when it has no data)
        if country_mode in ('gas', 'oil'):
            _panels = [(_draw_mix, (f'{country_mode}_share', f'{country_mode}_exposure'),
                        f"{country_mode} share of gross available energy (nrg_bal_c)"),
                       (_draw_dep, (f'dep_{country_mode}',), f"{country_mode} import dependency (nrg_ind_id)")]
        else:
            _panels = [(_draw_int, ('ghg_int',), "greenhouse-gas intensity (env_air_gge, nama_10_gdp)"),
                       (_draw_ren, ('ren',), "renewable share (nrg_ind_ren)")]
        _drawn = [p for p in _panels if _has(*p[1])]
        _missing = [p[2] for p in _panels if not _has(*p[1])]
        if _drawn:
            fig_tcr, _axes = plt.subplots(1, len(_drawn), figsize=(14 if len(_drawn) == 2 else 7, 3.5),
                                          squeeze=False)
            for (draw, _, _), ax in zip(_drawn, _axes[0]):
                draw(ax)
                ax.set_xticks(country_years)
            plt.tight_layout()
            st.pyplot(fig_tcr)
        if _missing:
            st.caption(f"Eurostat publishes no {' or '.join(_missing)} for {selected_country} in "
                       f"{country_years[0]} to {country_years[-1]}, so no trend is drawn for it.")

        # Per-country headlines: the same source and filters as the recent headlines section below.
        st.write(f"**{selected_country}: recent energy headlines**")
        news_date_note()
        _c_arts, _c_err = gdelt_try(news_data.build_query(selected_country, "energy", commodity['news_term']),
                                    NEWS_DAYS)
        _energy_re = _word_pattern(commodity['keywords'] + ['energy', 'oil', 'gas', 'carbon', 'power',
                                                            'fuel', 'electricity', 'pipeline', 'LNG',
                                                            'emission', 'climate', 'price', 'supply',
                                                            'tanker', 'refinery', 'fossil', 'renewable',
                                                            'heating', 'Hormuz', 'sanction', 'ETS'])
        country_rows, _seen_c = [], set()
        for _a in _c_arts:
            _h, _k = _a['title'], dedup_key(_a['title'])
            if not _k or _k in _seen_c:
                continue
            if (_energy_re.search(_h) and mentions_country(_h, selected_country)
                    and not is_non_news(_h) and not is_us_retail_fuel(_h) and not is_stock_tip(_h)):
                country_rows.append(_a)
                _seen_c.add(_k)
            if len(country_rows) >= 5:
                break
        if country_rows:
            st.markdown(headline_list_md(country_rows))
            st.caption(GDELT_NOTE)
        elif _c_err:
            st.info(f"No headlines this time: {_c_err}. The page asks again on the next run.")
        else:
            st.info(f"No energy headline from the last {NEWS_DAYS} days mentions {selected_country} by name or adjective.")


# ─── Section 2: Recent headlines ───
st.subheader("Recent Headlines")
st.write(f"{selected_commodity}-related headlines from the last {NEWS_DAYS} days, newest first. They are listed "
         "for context and not scored: the tone of a headline's wording says little about the price.")
news_date_note()

_filter_note = ("US stories are kept for WTI, and the filter does not require a European subject" if KEEP_US else
                "for TTF the headline must name a European place or a European gas name such as TTF, Gazprom or "
                "Equinor" if IS_GAS else
                "the filter drops US domestic stories; it does not require a European subject")
_arts, _news_err = gdelt_try(news_data.gdelt_query(commodity['news_query']), NEWS_DAYS)
# Newest first from GDELT; the caps keep one site from filling the list.
news_rows, _seen, _per_site = [], set(), {}
for _a in _arts:
    _k = dedup_key(_a['title'])
    if not _k or _k in _seen or _per_site.get(_a['domain'], 0) >= PER_SITE_CAP:
        continue
    # Same test as the country headlines: a commodity keyword in it, no US-domestic story (except WTI).
    if not is_relevant(_a['title'], commodity_kw_re, gas=IS_GAS, keep_us=KEEP_US):
        continue
    if IS_GAS and not is_european_gas(_a['title']):
        continue
    _seen.add(_k)
    _per_site[_a['domain']] = _per_site.get(_a['domain'], 0) + 1
    news_rows.append(_a)
    if len(news_rows) >= MAX_HEADLINES:
        break

if news_rows:
    st.markdown(headline_list_md(news_rows))
    st.caption(f"{len(news_rows)} of the {len(_arts)} headlines GDELT returned for this search, at most "
               f"{MAX_HEADLINES} and at most {PER_SITE_CAP} per site, duplicates removed ({_filter_note}). "
               + GDELT_NOTE)
elif _news_err:
    st.info(f"No headlines this time: {_news_err}. The page asks again on the next run.")
else:
    st.info(f"GDELT returned {len(_arts)} headlines for this search, none of which passed the filter "
            f"({_filter_note}).")


# ─── Section 3: Statistical checks ───
_asof = df_analysis.index[-1]
# Every threshold, percentile, count of days fired and the GARCH reference come from ref3y: the 3 years up to
# the end date, from compute_core's download (no second download, nothing after the end date).
ref = ref3y
_ref_end = pd.Timestamp(ref['end'])
# "750 trading days", or "750 trading days with a 30-day volatility" when some days with a price in the window
# have none (the first days after a suspension).
_ref_days = f"{ref['n']} trading days" + (" with a 30-day volatility" if ref['rows'] > ref['n'] else "")
ref3y_span = _ref_days if _ref_named else "3 years"   # the check window, in words
# The counts of days fired cover the last CHECK_DAYS trading days with a price up to the last one; a check
# whose value is missing on some of them (no 30-day volatility right after a suspension) counts fewer.
_count_days = ref['days'][-CHECK_DAYS:]
_count_from = _count_days[0]
_count_span = f"the {len(_count_days)} trading days from {_count_from:%d %b %Y} to {_ref_end:%d %b %Y}"
# The end date falls inside a stale run (CARB.L to 30 Jun 2021): the checks stop at the last price before it.
_stale_after = [(a, b) for a, b in common.stale_periods(commodity['ticker'], start_date, end_date)
                if pd.Timestamp(a) > _asof]
_stale_stop = (f" {commodity['ticker']} prices from {_stale_after[0][0]:%d %b %Y} are excluded as stale, so the checks "
               f"stop at {_asof:%d %b %Y}." if _stale_after else "")
st.subheader("Statistical Checks")
st.write(f"Is anything statistically unusual on {_asof:%d %b %Y}, the last trading day with a price in the selected "
         f"range, compared with this commodity's own {ref3y_span} up to that day? Each check shows on how many of "
         f"{_count_span} it fired, so you can judge how rare a flag is.{_stale_stop}")

# The same volatility statement as the Risk page (common.vol_summary): references that do not roll with the
# market, next to the 3-year checks below.
_vs = common.vol_summary(core, name=selected_commodity, with_roll=True, past_end=_past_end)
st.markdown(f"**{_vs['headline']}**" + (f"  \n{_vs['headline2']}" if _vs['headline2'] else ""))
# The roll days the heading points to, in one line (the same line as the Risk page, common.roll_line).
_roll_line = common.roll_line(commodity['ticker'], core.get('rolls'))
if _roll_line:
    st.caption(_roll_line)


def garch_10d_path(g):
    """The 10-day-ahead GARCH volatility forecast as it stood at the end of each day, from the
    fitted parameters: sigma2(t+10) = omega * (1 + p + ... + p^8) + p^9 * sigma2(t+1), p = alpha + beta.
    sigma(t+1) is the model's conditional volatility for the next day; for the last day it is
    the one-day-ahead forecast. The last value equals the page's forecast."""
    p = g['params']
    pers = p['alpha[1]'] + p['beta[1]']
    nxt = g['conditional_volatility'].shift(-1) ** 2
    nxt.iloc[-1] = float(g['forecast_vol'][0]) ** 2
    const = p['omega'] * sum(pers ** i for i in range(9))
    return np.sqrt(const + pers ** 9 * nxt)


def stale_text(runs):
    """'CARB.L prices from 11 Jun 2020 to 31 Aug 2021' for the stale runs inside a window (from the data)."""
    out = []
    for ticker in dict.fromkeys(t for t, _, _ in runs):
        own = [(a, b) for t, a, b in runs if t == ticker]
        if len(own) <= 3:
            out.append(f"{ticker} prices " + " and ".join(f"from {a:%d %b %Y} to {b:%d %b %Y}" for a, b in own))
        else:
            a, b = max(own, key=lambda x: (x[1] - x[0], x[1]))
            out.append(f"{ticker} prices in {len(own)} stale runs, the longest from {a:%d %b %Y} to {b:%d %b %Y},")
    return " and ".join(out)


def long_window_note(runs):
    """Bracket for a window that spans more than 3 years: which prices were left out."""
    if not runs:
        return " (more than 3 years: days without a price are not counted)"
    return f" ({stale_text(runs)} are excluded as stale)"


if ref['fallback']:
    ref_source = (f"the selected date range, {ref['start']:%d %b %Y} to {_ref_end:%d %b %Y}, {ref['n']} trading "
                  "days (fewer than 250 trading days of history up to the end date)")
    ref_name = "the selected range"
elif ref['short']:
    # Fewer than 750 values up to the end date: the window is what the data has, named by its days and dates.
    ref_source = (f"the {_ref_days} available, {ref['start']:%d %b %Y} to {_ref_end:%d %b %Y}"
                  + (long_window_note(ref['stale']) if ref['long'] else ""))
    ref_name = ref_source
elif ref['long']:
    ref_source = (f"{ref['start']:%d %b %Y} to {_ref_end:%d %b %Y}, {_ref_days}"
                  + long_window_note(ref['stale']))
    ref_name = f"the {_ref_days} from {ref['start']:%d %b %Y} to {_ref_end:%d %b %Y}" \
               + long_window_note(ref['stale'])
else:
    ref_source = (f"{ref['start']:%d %b %Y} to {_ref_end:%d %b %Y}, {_ref_days}: the 3 years up to "
                  "the last day of the selected range")
    ref_name = f"the 3 years to {_ref_end:%d %b %Y}"
# The correlation needs both series, so a gap in the comparison series moves its window too (Brent to
# 30 Jun 2022: from 14 Mar 2018, across CARB.L's suspension; TTF to 30 Jun 2020: 605 values from 09 Jan 2018 to
# 24 Jun 2020, the last day whose 30 rows still hold 20 pairs after CARB.L went stale on 11 Jun). Named separately
# whenever its first day, last day or number of values differs from the volatility window's (three_year_reference's
# corr_own), in one phrase used everywhere. A day with a 30-day correlation need not have both returns itself.
corr_short, corr_name, corr_source = ref_short, ref_name, ""
if ref['corr_own']:
    _cn = len(ref['corr'])
    corr_short = f"{_cn}-day"
    corr_name = (f"the {_cn} days with a 30-day correlation, {ref['corr_start']:%d %b %Y} to "
                 f"{ref['corr_end']:%d %b %Y}"
                 + (long_window_note(ref['corr_stale']) if ref['corr_long'] else ""))
    corr_source = f" The correlation check uses its own window: {corr_name}."


def garch_fit_sentence():
    """The one statement of which returns the GARCH check's model is fitted on, for the caption and the AI input."""
    if garch is None or not len(garch_returns):
        return ""
    a, b, n = garch_returns.index[0], garch_returns.index[-1], len(garch_returns)
    if a.normalize() == pd.Timestamp(ref['start']).normalize() and b.normalize() == _ref_end.normalize():
        out = (f"The GARCH model behind the forecast check is fitted on the {n} daily returns from {a:%d %b %Y} to "
               f"{b:%d %b %Y}, the window of its reference, the {ref_short} average 30-day volatility.")
    else:
        out = (f"The GARCH model behind the forecast check is fitted on {n} daily returns ({a:%d %b %Y} to "
               f"{b:%d %b %Y}); its reference, the average 30-day volatility, covers {ref_source}.")
    if ref['long'] and ref['stale']:
        out += (f" The fit runs across the excluded {stale_text(ref['stale']).rstrip(',')}: the returns before and "
                "after the gap are treated as consecutive days.")
    return out


garch_fit_note = garch_fit_sentence()

vol_lo, vol_hi = np.percentile(ref['vol'], [VOL_LO_PCT, VOL_HI_PCT])
# Rounded to 10 decimals on both sides: the latest volatility is computed on two paths (compute_core's window
# and the reference) that can differ by ~1e-14.
vol_pctile = (float((np.round(ref['vol'].values, 10) < round(float(latest_vol), 10)).mean() * 100)
              if latest_vol is not None else None)
vol_last = ref['vol'][ref['vol'].index >= _count_from]
_of = f"of the {len(vol_last)} trading days to {_ref_end:%d %b %Y}"
# The 30-day correlation of the last day with a price (common.rolling_corr, as on the Risk page); None when its
# window holds fewer than 20 days with both returns, and the caption then says why (common.corr_gap_sentence).
corr_now = core['latest_corr']
corr_gap_note = common.corr_gap_sentence(core)
has_corr_ref = len(ref['corr']) >= 250
has_corr = corr_now is not None and has_corr_ref
if has_corr_ref:
    corr_lo, corr_med, corr_hi = np.percentile(ref['corr'], [VOL_LO_PCT, 50, VOL_HI_PCT])
    corr_last = ref['corr'][ref['corr'].index >= _count_from]
IS_BRENT = commodity['ticker'] == 'BZ=F'
# Brent: on a roll day Yahoo's continuous BZ=F switches to the next contract, so that day's return
# includes the price gap between two contracts. Roll days (common.brent_roll_set: the ICE expiry rule,
# corrected by the roll check of the latest 30-day window, which finds the day Yahoo actually switched) are left
# out of the tail check, both from the 3-year VaR99 and from the last 252 days.
ref_rets_tail = ref['rets']
brent_rolls = pd.DatetimeIndex([])
if IS_BRENT:
    brent_rolls = common.brent_roll_set(df_analysis['Price'].dropna().index, core['rolls']).union(
        common.brent_roll_set(ref['rets'].index, core['rolls']))
    ref_rets_tail = ref['rets'][~ref['rets'].index.isin(brent_rolls)]
var99_3y = float(np.percentile(ref_rets_tail, 1) * 100)
tail_thr = TAIL_MULT * abs(var99_3y)
# The daily-loss count covers the same last 252 trading days of the reference window as the other checks.
recent_all = ref['rets'][ref['rets'].index >= _count_from] * 100
_is_roll = recent_all.index.isin(brent_rolls)
recent_rets = recent_all[~_is_roll]
tail_days = recent_rets[recent_rets < -tail_thr]
roll_excluded = recent_all[_is_roll & (recent_all < -tail_thr)]   # roll days that would have fired
n_roll_recent = int(_is_roll.sum())
# TTF: no contract calendar from Yahoo, so a flagged day on or just after a front-month expiry is
# only marked, not removed.
ttf_flagged = []
if commodity['ticker'] == 'TTF=F' and len(tail_days):
    _exp = common.ttf_expiry_days(ref['rets'].index)
    ttf_flagged = [d for d in tail_days.index if d in _exp]

if IS_BRENT:
    roll_note = (" Brent roll days are excluded from the daily-loss check and its "
                 f"{ref_short} VaR99: the first trading day after each ICE Brent contract's last trading day "
                 "(normally the first trading day of the month), or, within the last 30 days, the day the roll "
                 "check found the continuous series switching contract when Yahoo switched on another day. Further back "
                 "only the rule days are used, so an earlier switch on another day is treated as an ordinary day.")
elif commodity['ticker'] == 'TTF=F':
    roll_note = (" TTF expiry days are not removed: a flagged day on or just after a front-month expiry "
                 "is marked, because part of its move may be the gap between two contracts.")
elif commodity['ticker'] in ROLLING_FUTURES:
    roll_note = (" On a continuous front-month future a contract roll day can produce {move} without a "
                 "real price change; roll days are not removed for this future.")
else:
    roll_note = ""
# In the loss box "such a move" is the loss just described; the caption has no such loss before it.
roll_note_box, roll_note = roll_note.format(move="such a move"), roll_note.format(move="a large daily move")
# WTI: the box names the flagged days, so check them against the expiry rule instead of leaving a general doubt.
if commodity['ticker'] == 'CL=F' and len(tail_days):
    _near = common.rule_roll_days("CL", tail_days.index)
    _far = [d for d in tail_days.index if pd.Timestamp(d).normalize() not in _near]
    _bits = []
    if _far:
        _bits.append(", ".join(f"{d:%d %b %Y}" for d in _far)
                     + (" is" if len(_far) == 1 else " are") + " not a WTI expiry day or the day after one by the "
                     "exchange rule, the days the continuous series normally switches contract.")
    for d, (m, exp) in sorted(_near.items()):
        _bits.append(f"{d:%d %b %Y} is on or just after the last trading day of the {m:%b %y} contract "
                     f"({exp:%d %b %Y}, exchange rule), so part of its move may be the gap between two contracts.")
    roll_note_box = " " + " ".join(_bits)

# One plain statement about roll days for the table caption and the AI prompt.
if IS_BRENT and len(roll_excluded):
    tail_roll_fact = ("Excluded from the daily-loss check as Brent roll days: "
                      + ", ".join(f"{d:%d %b %Y} ({v:.2f}%)" for d, v in roll_excluded.items())
                      + f". With them the check would have fired on {len(roll_excluded)} more "
                      + ("day." if len(roll_excluded) == 1 else "days."))
elif IS_BRENT:
    tail_roll_fact = (f"{n_roll_recent} Brent roll days in the {len(recent_all)} trading days to "
                      f"{_ref_end:%d %b %Y} are excluded from the daily-loss check; none of them was beyond the "
                      "threshold.")
elif ttf_flagged:
    tail_roll_fact = (", ".join(f"{d:%d %b %Y}" for d in ttf_flagged)
                      + (" is" if len(ttf_flagged) == 1 else " are") + " on or just after a TTF "
                      "front-month expiry, so part of that day's move may be the gap between two contracts "
                      "(not verified: which contract Yahoo's series followed that day is not known).")
else:
    tail_roll_fact = ""

# Closing times: TTF and the carbon ETC close in European hours, Brent and WTI in the New York afternoon.
_EUROPEAN_CLOSE, _NEW_YORK_CLOSE = {'TTF=F', 'CARB.L'}, {'BZ=F', 'CL=F'}
_pair = {commodity['ticker'], ctx.compare_ticker}
corr_close_note = (" The two series close at different times (TTF and EU Carbon ETC in European hours, Brent and WTI in "
                   "the New York afternoon), so a move after the European close reaches the European series only "
                   "the next day, which pushes the same-day correlation down."
                   if _pair & _EUROPEAN_CLOSE and _pair & _NEW_YORK_CLOSE else "")

checks = []   # one row per check for the table
anomalies = []

# 1. Volatility spike / unusual calm: percentiles of the reference window.
n_hi = int((vol_last > vol_hi).sum())
n_lo = int((vol_last < vol_lo).sum())
# No 30-day volatility on the latest day (too few returns after a suspension): the two volatility checks
# give no latest value and do not fire; the volatility line above the table says why.
_vol_latest = common.fmt_vol(latest_vol) if latest_vol is not None else "n/a"
def _ann_pair(latest, thr):
    """Latest and threshold as the table shows them, with one decimal on the annualised figure when it would
    print the same for both (18% against 18%), so a reader can see why the check does or does not fire."""
    if latest is None:
        return "n/a", common.fmt_vol(thr)
    if f"{latest * common.ANNUAL:.0f}" != f"{thr * common.ANNUAL:.0f}":
        return common.fmt_vol(latest), common.fmt_vol(thr)
    return common.fmt_vol(latest, 1), common.fmt_vol(thr, 1)


_hi_now, _hi_thr = _ann_pair(latest_vol, vol_hi)
_lo_now, _lo_thr = _ann_pair(latest_vol, vol_lo)
checks.append(("Volatility spike", f"30-day vol above {_hi_thr} ({VOL_HI_PCT}th pct)",
               _hi_now, None if latest_vol is None else latest_vol > vol_hi, f"{n_hi} of {len(vol_last)}"))
checks.append(("Unusual calm", f"30-day vol below {_lo_thr} ({VOL_LO_PCT}th pct)",
               _lo_now, None if latest_vol is None else latest_vol < vol_lo, f"{n_lo} of {len(vol_last)}"))
if latest_vol is None:
    pass
elif latest_vol > vol_hi:
    anomalies.append({'level': '🟡 WARNING', 'type': 'Volatility Spike',
                      'detail': (f'30-day volatility of daily returns is {common.fmt_vol(latest_vol)}, above the '
                                 f'{VOL_HI_PCT}th percentile of {ref_name} ({common.fmt_vol(vol_hi)}); it sits at '
                                 f'the {vol_pctile:.0f}th percentile of that window. Fired on {n_hi} {_of}.')})
elif latest_vol < vol_lo:
    anomalies.append({'level': '🔵 WATCH', 'type': 'Unusual Calm',
                      'detail': (f'30-day volatility of daily returns is {common.fmt_vol(latest_vol)}, below the '
                                 f'{VOL_LO_PCT}th percentile of {ref_name} ({common.fmt_vol(vol_lo)}). Low volatility '
                                 'describes recent price moves only; it does not mean low risk. Fired on '
                                 f'{n_lo} {_of}.')})

# 2. GARCH 10-day forecast vs the 3-year average 30-day volatility.
if garch is not None and garch_ref:
    g_path = garch_10d_path(garch)
    g_path = g_path[g_path.index >= _count_from]
    g_ratio = garch_forecast_10d / garch_ref
    n_ghi = int((g_path > GARCH_HI * garch_ref).sum())
    n_glo = int((g_path < GARCH_LO * garch_ref).sum())
    _g_of = f"of the {len(g_path)} trading days to {g_path.index[-1]:%d %b %Y}"

    def _vol_pair(latest, thr):
        """Latest and threshold as the table shows them; one more decimal on both when they would print the same,
        so a reader can see why the check does or does not fire."""
        a, b = common.fmt_vol(latest), common.fmt_vol(thr)
        if a != b:
            return a, b
        return (f"{latest * common.ANNUAL:.1f}% ann. ({latest:.3f}% daily)",
                f"{thr * common.ANNUAL:.1f}% ann. ({thr:.3f}% daily)")
    _g_hi_now, _g_hi_thr = _vol_pair(garch_forecast_10d, GARCH_HI * garch_ref)
    _g_lo_now, _g_lo_thr = _vol_pair(garch_forecast_10d, GARCH_LO * garch_ref)
    checks.append(("GARCH forecast high", f"Day-10 daily vol forecast above {_g_hi_thr} "
                   f"({GARCH_HI:.1f}x {ref_short} mean)", _g_hi_now, g_ratio > GARCH_HI, f"{n_ghi} of {len(g_path)}"))
    checks.append(("GARCH forecast low", f"Day-10 daily vol forecast below {_g_lo_thr} "
                   f"({GARCH_LO:.1f}x {ref_short} mean)", _g_lo_now, g_ratio < GARCH_LO, f"{n_glo} of {len(g_path)}"))
    if g_ratio > GARCH_HI:
        anomalies.append({'level': '🟡 WARNING', 'type': f'GARCH Forecast More Than {(GARCH_HI - 1) * 100:.0f}% Above {ref_short.title()} Average',
                          'detail': (f'GARCH forecast of daily volatility on the 10th trading day ahead is '
                                     f'{common.fmt_vol(garch_forecast_10d)}, {(g_ratio - 1) * 100:.0f}% above the {garch_ref_name} '
                                     f'({common.fmt_vol(garch_ref)}). Fired on {n_ghi} {_g_of}.')})
    elif g_ratio < GARCH_LO:
        anomalies.append({'level': 'ℹ️ NOTE', 'type': f'GARCH Forecast More Than {(1 - GARCH_LO) * 100:.0f}% Below {ref_short.title()} Average',
                          'detail': (f'GARCH forecast of daily volatility on the 10th trading day ahead is '
                                     f'{common.fmt_vol(garch_forecast_10d)}, {(1 - g_ratio) * 100:.0f}% below the {garch_ref_name} '
                                     f'({common.fmt_vol(garch_ref)}). Fired on {n_glo} {_g_of}. Low forecast '
                                     'volatility is not low risk.')})

# 3. Correlation outside its own 3-year range. With no latest figure the row shows n/a and does not fire.
if has_corr_ref and not has_corr:
    n_corr = int(((corr_last > corr_hi) | (corr_last < corr_lo)).sum())
    checks.append((f"Correlation with {compare_label}", f"30-day corr outside {corr_lo:.2f} to {corr_hi:.2f} "
                   f"({VOL_LO_PCT}th to {VOL_HI_PCT}th pct)", "n/a", None, f"{n_corr} of {len(corr_last)}"))
if has_corr:
    corr_out = corr_now > corr_hi or corr_now < corr_lo
    n_corr = int(((corr_last > corr_hi) | (corr_last < corr_lo)).sum())
    checks.append((f"Correlation with {compare_label}", f"30-day corr outside {corr_lo:.2f} to {corr_hi:.2f} "
                   f"({VOL_LO_PCT}th to {VOL_HI_PCT}th pct)", f"{corr_now:.2f}", corr_out, f"{n_corr} of {len(corr_last)}"))
    if corr_out:
        anomalies.append({'level': '🔵 WATCH', 'type': f'Correlation Outside {corr_short.title()} Range',
                          'detail': (f'30-day correlation of daily returns with {compare_label} is {corr_now:.2f}, '
                                     f'outside its {VOL_LO_PCT}th to {VOL_HI_PCT}th percentile range of {corr_name}: '
                                     f'{corr_lo:.2f} to {corr_hi:.2f}, median {corr_med:.2f}. Fired on {n_corr} '
                                     f'of the {len(corr_last)} trading days to {corr_last.index[-1]:%d %b %Y}.'
                                     + corr_close_note)})

# 4. Large daily losses in the last 252 trading days of the reference window.
_tail_rule = (f"loss beyond {tail_thr:.2f}% ({TAIL_MULT:.0f}x {ref_short} VaR99) on the latest day; breaches counted "
              f"over the {len(recent_all)} trading days to {_ref_end:%d %b %Y}"
              + (", Brent roll days excluded" if IS_BRENT else ""))
# "Fires now" means the latest trading day, as for the other checks; the last breach in the window is
# shown with its date, so a loss months ago is not read as today's signal.
_last_ret = float(recent_all.iloc[-1])
_last_is_roll = bool(_is_roll[-1]) if IS_BRENT else False
_tail_now = (_last_ret < -tail_thr) and not _last_is_roll
# The roll check's return of the contract the series switched to, when the latest day is an identified roll.
_last_new_ret = next((nr for t, _, _, nr in (core['rolls'] or {}).get('rolls', [])
                      if pd.Timestamp(t) == recent_all.index[-1]), None) if IS_BRENT else None
# The roll was read from a row that may not be a close yet (common.futures_row_state): say it can still change.
_roll_live = (IS_BRENT and _last_is_roll
              and common.futures_row_state(recent_all.index[-1]) != 'close')
_roll_live_txt = ", read from the latest price, which can still change" if _roll_live else ""
# How much of the latest move is the gap between the contracts: the series' return minus the new contract's own.
_roll_gap = (_last_ret / 100 - _last_new_ret) if (_last_is_roll and _last_new_ret is not None) else None
_tail_latest = (f"latest {_last_ret:+.2f}%"
                + (f" (a roll day, not checked; the contract it switched to {_last_new_ret * 100:+.2f}%{_roll_live_txt})"
                   if _last_is_roll and _last_new_ret is not None else
                   f" (a roll day, not checked{_roll_live_txt})" if _last_is_roll else "")
                + (f"; last breach {tail_days.index[-1]:%d %b %Y} ({tail_days.iloc[-1]:.2f}%)" if len(tail_days)
                   else f"; worst in window {recent_rets.min():.2f}%")
                + (" (excl. roll days)" if IS_BRENT else ""))
checks.append(("Large daily loss", _tail_rule, _tail_latest, _tail_now, f"{len(tail_days)} of {len(recent_rets)}"))
if _last_is_roll:
    tail_roll_fact = (tail_roll_fact + " " if tail_roll_fact else "") + (
        f"The latest day, {recent_all.index[-1]:%d %b %Y}, is a Brent roll day and is not checked"
        + (" (read from the latest price, which can still change)" if _roll_live else "") + ".")
if len(tail_days):
    _tail_detail = (f'{len(tail_days)} daily {"loss" if len(tail_days) == 1 else "losses"} in the {len(recent_all)} trading days to {_ref_end:%d %b %Y} '
                    + ("(Brent roll days excluded) " if IS_BRENT else "")
                    + f'beyond {tail_thr:.2f}% ({TAIL_MULT:.0f}x the {ref_short} 99% VaR of {var99_3y:.2f}%); '
                    f'the largest was {tail_days.min():.2f}% on {tail_days.idxmin():%d %b %Y}.')
    if tail_roll_fact:
        _tail_detail += " " + tail_roll_fact
    elif roll_note_box and not IS_BRENT:
        _tail_detail += roll_note_box
    anomalies.append({'level': '🔵 WATCH' if _tail_now else 'ℹ️ NOTE',
                      'type': (f'Daily Loss Beyond {TAIL_MULT:.0f}x {ref_short.title()} VaR99'
                               + ('' if _tail_now else f' (last on {tail_days.index[-1]:%d %b %Y}, not on the latest day)')),
                      'detail': _tail_detail,
                      'past': (None if _tail_now else
                               f'daily loss beyond {tail_thr:.2f}% ({TAIL_MULT:.0f}x {ref_short} VaR99): {len(tail_days)} in the '
                               f'{len(recent_all)} trading days to {_ref_end:%d %b %Y}, the latest {tail_days.iloc[-1]:.2f}% on '
                               f'{tail_days.index[-1]:%d %b %Y}')})

check_tbl = pd.DataFrame(checks, columns=['Check', 'Fires when', 'Latest', 'Fires now',
                                          f'Days fired (last {CHECK_DAYS} trading days'
                                          + (', fewer where roll days are excluded' if IS_BRENT and n_roll_recent else '')
                                          + ')'])
check_tbl['Fires now'] = check_tbl['Fires now'].map(lambda v: 'n/a' if v is None else 'Yes' if v else 'No')
st.dataframe(check_tbl, hide_index=True, width="stretch")
if IS_BRENT and tail_roll_fact:
    st.caption(tail_roll_fact)

if anomalies:
    for a in anomalies:
        level = a['level']
        if 'WARNING' in level:
            bg, border = '#fffbe6', '#ffaa00'
        elif 'WATCH' in level:
            bg, border = '#f0f4ff', '#4488ff'
        else:
            bg, border = '#f4f4f4', '#999999'
        st.markdown(
            f"<div style='background:{bg}; border-left:4px solid {border}; color:#222; "
            f"padding:10px 14px; margin:6px 0; border-radius:4px;'>"
            f"<strong>{a['level']}: {a['type']}</strong><br>"
            f"<span style='font-size:0.9em'>{a['detail']}</span></div>",
            unsafe_allow_html=True
        )
else:
    st.info(f"None of the {len(checks)} checks fires on the latest data. They cover only volatility, the GARCH "
            "forecast, one correlation and daily losses; they say nothing about supply, storage, positions or news.")

# What the volatility line above the table compares the latest figure with: the pre-war level only when it gives a
# pre-war multiple (none for an end date before the reference quarter is complete), nothing when it has no figure.
if _vs['ctx'] is None:
    _vs_against = ""
elif _vs['ctx']['prewar_ratio'] is None:
    _vs_against = "; the volatility line above the table places it against the whole history instead"
else:
    _vs_against = (f"; the volatility line above the table places it against {_vs.get('level_name', 'the pre-war level')} "
                   "and the whole history instead")
_check_notes = (
    f"Thresholds and the counts of days fired come from the reference window and use no "
    f"data after {_ref_end:%d %b %Y}, so the counts are in-sample: about 5% of the window lies beyond each "
    f"percentile by construction, and the count shows how much of that fell in {_count_span}.{corr_source} Latest values are "
    f"at the last trading day with a price in the selected range ({_asof:%d %b %Y})."
    + (" " + corr_gap_note if corr_gap_note and has_corr_ref else "")
    + common.last_value_note(commodity['ticker'], end_date, _asof) + " " + garch_fit_note
    + " The Risk page fits its GARCH on the whole selected range, so its forecast can differ. The GARCH counts "
    "use the parameters fitted on the whole window for every past day. Volatility is the standard deviation "
    "of daily returns, shown annualised (x √252) with the daily figure in brackets. These checks ask whether the "
    f"latest day is unusual for the {ref3y_span} before it" + _vs_against + ". " + _vs['detail']
    + (" " + _vs['roll'] if _vs['roll'] else "") + roll_note + corr_close_note)
# On the page: the window, the date of the latest values and what kind of price the last one is; the method,
# definitions and roll notes are folded (the same text, nothing removed).
_lv = common.last_value_note(commodity['ticker'], end_date, _asof)
_kind = ("the last value is a live price, not a settlement" if "latest price" in _lv else
         "the last value may not yet be the settlement" if "may not yet be the settlement" in _lv else
         "the last value may be an intraday London price" if "London session" in _lv else "")
st.caption(f"Reference window: {ref_source}. Latest values: {_asof:%d %b %Y}" + (f" ({_kind})" if _kind else "") + "."
           + " The GARCH check is fitted on this window; the Risk page fits its GARCH on the whole selected range, so "
           "its day-10 forecast differs.")
if corr_gap_note and has_corr_ref:
    st.caption(corr_gap_note)
with st.expander("How the checks are built"):
    st.caption(_check_notes)


# ─── Section 4: AI Risk Narrative (LLM) ───
st.subheader("AI Summary of the Statistics")
st.write(f"A three-sentence summary of the statistics on this page for {selected_commodity}.")


def _side(cur, thr):
    return "ABOVE" if cur > thr else "BELOW" if cur < thr else "EQUAL TO"


def level_facts(vol_now, vol_spike, vol_calm, ref_mean, g_now, g_10d, garch_hi, garch_lo,
                corr_v=None, corr_lo_v=None, corr_hi_v=None, corr_label=""):
    """Every threshold on the page together with the current value it is compared with and the
    side that value is on, as (prompt lines, levels). Each level is (subject, threshold, current),
    subject 'vol' (30-day volatility), 'garch' (GARCH day-10 forecast) or 'corr' (30-day
    correlation); the levels feed drop_crossed_watch_sentences."""
    lines, levels = [], []

    def add(subject, label, cur, cur_txt, thr, thr_txt, extra=""):
        lines.append(f"{label} {cur_txt} is {_side(cur, thr)} {thr_txt}{extra}.")
        levels.append((subject, float(thr), float(cur)))

    # Without a latest 30-day volatility (vol_now None) there is no current value to place against the levels.
    v = f"{common.fmt_vol(vol_now)}" if vol_now is not None else ""
    if vol_now is not None:
        add('vol', "Current 30-day volatility", vol_now, v, vol_spike,
            f"the volatility-spike threshold {common.fmt_vol(vol_spike)} ({VOL_HI_PCT}th percentile of {ref_name})")
        add('vol', "Current 30-day volatility", vol_now, v, vol_calm,
            f"the unusual-calm threshold {common.fmt_vol(vol_calm)} ({VOL_LO_PCT}th percentile of {ref_name})")
        add('vol', "Current 30-day volatility", vol_now, v, ref_mean,
            f"the {ref_short} average 30-day volatility {common.fmt_vol(ref_mean)}")
    if g_10d is not None:
        g = f"{common.fmt_vol(g_10d)}"
        add('garch', "GARCH day-10 forecast", g_10d, g, garch_hi * ref_mean,
            f"the GARCH-high threshold {common.fmt_vol(garch_hi * ref_mean)} ({garch_hi:.1f}x the {ref_short} average)")
        add('garch', "GARCH day-10 forecast", g_10d, g, garch_lo * ref_mean,
            f"the GARCH-low threshold {common.fmt_vol(garch_lo * ref_mean)} ({garch_lo:.1f}x the {ref_short} average)")
        add('garch', "GARCH day-10 forecast", g_10d, g, ref_mean,
            f"the {ref_short} average 30-day volatility {common.fmt_vol(ref_mean)}",
            f", by {abs(g_10d / ref_mean - 1) * 100:.0f}%")
        if g_now is not None:
            add('garch', "GARCH day-10 forecast", g_10d, g, g_now,
                f"the current GARCH daily volatility {common.fmt_vol(g_now)}")
        # The GARCH forecast is a different measure from realised 30-day volatility, but a reader
        # (or the model) may compare them, so the side is stated here too.
        if vol_now is not None:
            add('vol', "Current 30-day volatility", vol_now, v, g_10d, f"the GARCH day-10 forecast {g}")
    if corr_v is not None:
        c = f"{corr_v:.2f}"
        add('corr', f"Current 30-day correlation with {corr_label}", corr_v, c, corr_hi_v,
            f"the top of its {VOL_LO_PCT}th to {VOL_HI_PCT}th percentile range of {corr_name}, {corr_hi_v:.2f}")
        add('corr', f"Current 30-day correlation with {corr_label}", corr_v, c, corr_lo_v,
            f"the bottom of that range, {corr_lo_v:.2f}")
    return lines, levels


# ─── Post-check of the AI text ───
# The model is told on which side of every level the current value is, but its output is still
# checked before display: a sentence that tells the reader to watch whether a value pushes
# above (or falls below) a level it is already above (below) is dropped. Deterministic rules:
# 1. "ann." is written out as "annualised" first, so the abbreviation the page itself uses
#    ("51% ann. (3.19% daily)") does not end a sentence; then the text is split into sentences;
# 2. only sentences with a forward-looking cue (watch, monitor, whether, if, should, would, could, next,
#    coming, following, ahead) are tested ("Should volatility rise above ..." included);
# 3. for every crossing verb (pushes, breaks, crosses, rises, climbs, moves, returns, jumps, exceeds, tops,
#    surpasses, falls, drops, slips, dips, sinks, eases, retreats, declines) the numbers in the next 80 characters are
#    read: the first one, and the next one if it follows in brackets (the other scale of the same
#    level), with a minus sign directly before it (hyphen, U+2212 or U+2013), also at the end of a sentence
#    ("above -0.21.").
#    A percentage (with or without decimals) is a volatility, annualised if
#    "annualised" (or "annualized", "annual") follows it and daily if "daily" follows it, either if
#    neither does; a decimal number without "%" is a correlation. The move is downward if "below",
#    "under", "down", "lower" or a falling verb appears between the verb and the first number,
#    otherwise upward;
# 4. the subject is the one of GARCH/forecast, correlation or volatility named nearest before the verb, after
#    the cue ("whether the correlation stays near 0.39 while volatility rises above ..." is about volatility;
#    "the GARCH forecast of volatility" is the forecast); if none, every subject is tried;
# 5. a number matches a level when it equals the level rounded to the same decimals, an annualised
#    number against the level x sqrt(252); the sentence is dropped if the current value is already
#    above that level for an upward move, or already below it for a downward move.
_FORWARD_CUE = re.compile(r"\b(?:watch\w*|monitor\w*|whether|if|should|would|could|next|coming|following|ahead)\b",
                          re.I)
# Upward verbs: push, break ("breaks above"), cross, climb, rise, move, return, jump, spike, exceed, top ("tops";
# not the bare noun "top"), surpass. Downward verbs (also in _DOWNWARD): fall ("falls through"), drop ("drops
# below"), slip, dip, sink, ease, retreat, decline.
_CROSS_VERB = re.compile(
    r"\b(?:push(?:es|ing)?|break(?:s|ing)?|cross(?:es|ing)?|climb(?:s|ing)?|ris(?:e|es|ing)|mov(?:e|es|ing)"
    r"|return(?:s|ing)?|jump(?:s|ing)?|spik(?:e|es|ing)|exceed(?:s|ing)?|top(?:s|ping)|surpass(?:es|ing)?"
    r"|fall(?:s|ing)?|drop(?:s|ping)?|slip(?:s|ping)?"
    r"|dip(?:s|ping)?|sink(?:s|ing)?|eas(?:e|es|ing)|retreat(?:s|ing)?|declin(?:e|es|ing))\b", re.I)
# Verbs that name no direction ("returns to 28%", "moves to 0.40"): the direction is taken from an explicit
# word before the number (_DOWNWARD, _UPWARD), else from the level against the current value, so a sentence
# about moving back to a level on the other side is not read as a crossing.
_NEUTRAL_VERB = re.compile(r"^(?:mov(?:e|es|ing)|return(?:s|ing)?)$", re.I)
_UPWARD = re.compile(r"\b(?:above|over|beyond|higher|up)\b", re.I)
_DOWNWARD = re.compile(
    r"\b(?:below|under|beneath|down|lower|fall(?:s|ing)?|drop(?:s|ping)?|slip(?:s|ping)?|dip(?:s|ping)?"
    r"|sink(?:s|ing)?|eas(?:e|es|ing)|retreat(?:s|ing)?|declin(?:e|es|ing))\b", re.I)
# A number: a percentage (integer or decimal) or a decimal; bare integers ("ten days", "10 days") are not levels.
# A minus sign (hyphen, U+2212 or U+2013, which a model may write for minus) directly before it is read, unless a
# letter or digit precedes the sign ("2021-22").
# The number may end a sentence: a full stop after it counts as a decimal point only when a digit follows.
_NUMBER = re.compile(r"(?:(?<![\w.])([-\u2212\u2013])|(?<![\d.]))(\d+)(?:\.(\d+))?(?!\d)(?!\.\d)(\s*%)?")
_ANN_ABBR = re.compile(r"\bann\.", re.I)
_ANNUAL_AFTER = re.compile(r"\s*(?:annuali[sz]ed|annual)\b", re.I)
_DAILY_AFTER = re.compile(r"\s*daily\b", re.I)
# "GARCH volatility" and "forecast of volatility" are the forecast: the GARCH pattern takes the word volatility
# that follows it, so that word is not read as a second subject.
_SUBJECTS = [('garch', re.compile(r"\b(?:GARCH|forecast)(?:\s+(?:of\s+)?(?:volatility|vol)\b)?\b", re.I)),
             ('corr', re.compile(r"\bcorrelation\b", re.I)),
             ('vol', re.compile(r"\b(?:volatility|vol)\b", re.I))]


def _subject(text):
    """The subject named last in `text` (the text before the verb, so the one nearest the verb), or None."""
    found = [(m.start(), m.end(), name) for name, p in _SUBJECTS for m in p.finditer(text)]
    garch = [(a, b) for a, b, name in found if name == 'garch']
    found = [(a, name) for a, b, name in found
             if name == 'garch' or not any(ga <= a and b <= gb for ga, gb in garch)]
    return max(found)[1] if found else None


def _numbers(window, start):
    """(value, decimals, kind) for each level-like number in `window` from `start`: kind 'ann', 'daily' or
    'pct' (a percentage of either scale) for a percentage, 'corr' for a decimal without '%'."""
    out = []
    for m in _NUMBER.finditer(window, start):
        sign, whole, frac, pct = m.group(1), m.group(2), m.group(3), m.group(4)
        if not pct and frac is None:
            continue
        num, dec = float(whole + ("." + frac if frac else "")), len(frac or "")
        if sign:
            num = -num
        if not pct:
            out.append((m.start(), num, dec, 'corr'))
            continue
        rest = window[m.end():]
        kind = 'ann' if _ANNUAL_AFTER.match(rest) else 'daily' if _DAILY_AFTER.match(rest) else 'pct'
        out.append((m.start(), num, dec, kind))
    return out


def _matches(num, dec, kind, sub, thr):
    """True if the number written in the text is this level, to the decimals written."""
    if kind == 'corr':
        return sub == 'corr' and round(thr, dec) == round(num, dec)
    if sub == 'corr':
        return False
    daily_ok = round(thr, dec) == round(num, dec)
    ann_ok = round(thr * common.ANNUAL, dec) == round(num, dec)
    return {'ann': ann_ok, 'daily': daily_ok}.get(kind, daily_ok or ann_ok)


def drop_crossed_watch_sentences(text, levels):
    """(text without the offending sentences, list of dropped sentences). See the rules above. `levels` are
    (subject, threshold, current) with subject 'vol', 'garch' or 'corr' and volatility levels as daily %."""
    text = _ANN_ABBR.sub("annualised", text.strip())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    kept, dropped = [], []
    for s in sentences:
        bad = False
        if _FORWARD_CUE.search(s):
            for vm in _CROSS_VERB.finditer(s):
                window = s[vm.start():vm.start() + 80]
                nums = _numbers(window, len(vm.group(0)))
                if not nums:
                    continue
                downward = bool(_DOWNWARD.search(window[:nums[0][0]]))
                neutral = (bool(_NEUTRAL_VERB.match(vm.group(0))) and not downward
                           and not _UPWARD.search(window[max(0, nums[0][0] - 15):nums[0][0]]))
                # The level named is the first number, with its other scale if it follows in brackets
                # ("28% annualised (1.74% daily)"); later numbers belong to other parts of the sentence.
                nums = nums[:1] + [x for x in nums[1:2]
                                   if x[0] - nums[0][0] <= 30 and "(" in window[nums[0][0]:x[0]]]
                before = s[:vm.start()]
                cues = list(_FORWARD_CUE.finditer(before))
                subj = _subject(before[cues[-1].end():] if cues else before) or _subject(before)
                for _, num, dec, kind in nums:
                    for sub, thr, cur in levels:
                        if subj is not None and sub != subj:
                            continue
                        if not _matches(num, dec, kind, sub, thr):
                            continue
                        if neutral:
                            continue          # direction from level vs current: never a crossing already made
                        if (not downward and cur > thr) or (downward and cur < thr):
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    break
        (dropped if bad else kept).append(s)
    return " ".join(kept), dropped


# No parameter starts with an underscore: st.cache_data skips such parameters when building the
# cache key, and then a changed number could return an old narrative.
@st.cache_data(ttl=1800, show_spinner="Generating AI risk interpretation...")
def generate_risk_narrative(commodity_name, risk_level_str, latest_vol_v, avg_vol_v,
                            var_95_v, garch_now, garch_10d, garch_ref_v, garch_ref_label,
                            anomaly_types_str, date_str, levels_str, tail_fact_str,
                            past_breaches_str="", garch_fit_str="", prewar_ref="the pre-war level (Oct to Dec 2025)",
                            check_window_str="the last 3 years", ref_short_str="3-year", last_value_str=""):
    """The prompt has no headlines: the page lists them unscored, and unverified text would only invite the model
    to present a headline as fact or read a direction into it. `prewar_ref` names the
    fixed reference quarter as common.vol_summary does for the AI input ('' when there is no multiple, "the Oct
    to Dec 2025 level" before the strikes of 28 Feb 2026), so the prompt names no event after the data date.
    `latest_vol_v` is None when there is no 30-day volatility on the data date (the summary says why)."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return None, "no_key"

    if garch_10d and garch_now:
        _dir = ("higher than" if garch_10d > garch_now * 1.02 else
                "lower than" if garch_10d < garch_now * 0.98 else "roughly the same as")
        garch_line = (f"GARCH daily volatility now {common.fmt_vol(garch_now)}; forecast for day 10 ahead {common.fmt_vol(garch_10d)}, "
                      f"{_dir} now")
    elif garch_10d:
        garch_line = f"GARCH forecast of daily volatility on day 10 ahead: {common.fmt_vol(garch_10d)}"
    else:
        garch_line = "GARCH forecast: unavailable"
    if garch_10d and garch_ref_v:
        _pos = "above" if garch_10d > garch_ref_v else "below" if garch_10d < garch_ref_v else "equal to"
        garch_line += (f"; the day-10 forecast is {abs(garch_10d / garch_ref_v - 1) * 100:.0f}% {_pos} the "
                       f"{garch_ref_label} ({common.fmt_vol(garch_ref_v)}). {garch_fit_str}").rstrip()
    anomaly_line = anomaly_types_str if anomaly_types_str else "None"
    past_line = past_breaches_str if past_breaches_str else "None"

    if latest_vol_v is None:
        vol_refs = ("- There is no current 30-day volatility for this date; the volatility summary says why. Do not "
                    "describe or estimate the level of volatility, and do not mention a percentile or a multiple of it.")
    elif prewar_ref:
        _rule = ("State the pre-war multiple whenever you describe the level of volatility."
                 if prewar_ref.startswith("the pre-war level") else
                 f"State the multiple of {prewar_ref} whenever you describe the level of volatility.")
        vol_refs = ("- The volatility summary places current 30-day volatility against two references that do not "
                    f"roll with the market: its multiple of {prewar_ref} and its percentile among every 30-day "
                    f"volatility since the start of the data (named in the summary). {_rule}")
    else:
        vol_refs = ("- The volatility summary places current 30-day volatility against a reference that does not "
                    "roll with the market: its percentile among every 30-day volatility since the start of the data "
                    "(named in the summary). There is no multiple of a fixed earlier level for this date: do not "
                    "mention one, and describe the level of volatility only by its percentile.")
    vol_line = (f"{common.fmt_vol(latest_vol_v)} (average over the selected period {common.fmt_vol(avg_vol_v)})"
                if latest_vol_v is not None else
                f"not available on the data date (see the volatility summary); average over the selected period "
                f"{common.fmt_vol(avg_vol_v)}")
    tail_line = tail_fact_str if tail_fact_str else "None"
    prompt = f"""You are a senior risk analyst on a European energy trading desk, writing the short risk comment on {commodity_name} for the date shown below.

HOW TO READ THE INPUTS

Volatility
- Every volatility figure is the standard deviation of daily returns, written annualised first (daily x the square root of 252) with the daily figure in brackets, for example "51% ann. (3.19% daily)". The two numbers are the same volatility on two scales: never compare an annualised figure with a daily one.
- The GARCH figure forecasts the volatility of a single day's return on the tenth trading day ahead (written the same way). It is not a cumulative move over ten days.
- The GARCH check compares the day-10 forecast with the commodity's average 30-day volatility over {check_window_str}, not with current volatility. {garch_fit_str} Describe the forecast only by the two facts given: its direction relative to current GARCH volatility (higher, lower or about the same) and whether it is above or below the {ref_short_str} average. Use neutral wording: do not call it normalisation, mean reversion, a return to normal, escalation, a build-up of risk, calming, reassuring or a danger signal.

Volatility summary
{vol_refs}
- There is no Normal / Elevated / Extreme label. Do not invent one, and do not call volatility high or low without one of the two references.
- Volatility measures how much prices move, not overall risk. Never call the market "low risk" because volatility is low: supply disruptions, storage levels and policy events can make a calm-volatility market risky.
- The selected-period average volatility is context only.
- Write volatility as the inputs do: annualised first, the daily figure in brackets.

VaR
- VaR 95% is the 5th percentile of daily returns over the last 250 trading days: a loss threshold, given as a negative number.

Checks
- The latest-day checks field lists only the NAMES of the statistical checks firing on the latest day (percentiles of {check_window_str}). The past-breaches field lists events that are over; mention one only with its date, as past. Their thresholds are in the LEVELS list; do not invent other numbers.

MARKET DATA AS OF {date_str}
- Last value: {last_value_str or 'a close (settlement or exchange close).'}
- Volatility summary: {risk_level_str}
- 30-day rolling volatility: {vol_line}
- VaR 95%, 1-day (historical, last 250 trading days): {var_95_v:.2f}%
- {garch_line}
- Checks firing on the latest day: {anomaly_line}
- Past breaches in the last 252 trading days up to the data date (with dates): {past_line}
- Daily-loss check and contract rolls: {tail_line}

LEVELS AND WHERE EACH CURRENT VALUE IS (already computed; do not recompute or contradict)
{levels_str}

TASK
Write EXACTLY 3 sentences. Keep the whole comment under 100 words. No headers, no bullets, no line breaks. Write as of the data date above, not as of any later date.
1. What this combination of readings means. Do not restate the figures - the reader is looking at them. Use at most one number, and only if it carries the argument.
2. The sharpest tension or divergence in the data, and what it changes in how the other figures should be read.
3. One concrete, checkable thing to watch over the following ten trading days. Name one level from the LEVELS list and the value it applies to, and describe a move consistent with the side that value is on NOW: for a value already ABOVE a level, watch whether it stays above or falls back below; for a value already BELOW a level, watch whether it rises above or stays below. Do not simply say to watch whether the forecast proves correct.

HARD RULES
- Never write "watch whether X pushes above", "pushes through", "breaks", "crosses" or "rises above" a level that X is already ABOVE, and never write "falls below" or "drops below" a level that X is already BELOW. Check the LEVELS list before naming any level.
- Compare a value only with a level listed for that value.
- GARCH is a model fitted on past daily returns. Never call it implied, options-implied or option-based, and never say it is or is not a "model artifact".
- A check that last fired on an earlier date is a past event: name its date if you mention it, and never present it as a current signal.
- Give no trading, hedging or position-sizing advice.
- If a field reads N/A or None, pass over it in silence. Do not comment on its absence.
- Never attribute a claim, forecast or warning to a named bank, agency, institution or analyst.
- Do not assert a causal chain the data above cannot support. If a link is speculative, mark it as speculative.
- Your only raw input is a series of daily prices: closes, except that the last value may not be a close when the "Last value" line says so; then never call the latest day's move a close or a settlement. You have no term structure, no futures curve, no inventories, no storage, no positioning, no flows and no fundamentals. Never write about contango, backwardation, spreads, curve shape, stockpiles, supply, demand or hedging costs, and do not discuss roll risk in general. One exception: if the "Daily-loss check and contract rolls" field states that a day was a contract roll day and was excluded, or may be partly a roll, you may state that fact as given; never use an excluded roll day as evidence of tail risk.
- Do not introduce seasonal, geopolitical or fundamental drivers that are not present in the inputs above.
- Avoid: "low-risk equilibrium", "market participants should", "underlying stress", "tranquility", "it is important to note".
- Plain desk English. No padding.

Output only the 3 sentences."""

    try:
        response = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": AI_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        if response.status_code == 200:
            narrative = response.json()['content'][0]['text'].strip()
            return narrative, None
        else:
            return None, f"api_error_{response.status_code}"
    except Exception as e:
        return None, str(e)


anomaly_types_for_llm = "; ".join(a['type'] for a in anomalies if not a.get('past'))
past_breaches_for_llm = "; ".join(a['past'] for a in anomalies if a.get('past'))
# The same vol_summary as the volatility line in the checks section (past_end, roll check included), in the
# wording it gives for the AI input: no event after the data date (common.vol_summary).
vol_summary_for_llm = _vs['ai']

today_date_str = df_analysis.index[-1].strftime('%Y-%m-%d')

level_lines, level_list = level_facts(
    float(latest_vol) if latest_vol is not None else None, float(vol_hi), float(vol_lo), float(garch_ref),
    garch['current_cond_vol'] if garch is not None else None, garch_forecast_10d, GARCH_HI, GARCH_LO,
    corr_v=corr_now if has_corr else None, corr_lo_v=float(corr_lo) if has_corr else None,
    corr_hi_v=float(corr_hi) if has_corr else None, corr_label=compare_label)
_gap_words = ("includes the price gap between two contracts" if _roll_gap is None else
              "is mostly the price gap between two contracts, not a Brent price move"
              if abs(_roll_gap) > abs(_last_new_ret) else
              "is partly the price gap between two contracts")
_worst_txt = (((f"Latest daily return ({recent_all.index[-1]:%d %b %Y}): {_last_ret:+.2f}% on the continuous "
               "series, but this is a Brent roll day"
               + (" (read from the latest price, which can still change)" if _roll_live else "")
               + ": the series switched to the next contract, "
               f"so the figure {_gap_words}"
               + (f" (the contract it switched to moved {_last_new_ret * 100:+.2f}%)"
                  if _last_new_ret is not None else "")
               + f"; it is not checked against the daily-loss threshold of -{tail_thr:.2f}%. ")
              if _last_is_roll else
              (f"Latest daily return ({recent_all.index[-1]:%d %b %Y}): {_last_ret:+.2f}%, "
               + ("beyond" if _tail_now else "within") + f" the daily-loss threshold of -{tail_thr:.2f}%. "))
              + (f"The last loss beyond it was {tail_days.iloc[-1]:.2f}% on {tail_days.index[-1]:%d %b %Y}, "
                 f"{(recent_all.index[-1] - tail_days.index[-1]).days} days before the data date: a past event, "
                 "not a current signal." if len(tail_days) and not _tail_now else "")
              + (" Brent roll days excluded." if IS_BRENT else ""))
_worst_txt = " ".join(_worst_txt.split())
levels_for_llm = "\n".join(f"- {s}" for s in level_lines + [_worst_txt])

narrative, error = generate_risk_narrative(
    commodity_name=selected_commodity,
    risk_level_str=vol_summary_for_llm,
    latest_vol_v=float(latest_vol) if latest_vol is not None else None,
    avg_vol_v=float(avg_vol),
    var_95_v=float(var_95),
    garch_now=garch['current_cond_vol'] if garch is not None else None,
    garch_10d=garch_forecast_10d,
    garch_ref_v=float(garch_ref) if garch is not None else None,
    garch_ref_label=garch_ref_name,
    anomaly_types_str=anomaly_types_for_llm,
    past_breaches_str=past_breaches_for_llm,
    date_str=today_date_str,
    levels_str=levels_for_llm,
    tail_fact_str=tail_roll_fact,
    garch_fit_str=garch_fit_note,
    prewar_ref=_vs['prewar_ai'],
    check_window_str=ref_name,
    ref_short_str=ref_short,
    last_value_str=common.last_value_note(commodity['ticker'], end_date, _asof).strip(),
)

narrative_dropped = []
if narrative:
    narrative, narrative_dropped = drop_crossed_watch_sentences(narrative, level_list)

_ai_scope = ("It only summarises the statistics on this page (volatility, VaR, GARCH "
             "and the checks above; not the headlines); it does not see fundamentals such as storage, flows, positioning or "
             "the forward curve.")
_dropped_note = (f" An automatic check removed {len(narrative_dropped)} sentence(s) that asked the reader to "
                 "watch for a level the data had already passed." if narrative_dropped else "")
if narrative:
    _safe = html.escape(narrative).replace("$", "\\$")
    st.markdown(
        f"<div style='background:#f8f9ff; border-left:4px solid #6655ee; color:#222; "
        f"padding:14px 18px; border-radius:6px; font-size:0.97em; line-height:1.7'>"
        f"{_safe}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Generated by Claude ({AI_MODEL}) · data as of {df_analysis.index[-1]:%d %b %Y} · {_ai_scope} "
               "Cached for 30 minutes for identical inputs." + _dropped_note)
elif narrative_dropped:
    st.caption("AI summary withheld: every sentence asked the reader to watch for a level the data had "
               "already passed.")
elif error == "no_key":
    st.caption("AI summary not enabled in this deployment (no API key). " + _ai_scope)
else:
    st.caption("AI summary temporarily unavailable; the statistics above are unaffected.")


# ─── Section 5: Data Export ───
st.subheader("Export Data")

export_df = df_analysis[['Price', 'Compare', 'Volatility', 'Rolling Correlation']].copy()
export_df.columns = [selected_commodity, compare_label, '30-day volatility of daily returns (daily, %)',
                     '30-day rolling correlation (returns)']
export_df.insert(3, '30-day volatility, annualised (daily x sqrt(252), %)', df_analysis['Volatility'] * common.ANNUAL)
export_df['Daily_Return_%'] = df_analysis['Returns'] * 100
if IS_BRENT:
    export_df['Brent roll day (excluded from the daily-loss check)'] = export_df.index.isin(brent_rolls)
st.download_button(
    label=f"Download {selected_commodity} Risk Data (CSV)",
    data=export_df.to_csv(),
    file_name=f"{selected_commodity.lower().replace(' ', '_')}_risk_data.csv",
    mime="text/csv"
)

if news_rows:
    st.download_button(
        label="Download Recent Headlines (CSV)",
        data=pd.DataFrame([{'Headline': a['title'], 'Site': a['domain'], 'Source country (GDELT)': a['country'],
                            'Seen by GDELT (UTC)': f"{a['seen']:%Y-%m-%d %H:%M}", 'Link': a['url']}
                           for a in news_rows]).to_csv(index=False),
        file_name="recent_headlines.csv",
        mime="text/csv"
    )

if not cr_df.empty:
    # Ranked table plus the raw Eurostat inputs behind it, so every number can be recomputed.
    raw_cols = {'gas': ['gae_gas', 'gae_total', 'dep_gas', 'dep_total', 'ren'],
                'oil': ['gae_oil', 'gae_total', 'dep_oil', 'dep_total', 'ren'],
                'carbon': ['ghg', 'gdp_clv', 'ren']}[country_mode]
    raw_names = {'gae_gas': 'Gross available gas (ktoe, nrg_bal_c G3000)',
                 'gae_oil': 'Gross available oil (ktoe, nrg_bal_c O4000XBIO)',
                 'gae_total': 'Gross available energy, all products (ktoe, nrg_bal_c TOTAL)',
                 'dep_gas': 'Gas import dependency (%, nrg_ind_id G3000)',
                 'dep_oil': 'Oil import dependency (%, nrg_ind_id O4000XBIO)',
                 'dep_total': 'Total energy import dependency (%, nrg_ind_id TOTAL)',
                 'ren': 'Renewable share (%, nrg_ind_ren REN)',
                 'ghg': 'GHG excl. LULUCF (kt CO2e, env_air_gge TOTX4_MEMO)',
                 'gdp_clv': 'GDP (M€ 2020 prices, nama_10_gdp CLV20_MEUR)'}
    cr_export = ranked[['Country', mcol] + raw_cols].rename(columns={mcol: M['measure'], **raw_names})
    eu_export = pd.DataFrame([eu])[['Country', mcol] + raw_cols].rename(columns={mcol: M['measure'], **raw_names})
    # Which of the row's values Eurostat flags as provisional ("p"), by the column names above.
    _prov_names = {**raw_names, mcol: M['measure']}

    def _prov_list(r):
        return "; ".join(_prov_names[c] for c in [mcol] + raw_cols if is_provisional(r, c))
    cr_export['Provisional values (Eurostat flag p)'] = [_prov_list(r) for _, r in ranked.iterrows()]
    eu_export['Provisional values (Eurostat flag p)'] = [_prov_list(eu)]
    cr_export.insert(0, 'Rank', range(1, len(cr_export) + 1))
    eu_export.insert(0, 'Rank', 'reference')
    cr_export = pd.concat([cr_export, eu_export], ignore_index=True)
    cr_export.insert(1, 'Year', selected_year)
    st.download_button(
        label=f"Download Country Exposure Data ({selected_year}, CSV)",
        data=cr_export.to_csv(index=False),
        file_name=f"country_{country_mode}_{selected_year}.csv",
        mime="text/csv"
    )
