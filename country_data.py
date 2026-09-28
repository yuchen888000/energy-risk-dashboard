"""Structural country energy data for EU-27 + CH, UK, NO, TR (2020–2024).

Import dependency (gas, oil, total) and the renewable share are downloaded from the
Eurostat API at run time (datasets nrg_ind_id and nrg_ind_ren) and cached for a day.
The hard-coded lists below are fallbacks only: they are hand-entered estimates, used
for any country, year or series Eurostat does not return, and every value is labelled
with where it came from. Carbon intensity has no Eurostat download here and is always
a hand-entered estimate.
Every list follows the order of COUNTRIES.
"""
import re

import requests
import streamlit as st

COUNTRIES = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands',
             'Poland', 'Belgium', 'Austria', 'Greece', 'Czech Republic',
             'Hungary', 'Romania', 'Bulgaria', 'Finland', 'Sweden',
             'Denmark', 'Ireland', 'Portugal', 'Lithuania', 'Latvia',
             'Estonia', 'Slovakia', 'Croatia', 'Slovenia', 'Luxembourg',
             'Cyprus', 'Malta',
             'Switzerland', 'United Kingdom', 'Norway', 'Turkey']

# gas_dep corrections:
# Romania (idx=11): produces ~10 bcm/yr, nearly self-sufficient → actual import dep 17-24%
# Denmark (idx=15): North Sea gas still active in 2020; 67% was too high → corrected to 50%
#                   (2021-2024 values 60,55,55,54 are approximately correct as fields deplete)
gas_dep = {
    2020: [89,98,93,99,0,79,100,82,99,98,85,17,96,97,6,50,100,100,100,100,100,86,55,99,100,100,100,100,48,0,99],
    2021: [91,98,93,99,5,82,100,81,99,97,85,18,94,96,8,60,100,100,100,100,100,85,57,99,100,100,100,100,47,0,99],
    2022: [95,98,93,99,15,78,100,80,99,97,85,20,92,95,10,55,100,100,100,100,100,85,60,99,100,100,100,100,47,0,99],
    2023: [95,98,93,99,68,78,100,80,99,97,85,22,92,95,12,55,100,100,100,100,100,85,60,99,100,100,100,100,47,0,99],
    2024: [94,98,93,99,70,77,100,80,99,97,84,24,91,94,12,54,100,100,100,100,100,84,58,99,100,100,100,100,46,0,99],
}
# oil_dep corrections:
# Denmark (idx=15): North Sea oil production ~75k bbl/day in 2020, consumption ~160k.
#   Import dependency ~52% in 2020, rising to ~65% by 2024 as fields deplete.
#   Previous value of 100% was completely wrong.
# FIX: UK oil import dependency corrected (~50%) — UK has North Sea domestic production
oil_dep = {
    2020: [96,98,92,99,95,97,99,93,100,97,82,45,100,91,100,52,100,100,100,100,60,92,82,100,100,96,100,100,50,0,93],
    2021: [96,98,92,99,96,97,99,93,100,97,83,44,100,90,100,55,100,100,100,100,58,92,80,100,100,96,100,100,51,0,93],
    2022: [96,98,93,99,96,97,99,94,100,97,84,42,100,90,100,58,100,100,100,100,55,92,78,100,100,96,100,100,52,0,92],
    2023: [97,98,93,99,96,97,99,94,100,97,84,40,100,90,100,62,100,100,100,100,52,92,76,100,100,97,100,100,53,0,92],
    2024: [97,98,93,99,96,97,99,94,100,97,84,39,100,90,100,65,100,100,100,100,50,92,75,100,100,97,100,100,54,0,92],
}
total_dep = {
    2020: [64,47,73,73,45,42,78,60,81,37,55,28,37,42,33,44,86,65,74,46,10,54,53,48,95,93,97,75,36,-580,72],
    2021: [64,47,74,73,46,41,78,61,81,37,55,28,37,43,31,45,86,65,74,45,8,53,52,48,95,93,97,75,35,-600,72],
    2022: [63,47,75,73,46,40,78,62,83,37,55,28,37,45,29,47,86,65,74,45,6,53,52,48,95,92,96,75,35,-620,72],
    2023: [63,47,75,73,46,40,78,62,81,37,55,28,37,45,29,47,86,65,74,45,3,53,52,48,95,92,96,75,35,-650,72],
    2024: [62,46,74,72,45,39,77,61,80,36,54,27,36,44,28,46,85,64,73,44,3,52,51,47,94,91,95,74,34,-660,71],
}
ren_share = {
    2020: [19,19,20,21,14,16,13,37,22,17,14,24,23,44,60,42,16,34,26,42,28,17,31,25,11,17,11,28,13,78,18],
    2021: [19,19,19,21,13,16,13,36,22,17,14,24,23,44,63,42,12,34,27,42,28,17,31,25,11,18,12,29,14,80,18],
    2022: [21,21,19,22,15,17,13,36,22,18,14,24,23,47,60,42,13,34,28,43,30,17,31,25,12,19,13,30,15,85,19],
    2023: [22,22,19,24,17,17,14,36,23,18,14,28,24,48,66,44,14,35,30,44,38,18,32,26,12,20,13,32,16,98,20],
    2024: [23,23,20,25,18,18,15,37,24,19,15,29,25,49,67,45,15,36,32,45,40,19,33,27,13,21,14,33,17,98,21],
}
# carbon_int corrections:
# Latvia (idx=19): ~53% hydro share → actual intensity ~115-125 tCO2/M€, NOT 180
# Luxembourg (idx=24): fuel tourism inflates energy/GDP → actual ~140-155, NOT 100 (France-level)
# Romania (idx=11): 30% hydro + 19% nuclear → ~245-265 tCO2/M€, NOT 340 (heavy-coal level)
carbon_int = {
    2020: [195,100,150,130,170,400,160,120,220,300,260,265,480,140,65,110,115,130,200,125,370,250,175,170,155,210,165,60,135,80,320],
    2021: [190,98,148,125,165,395,158,118,215,295,255,260,470,135,62,108,112,128,198,123,365,245,172,168,152,205,162,58,132,78,315],
    2022: [185,96,145,122,162,385,156,116,212,292,252,255,460,132,60,106,110,126,196,120,355,242,170,166,148,200,158,56,130,76,312],
    2023: [180,95,145,120,160,380,155,115,210,290,250,250,450,130,58,105,108,125,195,118,350,240,170,165,144,195,155,55,128,75,310],
    2024: [176,93,142,118,157,375,152,113,208,287,247,245,445,128,56,103,106,123,192,115,345,237,168,163,140,190,152,54,126,73,305],
}
price_sens = {
    2020: [9.0,7.2,8.5,7.0,8.2,7.0,7.8,7.5,8.2,7.2,7.5,6.8,7.5,7.5,3.2,5.5,6.8,6.5,8.0,7.2,7.5,7.0,5.8,6.2,6.0,7.8,7.2,6.8,7.8,2.2,8.5],
    2021: [9.1,7.3,8.6,7.1,8.3,7.0,7.9,7.6,8.3,7.1,7.5,6.7,7.4,7.6,3.3,5.6,6.9,6.6,8.1,7.3,7.6,7.1,5.9,6.3,6.1,7.9,7.3,6.9,7.9,2.1,8.5],
    2022: [9.5,7.8,9.0,7.5,8.8,7.2,8.2,8.0,8.8,7.3,7.8,6.8,7.5,8.0,3.5,5.8,7.2,7.0,8.5,7.8,8.0,7.3,6.2,6.8,6.5,8.2,7.5,7.2,8.2,2.0,8.8],
    2023: [9.2,7.5,8.8,7.2,8.5,6.8,8.0,7.8,8.5,7.0,7.5,6.5,7.2,7.8,3.5,5.8,7.0,6.8,8.2,7.5,7.8,7.2,6.0,6.5,6.2,8.0,7.3,7.0,8.0,2.0,8.5],
    2024: [9.0,7.3,8.6,7.0,8.3,6.6,7.8,7.6,8.3,6.8,7.3,6.3,7.0,7.6,3.4,5.6,6.8,6.6,8.0,7.3,7.6,7.0,5.8,6.3,6.0,7.8,7.1,6.8,7.8,1.8,8.3],
}


# ─── Eurostat download ───
EUROSTAT_GEO = {
    'Germany': 'DE', 'France': 'FR', 'Italy': 'IT', 'Spain': 'ES', 'Netherlands': 'NL',
    'Poland': 'PL', 'Belgium': 'BE', 'Austria': 'AT', 'Greece': 'EL', 'Czech Republic': 'CZ',
    'Hungary': 'HU', 'Romania': 'RO', 'Bulgaria': 'BG', 'Finland': 'FI', 'Sweden': 'SE',
    'Denmark': 'DK', 'Ireland': 'IE', 'Portugal': 'PT', 'Lithuania': 'LT', 'Latvia': 'LV',
    'Estonia': 'EE', 'Slovakia': 'SK', 'Croatia': 'HR', 'Slovenia': 'SI', 'Luxembourg': 'LU',
    'Cyprus': 'CY', 'Malta': 'MT', 'Switzerland': 'CH', 'United Kingdom': 'UK',
    'Norway': 'NO', 'Turkey': 'TR',
}
YEARS = [2020, 2021, 2022, 2023, 2024]
_EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}"
EUROSTAT = "Eurostat"
ESTIMATE = "hand-entered estimate"

# (dataset, extra filters) for each downloaded series
_SERIES = {
    'gas':   ('nrg_ind_id', {'siec': 'G3000', 'unit': 'PC'}),      # natural gas
    'oil':   ('nrg_ind_id', {'siec': 'O4000XBIO', 'unit': 'PC'}),  # oil and petroleum products
    'total': ('nrg_ind_id', {'siec': 'TOTAL', 'unit': 'PC'}),      # all products
    'ren':   ('nrg_ind_ren', {'nrg_bal': 'REN', 'unit': 'PC'}),    # renewables, gross final consumption
}


def _jsonstat_values(js):
    """{(geo, year): value} from a Eurostat JSON-stat 2.0 response filtered to one series."""
    ids, sizes = js['id'], js['size']
    cats = {d: js['dimension'][d]['category']['index'] for d in ids}
    # position -> code for each dimension
    pos = {d: {v: k for k, v in cats[d].items()} for d in ids}
    out = {}
    for flat, val in js.get('value', {}).items():
        if val is None:
            continue
        idx, rem = {}, int(flat)
        for d, n in zip(reversed(ids), reversed(sizes)):
            idx[d] = pos[d][rem % n]
            rem //= n
        out[(idx['geo'], int(idx['time']))] = float(val)
    return out


@st.cache_data(ttl=86400, show_spinner="Downloading country data from Eurostat...")
def _fetch_eurostat(dataset, filters):
    params = [('format', 'JSON'), ('lang', 'EN'), ('freq', 'A'),
              ('sinceTimePeriod', str(YEARS[0])), ('untilTimePeriod', str(YEARS[-1]))]
    params += list(filters.items())
    params += [('geo', g) for g in EUROSTAT_GEO.values()]
    # Raises on failure, so a failed download is not cached for a day.
    r = requests.get(_EUROSTAT_URL.format(dataset), params=params, timeout=30)
    r.raise_for_status()
    return _jsonstat_values(r.json())


def load_country_data():
    """Country series for 2020–2024 with the source of every value.

    Returns (data, source): data[series][year] is a list in COUNTRIES order, source has
    the same shape with EUROSTAT or ESTIMATE. Series: gas, oil, total, ren (downloaded,
    estimate fallback) and carbon (always the hand-entered estimate).
    """
    fallback = {'gas': gas_dep, 'oil': oil_dep, 'total': total_dep, 'ren': ren_share}
    data, source = {}, {}
    for key, (dataset, filters) in _SERIES.items():
        try:
            got = _fetch_eurostat(dataset, filters)
        except Exception:
            got = {}
        data[key], source[key] = {}, {}
        for yr in YEARS:
            vals, srcs = [], []
            for i, country in enumerate(COUNTRIES):
                v = got.get((EUROSTAT_GEO[country], yr))
                if v is None:
                    vals.append(fallback[key][yr][i]); srcs.append(ESTIMATE)
                else:
                    vals.append(round(v, 1)); srcs.append(EUROSTAT)
            data[key][yr], source[key][yr] = vals, srcs
    data['carbon'] = carbon_int
    source['carbon'] = {yr: [ESTIMATE] * len(COUNTRIES) for yr in YEARS}
    return data, source


def dependency_key(commodity_name):
    """Series key and label of the import dependency relevant to a commodity."""
    if commodity_name == 'TTF Natural Gas':
        return 'gas', 'Gas Import Dependency'
    if commodity_name in ('WTI Crude Oil', 'Brent Crude Oil'):
        return 'oil', 'Oil Import Dependency'
    return 'total', 'Total Energy Dependency'


def dependency_for(commodity_name):
    """Return (column name, label, {year: values}) of the dependency relevant to a commodity."""
    if commodity_name == 'TTF Natural Gas':
        return 'Gas Dep. (%)', 'Gas Import Dependency', gas_dep
    if commodity_name in ('WTI Crude Oil', 'Brent Crude Oil'):
        return 'Oil Dep. (%)', 'Oil Import Dependency', oil_dep
    return 'Total Energy Dep. (%)', 'Total Energy Dependency', total_dep


# Words that show a headline is about a country: its name, short forms and adjective.
# Matched case-sensitively on whole words, so "Polish" matches but "polish" does not.
COUNTRY_TERMS = {
    'Germany': ['Germany', 'German', 'Germans'],
    'France': ['France', 'French'],
    'Italy': ['Italy', 'Italian'],
    'Spain': ['Spain', 'Spanish'],
    'Netherlands': ['Netherlands', 'Dutch', 'Holland'],
    'Poland': ['Poland', 'Polish'],
    'Belgium': ['Belgium', 'Belgian'],
    'Austria': ['Austria', 'Austrian'],
    'Greece': ['Greece', 'Greek'],
    'Czech Republic': ['Czech Republic', 'Czechia', 'Czech'],
    'Hungary': ['Hungary', 'Hungarian'],
    'Romania': ['Romania', 'Romanian'],
    'Bulgaria': ['Bulgaria', 'Bulgarian'],
    'Finland': ['Finland', 'Finnish'],
    'Sweden': ['Sweden', 'Swedish'],
    'Denmark': ['Denmark', 'Danish'],
    'Ireland': ['Ireland', 'Irish'],
    'Portugal': ['Portugal', 'Portuguese'],
    'Lithuania': ['Lithuania', 'Lithuanian'],
    'Latvia': ['Latvia', 'Latvian'],
    'Estonia': ['Estonia', 'Estonian'],
    'Slovakia': ['Slovakia', 'Slovak', 'Slovakian'],
    'Croatia': ['Croatia', 'Croatian'],
    'Slovenia': ['Slovenia', 'Slovenian', 'Slovene'],
    'Luxembourg': ['Luxembourg', 'Luxembourgish'],
    'Cyprus': ['Cyprus', 'Cypriot'],
    'Malta': ['Malta', 'Maltese'],
    'Switzerland': ['Switzerland', 'Swiss'],
    'United Kingdom': ['United Kingdom', 'UK', 'U.K.', 'Britain', 'British'],
    'Norway': ['Norway', 'Norwegian'],
    'Turkey': ['Turkey', 'Türkiye', 'Turkiye', 'Turkish'],
}


def mentions_country(text, country):
    """True if `text` names `country` or uses its adjective (whole words only)."""
    terms = COUNTRY_TERMS.get(country, [country])
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in terms) + r')(?!\w)'
    return re.search(pattern, text) is not None
