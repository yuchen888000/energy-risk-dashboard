"""Structural country energy data for EU-27 + CH, UK, NO, TR (2020–2024), all from Eurostat.

Downloaded from the Eurostat API at run time and cached for a day:
- import dependency (gas, oil, total): nrg_ind_id
- renewable share of gross final energy consumption: nrg_ind_ren
- greenhouse-gas intensity of GDP: env_air_gge (total GHG excluding LULUCF) divided by
  nama_10_gdp (GDP at current prices), in tonnes CO2-equivalent per million euro.
No value is typed in by hand. A country that lacks any of these series for a year (the UK
after Brexit, Switzerland, and Turkey's renewable share) is listed as not scored.
Every list follows the order of COUNTRIES; a missing value is None.
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

# (dataset, extra filters) for each downloaded series
_SERIES = {
    'gas':   ('nrg_ind_id', {'siec': 'G3000', 'unit': 'PC'}),      # natural gas
    'oil':   ('nrg_ind_id', {'siec': 'O4000XBIO', 'unit': 'PC'}),  # oil and petroleum products
    'total': ('nrg_ind_id', {'siec': 'TOTAL', 'unit': 'PC'}),      # all products
    'ren':   ('nrg_ind_ren', {'nrg_bal': 'REN', 'unit': 'PC'}),    # renewables, gross final consumption
    # thousand tonnes CO2-equivalent, total excluding LULUCF and memo items
    'ghg':   ('env_air_gge', {'airpol': 'GHG', 'unit': 'THS_T', 'src_crf': 'TOTX4_MEMO'}),
    'gdp':   ('nama_10_gdp', {'na_item': 'B1GQ', 'unit': 'CP_MEUR'}),  # GDP, million euro
}
SERIES = ('gas', 'oil', 'total', 'ren', 'carbon')


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
    """Country series for 2020–2024 from Eurostat.

    Returns (data, complete, failed):
    - data[series][year]: list in COUNTRIES order, None where Eurostat has no value.
      Series: gas, oil, total, ren (%), carbon (tCO2e per million euro of GDP).
    - complete[year]: list of bools, True where a country has all five series that year.
    - failed: names of datasets that could not be downloaded.
    """
    raw, failed = {}, []
    for key, (dataset, filters) in _SERIES.items():
        try:
            raw[key] = _fetch_eurostat(dataset, filters)
        except Exception:
            raw[key] = {}
            failed.append(dataset)
    data = {k: {} for k in SERIES}
    complete = {}
    for yr in YEARS:
        for key in ('gas', 'oil', 'total', 'ren'):
            data[key][yr] = [None if (v := raw[key].get((EUROSTAT_GEO[c], yr))) is None else round(v, 1)
                             for c in COUNTRIES]
        carbon = []
        for c in COUNTRIES:
            ghg, gdp = raw['ghg'].get((EUROSTAT_GEO[c], yr)), raw['gdp'].get((EUROSTAT_GEO[c], yr))
            carbon.append(round(ghg * 1000 / gdp) if ghg is not None and gdp else None)
        data['carbon'][yr] = carbon
        complete[yr] = [all(data[k][yr][i] is not None for k in SERIES) for i in range(len(COUNTRIES))]
    return data, complete, sorted(set(failed))


def dependency_key(commodity_name):
    """Series key and label of the import dependency relevant to a commodity."""
    if commodity_name == 'TTF Natural Gas':
        return 'gas', 'Gas Import Dependency'
    if commodity_name in ('WTI Crude Oil', 'Brent Crude Oil'):
        return 'oil', 'Oil Import Dependency'
    return 'total', 'Total Energy Dependency'


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
