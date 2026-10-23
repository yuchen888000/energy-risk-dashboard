"""Country energy data for EU-27 + CH, UK, NO, TR, all from Eurostat.

Downloaded from the Eurostat API at run time and cached for a day:
- import dependency (gas, oil, all products): nrg_ind_id, in % (net imports / gross
  available energy of that fuel)
- gross available energy by fuel (natural gas; oil and petroleum products excluding the biofuel
  portion, O4000XBIO; all products): nrg_bal_c, in thousand tonnes of oil equivalent (ktoe)
- renewable share of gross final energy consumption: nrg_ind_ren
- greenhouse-gas intensity of GDP: env_air_gge (total GHG excluding LULUCF) divided by
  nama_10_gdp at constant prices (chain-linked volumes, 2020 reference year), in tonnes
  CO2-equivalent per million euro.
No value is typed in by hand. Years run from FIRST_YEAR to the latest year Eurostat
returns; a country without a value is shown as missing, never filled in. Values Eurostat flags
as provisional (status "p") keep that flag, and every measure derived from one is provisional too.
"""
import re

import pandas as pd
import requests
import streamlit as st

COUNTRIES = ['Germany', 'France', 'Italy', 'Spain', 'Netherlands',
             'Poland', 'Belgium', 'Austria', 'Greece', 'Czech Republic',
             'Hungary', 'Romania', 'Bulgaria', 'Finland', 'Sweden',
             'Denmark', 'Ireland', 'Portugal', 'Lithuania', 'Latvia',
             'Estonia', 'Slovakia', 'Croatia', 'Slovenia', 'Luxembourg',
             'Cyprus', 'Malta',
             'Switzerland', 'United Kingdom', 'Norway', 'Turkey']
EU27 = 'EU-27'

# ─── Eurostat download ───
EUROSTAT_GEO = {
    'Germany': 'DE', 'France': 'FR', 'Italy': 'IT', 'Spain': 'ES', 'Netherlands': 'NL',
    'Poland': 'PL', 'Belgium': 'BE', 'Austria': 'AT', 'Greece': 'EL', 'Czech Republic': 'CZ',
    'Hungary': 'HU', 'Romania': 'RO', 'Bulgaria': 'BG', 'Finland': 'FI', 'Sweden': 'SE',
    'Denmark': 'DK', 'Ireland': 'IE', 'Portugal': 'PT', 'Lithuania': 'LT', 'Latvia': 'LV',
    'Estonia': 'EE', 'Slovakia': 'SK', 'Croatia': 'HR', 'Slovenia': 'SI', 'Luxembourg': 'LU',
    'Cyprus': 'CY', 'Malta': 'MT', 'Switzerland': 'CH', 'United Kingdom': 'UK',
    'Norway': 'NO', 'Turkey': 'TR',
    EU27: 'EU27_2020',   # reference row, never ranked
}
_GEO_NAME = {v: k for k, v in EUROSTAT_GEO.items()}
FIRST_YEAR = 2020
_EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}"

# (dataset, filters) for each downloaded series
_SERIES = {
    'dep_gas':   ('nrg_ind_id', {'siec': 'G3000', 'unit': 'PC'}),       # natural gas
    'dep_oil':   ('nrg_ind_id', {'siec': 'O4000XBIO', 'unit': 'PC'}),   # oil and petroleum products
    'dep_total': ('nrg_ind_id', {'siec': 'TOTAL', 'unit': 'PC'}),       # all products
    'gae_gas':   ('nrg_bal_c', {'nrg_bal': 'GAE', 'siec': 'G3000', 'unit': 'KTOE'}),
    'gae_oil':   ('nrg_bal_c', {'nrg_bal': 'GAE', 'siec': 'O4000XBIO', 'unit': 'KTOE'}),
    'gae_total': ('nrg_bal_c', {'nrg_bal': 'GAE', 'siec': 'TOTAL', 'unit': 'KTOE'}),
    'ren':       ('nrg_ind_ren', {'nrg_bal': 'REN', 'unit': 'PC'}),     # renewables, gross final consumption
    # thousand tonnes CO2-equivalent, total excluding LULUCF and memo items
    'ghg':       ('env_air_gge', {'airpol': 'GHG', 'unit': 'THS_T', 'src_crf': 'TOTX4_MEMO'}),
    # GDP in million euro, chain-linked volumes (2020 prices), so inflation does not
    # show up as a fall in emissions intensity
    'gdp_clv':   ('nama_10_gdp', {'na_item': 'B1GQ', 'unit': 'CLV20_MEUR'}),
}

# The question, ranking measure and columns for each commodity.
MODES = {
    'gas': dict(
        question="Which countries are most exposed to imported gas?",
        measure='Net gas imports (% of all energy)',
        share='Gas share of energy (%)', dep='Gas import dependency (%)',
        inputs=('gae_gas', 'gae_total', 'dep_gas'), fuel='Natural gas'),   # Eurostat's label for G3000
    'oil': dict(
        question="Which countries are most exposed to imported oil?",
        measure='Net oil imports (% of all energy)',
        share='Oil share of energy (%)', dep='Oil import dependency (%)',
        inputs=('gae_oil', 'gae_total', 'dep_oil'),
        fuel='Oil and petroleum products (excluding biofuel portion)'),   # Eurostat's label for O4000XBIO
    'carbon': dict(
        question="Which economies emit the most greenhouse gas per unit of output?",
        measure='GHG intensity (tCO2e per M€, 2020 prices)',
        share=None, dep=None,
        inputs=('ghg', 'gdp_clv'), fuel=None),
}


def _jsonstat_values(js):
    """({(geo, year): value}, {(geo, year): status}) from a Eurostat JSON-stat 2.0 response filtered to one
    series. The status is Eurostat's flag for that value ("p" = provisional), keyed like the value."""
    ids, sizes = js['id'], js['size']
    cats = {d: js['dimension'][d]['category']['index'] for d in ids}
    # position -> code for each dimension
    pos = {d: {v: k for k, v in cats[d].items()} for d in ids}
    status = js.get('status', {}) or {}
    out, flags = {}, {}
    for flat, val in js.get('value', {}).items():
        if val is None:
            continue
        idx, rem = {}, int(flat)
        for d, n in zip(reversed(ids), reversed(sizes)):
            idx[d] = pos[d][rem % n]
            rem //= n
        key = (idx['geo'], int(idx['time']))
        out[key] = float(val)
        if status.get(str(flat)):
            flags[key] = str(status[str(flat)])
    return out, flags


@st.cache_data(ttl=86400, show_spinner="Downloading country data from Eurostat...")
def _fetch_eurostat(dataset, filters):
    """(values, status flags, last update) for one series, from FIRST_YEAR to the latest year available."""
    params = [('format', 'JSON'), ('lang', 'EN'), ('freq', 'A'),
              ('sinceTimePeriod', str(FIRST_YEAR))]
    params += list(filters.items())
    params += [('geo', g) for g in EUROSTAT_GEO.values()]
    # Raises on failure, so a failed download is not cached for a day.
    r = requests.get(_EUROSTAT_URL.format(dataset), params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    values, flags = _jsonstat_values(js)
    return values, flags, str(js.get('updated', ''))[:10]


# Inputs of each derived column, so a measure built from a provisional value is marked provisional too.
_DERIVED_FROM = {
    'gas_share': ('gae_gas', 'gae_total'), 'gas_exposure': ('gae_gas', 'gae_total', 'dep_gas'),
    'oil_share': ('gae_oil', 'gae_total'), 'oil_exposure': ('gae_oil', 'gae_total', 'dep_oil'),
    'ghg_int': ('ghg', 'gdp_clv'),
}


def is_provisional(row, col):
    """True if Eurostat flags `col` (or an input of it) as provisional in this row of load_country_data."""
    return col in str(row.get('provisional', '') or '').split(',')


def _ratio(a, b, scale=100.0):
    return None if a is None or not b else a / b * scale


def load_country_data():
    """One row per (country, year) with the raw Eurostat inputs and the derived measures.

    Returns (df, failed, updated):
    - df columns: Country, Year, the raw series (dep_gas, dep_oil, dep_total in %; gae_* in
      ktoe; ren in %; ghg in kt; gdp_clv in M€) and the derived columns gas_share and
      oil_share (% of gross available energy), gas_exposure and oil_exposure (net imports of
      the fuel as % of all gross available energy = share x dependency, with a negative
      dependency (net exporter) counted as 0), ghg_int (tCO2e per million euro of GDP at 2020 prices).
      `provisional` lists the columns whose value Eurostat flags as provisional, comma-separated ("" if
      none); a derived column is listed when any of its inputs is.
    - failed: names of datasets that could not be downloaded.
    - updated: {dataset: last Eurostat update date}.
    """
    raw, flags, failed, updated = {}, {}, [], {}
    for key, (dataset, filters) in _SERIES.items():
        try:
            raw[key], flags[key], updated[dataset] = _fetch_eurostat(dataset, filters)
        except Exception:
            raw[key], flags[key] = {}, {}
            failed.append(dataset)
    years = sorted({yr for vals in raw.values() for (_, yr) in vals})
    rows = []
    for name, geo in EUROSTAT_GEO.items():
        for yr in years:
            r = {'Country': name, 'Year': yr}
            for key in _SERIES:
                r[key] = raw[key].get((geo, yr))
            for fuel in ('gas', 'oil'):
                share = _ratio(r[f'gae_{fuel}'], r['gae_total'])
                dep = r[f'dep_{fuel}']
                r[f'{fuel}_share'] = share
                if share is None:
                    r[f'{fuel}_exposure'] = None
                elif share == 0:
                    # No use of the fuel: nothing to import, whatever the dependency field says.
                    r[f'{fuel}_exposure'] = 0.0
                elif dep is None:
                    r[f'{fuel}_exposure'] = None
                else:
                    # share x dependency = net imports of the fuel / all gross available energy.
                    # Dependency below 0 = net exporter: no net imports, counted as 0. Above 100
                    # (imports that also refilled stocks) is kept, so the product stays equal to
                    # net imports over gross available energy.
                    r[f'{fuel}_exposure'] = share * max(dep, 0.0) / 100
            r['ghg_int'] = _ratio(r['ghg'], r['gdp_clv'], scale=1000.0)  # kt -> t, per M€
            prov = {key for key in _SERIES if flags[key].get((geo, yr)) == 'p' and r[key] is not None}
            for col, inputs in _DERIVED_FROM.items():
                if r[col] is not None and prov & set(inputs):
                    prov.add(col)
            r['provisional'] = ",".join(sorted(prov))
            rows.append(r)
    df = pd.DataFrame(rows)
    return df, sorted(set(failed)), updated


def measure_column(mode):
    """Name of the derived column that ranks countries in a mode."""
    return {'gas': 'gas_exposure', 'oil': 'oil_exposure', 'carbon': 'ghg_int'}[mode]


def available_years(df, mode):
    """Years for which Eurostat publishes the EU-27 value of the mode's measure, i.e. years
    with complete member-state data. Provisional later years for only some inputs (for
    example the renewable share) do not count."""
    if df.empty:
        return []
    col = measure_column(mode)
    eu = df[(df['Country'] == EU27) & df[col].notna()]
    return sorted(int(y) for y in eu['Year'])


def dependency_key(commodity_name):
    """Country-section mode for a commodity: 'gas', 'oil' or 'carbon'."""
    if commodity_name == 'TTF Natural Gas':
        return 'gas'
    if commodity_name in ('WTI Crude Oil', 'Brent Crude Oil'):
        return 'oil'
    return 'carbon'


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
