# Greenhouse curve explorer

An independent Streamlit application for exploring and comparing Potato and
Tomato light-response curves and measurements from processed Excel workbooks. The app keeps the
uploaded dates and treatment labels unchanged and shares one loading,
aggregation, and plotting pipeline across both crops.

## Features

- Separate Potato and Tomato panels with independent filter state.
- Dynamic response-variable, treatment, plant, block/replicate, and date options.
- Response-variable groups beside the header: Light response curve, OJIP curve,
  Traits, Weather, and Soil. Selector labels omit units; plot axes include them.
- Light-response curves use PAR; the other groups use measurement dates, with
  all dates selected initially. Treatment time series show daily mean and SD
  across plants, including measurements without a recorded PAR value.
- Native Excel dates and `MM-DD-YYYY` text normalized to `YYYY-MM-DD`.
- Plant curves that merge duplicate LC records at the same plant, date, and PAR.
- Daily treatment mean, sample SD, and effective plant count.
- Period summaries that preserve the original Facet calculation order by
  averaging all valid records within each plant before treatment statistics.
- Interactive Plotly charts and matching 300 DPI PNG/PDF Matplotlib exports.
- CSV download of all plotting data for the current filter and statistical mode.
- File-signature-aware workbook caching and cached statistical results.

## Expected files

Place these workbooks in `data/`:

```text
data/Potato_LC_data_for_ts_plots.xlsx
data/Tomato_LC_data_for_ts_plots.xlsx
data/Potato_meteo_HH2_LSA_biomass_OJIP_tillHarvest_LSA_updated.xlsx
data/LMM_tomato_input_tillHarvest_LSA_updated.xlsx
```

Each workbook must contain `Date`, `Label`, `Water`, `Nitrogen`, and
either `Block` or `Replicate`. Light-response workbooks also require `PAR`.
`DAD` is optional. The first sheet is used (`QC_filter` for potato light curves).
A response variable is offered
only when its source column exists and has at least one finite numeric value.
Rows with no PAR and no supported response value are retained as date/plant
placeholders, allowing single-plant date facets to report `No data` explicitly.
Each group reads its own source workbook without joining daily measurements
onto light-curve rows. Missing or entirely empty variables are omitted; the
current tomato measurement workbook does not contain `gs_Avg(mmol_m2_s)`.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The command prints a local URL, normally `http://localhost:8501`.

## Deploy

For Streamlit Community Cloud, connect this repository and use the default
`streamlit_app.py` entry point. Keep all four Excel files committed under `data/`;
dependencies are installed from `requirements.txt` automatically.

For another Python host, install the requirements and start:

```text
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

No secrets or external services are required. Replacing a workbook changes
its modification time or size, which invalidates the cached file read on the
next Streamlit rerun.

## UI coverage

With the app running, install the development dependencies and run the targeted
Playwright coverage for group selection, axis labels, and treatment statistics:

```powershell
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python -m unittest discover -s tests -p test_variable_groups.py
```

Set `GREENHOUSE_BASE_URL` to target an app URL other than `http://localhost:8501`.

## Statistical definitions

The definitions below apply to light-response curves. For the other variable
groups, plant points are means within `Date × Label × Water × Nitrogen`, and
daily treatment statistics use `Date × Water × Nitrogen`. Dates remain on the
x-axis instead of being collapsed into a period summary.

Plant points are means over:

```text
Date × Label × Water × Nitrogen × PAR
```

Daily treatment statistics are calculated across those plant points for:

```text
Date × Water × Nitrogen × PAR
```

Period summaries first average duplicate valid records within:

```text
Date × Label × Water × Nitrogen × PAR
```

The resulting daily plant means are averaged within:

```text
Label × Water × Nitrogen × PAR
```

Each selected date with a valid daily plant mean therefore contributes equally
to that plant's period mean. Treatment mean, sample SD (`ddof=1`), and `n` are
then calculated across plant-period means. When `n = 1`, the mean remains
visible and SD is reported as not estimable.
