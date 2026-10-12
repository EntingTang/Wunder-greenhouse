# Greenhouse curve explorer

An independent Streamlit application for exploring and comparing Potato and
Tomato light-response curves from processed Excel workbooks. The app keeps the
uploaded dates and treatment labels unchanged and shares one loading,
aggregation, and plotting pipeline across both crops.

## Features

- Separate Potato and Tomato panels with independent filter state.
- Dynamic response-variable, treatment, plant, block/replicate, and date options.
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
```

Each workbook must contain `Date`, `Label`, `Water`, `Nitrogen`, `PAR`, and
either `Block` or `Replicate`. `DAD` is optional. A response variable is offered
only when its source column exists and has at least one finite numeric value.

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The command prints a local URL, normally `http://localhost:8501`.

## Deploy

For Streamlit Community Cloud, connect this repository and set `app.py` as the
entry point. Keep both Excel files committed under `data/`; dependencies are
installed from `requirements.txt` automatically.

For another Python host, install the requirements and start:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

No secrets or external services are required. Replacing either workbook changes
its modification time or size, which invalidates the cached file read on the
next Streamlit rerun.

## Statistical definitions

Plant points are means over:

```text
Date × Label × Water × Nitrogen × PAR
```

Daily treatment statistics are calculated across those plant points for:

```text
Date × Water × Nitrogen × PAR
```

Period summaries first average all selected, valid raw records within:

```text
Label × Water × Nitrogen × PAR
```

They then calculate treatment mean, sample SD (`ddof=1`), and `n`. This means a
date with more records has more weight in its plant's period mean. When `n = 1`,
the mean remains visible and SD is reported as not estimable.
