# Greenhouse curve explorer

An app for exploring and comparing Potato and
Tomato light-response curves and measurements from processed Excel workbooks.


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
