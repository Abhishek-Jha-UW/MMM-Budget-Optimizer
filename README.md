# MMM Budget Optimizer (Streamlit)

Marketing Mix Modeling (MMM) starter app: upload weekly spend + KPI data, fit a lightweight MMM (adstock + saturation + ridge regression), view channel contributions, and optimize next-period budget under constraints.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

- Push this folder to GitHub as its own repo
- App entrypoint: `app.py`
- (Optional) Add `OPENAI_API_KEY` as a Streamlit secret for AI-assisted setup and an auto-written executive summary

## Data format (CSV upload)

Required:
- `date`: weekly date (any parseable date)
- `kpi`: numeric KPI (e.g., revenue, conversions)

Optional:
- Any number of spend columns prefixed with `spend_` (e.g., `spend_search`, `spend_social`)
- Control variables (e.g., `price_index`, `promo_flag`, `site_sessions`, etc.)

Example columns:

`date, kpi, spend_search, spend_social, spend_display, promo_flag, price_index`

## What the model does (v1)

- **Adstock** per channel to model carryover
- **Saturation** per channel to model diminishing returns
- **Ridge regression** for stable estimation
- **Contribution decomposition** (baseline + controls + channels)
- **Budget optimization** (next week / next period) using response curves and constraints

