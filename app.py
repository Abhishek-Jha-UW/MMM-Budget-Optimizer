from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from model import (
    ChannelSpec,
    channel_summary,
    contributions_over_time,
    fit_mmm,
    generate_synthetic_mmm_data,
    infer_spend_columns,
    optimize_budget,
)


APP_TITLE = "MMM Budget Optimizer"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class AiPlan(BaseModel):
    channels: List[str] = Field(..., min_length=2, max_length=10)
    kpi_name: str = Field(..., min_length=1)
    controls: List[str] = Field(default_factory=list)
    notes: str = Field(default="", description="Short guidance / assumptions to show the user.")


def _get_api_key() -> str:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        key = str(st.secrets["OPENAI_API_KEY"]).strip()
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def propose_setup_with_ai(*, api_key: str, model_name: str, business_context: str) -> AiPlan:
    """
    Returns a small JSON plan: channels, KPI name, and suggested controls.
    """
    client = _openai_client(api_key)
    system = (
        "You are a marketing analytics lead designing a lightweight MMM.\n"
        "Return ONLY valid JSON that matches the schema exactly.\n"
        "No markdown, no extra keys."
    )
    user = f"""
Business context:
{business_context}

Task:
- Propose 3 to 7 paid/owned channels (simple names like search, social, display, email, affiliates).
- Propose a KPI name (e.g., revenue, conversions).
- Suggest 0 to 4 control variables typically relevant (e.g., promo_flag, price_index, seasonality_index).
- Add 2-4 sentences of notes (assumptions, caveats).

JSON schema:
{{
  "channels": ["search", "social"],
  "kpi_name": "revenue",
  "controls": ["promo_flag", "price_index"],
  "notes": "..."
}}
""".strip()

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        raise RuntimeError("Empty response while generating setup.")
    payload = json.loads(text)
    return AiPlan.model_validate(payload)

def executive_summary_with_ai(*, api_key: str, model_name: str, payload: Dict[str, Any]) -> str:
    """
    Writes an executive summary from computed outputs only.
    """
    client = _openai_client(api_key=api_key)
    system = (
        "You are a marketing mix modeling (MMM) analyst.\n"
        "Write a concise executive summary for a non-technical stakeholder.\n"
        "Use ONLY the provided JSON payload. Do not invent numbers or channels.\n"
        "Include: key drivers, what to do next, and 2-3 caveats.\n"
    )
    user = "Here are the computed MMM outputs (JSON):\n\n" + json.dumps(payload, indent=2)
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        raise RuntimeError("Empty response while generating executive summary.")
    return text.strip()


def _data_requirements_help() -> str:
    return (
        "Upload a weekly CSV with columns:\n"
        "- date (weekly)\n"
        "- kpi (numeric)\n"
        "- spend_<channel> columns, e.g. spend_search, spend_social\n"
        "- optional control columns: promo_flag, price_index, etc.\n"
    )


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Lightweight MMM (adstock + saturation + ridge) with a practical budget optimizer.")


with st.sidebar:
    st.subheader("About")
    st.caption("MSBA Graduate • University of Washington")
    st.markdown("---")
    st.markdown("**Optional AI features**")
    st.caption("Uses `OPENAI_API_KEY` from Streamlit secrets / env vars.")
    use_ai = st.toggle("Enable AI features", value=True)
    model_name = DEFAULT_OPENAI_MODEL
    auto_ai_summary = st.toggle("Auto-generate executive summary", value=True)

    st.markdown("---")
    st.markdown("**Model settings**")
    ridge_alpha = st.slider("Ridge regularization (alpha)", 0.1, 20.0, 2.0, 0.1)
    add_trend = st.toggle("Include trend", value=True)
    add_seasonality = st.toggle("Include seasonality (Fourier)", value=True)
    seasonality_K = st.slider("Seasonality terms (K)", 1, 5, 2, 1)


tab_data, tab_model, tab_opt = st.tabs(["1) Data", "2) Model + Insights", "3) Budget Optimizer"])


with tab_data:
    st.subheader("Load data")
    st.caption(_data_requirements_help())

    upload = st.file_uploader("Upload CSV", type=["csv"])
    demo_col1, demo_col2 = st.columns([1, 1])
    with demo_col1:
        use_demo = st.checkbox("Use synthetic demo data", value=upload is None)
    with demo_col2:
        n_weeks = st.slider("Demo weeks", 52, 156, 104, 4, disabled=not use_demo)

    business_context = st.text_area(
        "Business context (optional)",
        placeholder="Example: DTC skincare brand in the US. Channels: paid search, paid social, influencer, email. KPI: weekly revenue.",
        height=110,
    )

    df: Optional[pd.DataFrame] = None
    if upload is not None and not use_demo:
        df = pd.read_csv(upload)
    elif use_demo:
        df = generate_synthetic_mmm_data(n_weeks=int(n_weeks))

    if df is not None:
        st.success(f"Loaded {len(df):,} rows and {df.shape[1]:,} columns.")
        st.dataframe(df.head(30), use_container_width=True)

        spend_cols = infer_spend_columns(df)
        st.caption(f"Detected spend columns: {', '.join(spend_cols) if spend_cols else '(none)'}")


def _default_channels_from_df(df: pd.DataFrame) -> List[str]:
    cols = infer_spend_columns(df)
    return [c.replace("spend_", "", 1) for c in cols]


with tab_model:
    st.subheader("Fit MMM and review contributions")

    if df is None:
        st.info("Load a dataset in the Data tab to continue.")
        st.stop()

    if "date" not in df.columns or "kpi" not in df.columns:
        st.error("Your data must include 'date' and 'kpi' columns.")
        st.stop()

    # AI-assisted setup (optional; demo-friendly)
    if use_ai and business_context.strip():
        c_ai1, c_ai2 = st.columns([1, 1])
        with c_ai1:
            run_ai_setup = st.button("AI: propose channels + controls", type="primary")
        with c_ai2:
            st.caption("Uses your `OPENAI_API_KEY` from Streamlit secrets / env vars.")

        if run_ai_setup:
            key = _get_api_key()
            if not key:
                st.error("Missing OPENAI_API_KEY. Add it as a Streamlit secret (recommended) or env var.")
            else:
                with st.spinner("Generating setup…"):
                    try:
                        plan = propose_setup_with_ai(api_key=key, model_name=model_name, business_context=business_context)
                        st.session_state["ai_plan"] = plan.model_dump()
                    except Exception as e:
                        st.error(f"AI setup failed: {e}")

    ai_plan = st.session_state.get("ai_plan")
    suggested_channels = ai_plan.get("channels", []) if isinstance(ai_plan, dict) else []
    suggested_controls = ai_plan.get("controls", []) if isinstance(ai_plan, dict) else []
    notes = ai_plan.get("notes", "") if isinstance(ai_plan, dict) else ""
    if notes:
        st.info(notes)

    left, right = st.columns([1.2, 1.0])
    with left:
        detected = _default_channels_from_df(df)
        default_channels = suggested_channels or detected or ["search", "social", "display", "email"]
        selected_channels = st.multiselect(
            "Channels (mapped from spend_<channel> columns)",
            options=sorted(set(default_channels + detected)),
            default=[c for c in default_channels if f"spend_{c}" in df.columns] or detected,
            help="The app expects columns named spend_<channel> in your CSV.",
        )
    with right:
        numeric_cols = [c for c in df.columns if c not in ["date", "kpi"]]
        default_controls = [c for c in suggested_controls if c in numeric_cols]
        control_cols = st.multiselect(
            "Control columns (optional)",
            options=sorted(numeric_cols),
            default=default_controls,
            help="Controls are added as-is (numeric).",
        )

    st.markdown("**Channel transform parameters**")
    ch_specs: List[ChannelSpec] = []
    for ch in selected_channels:
        with st.expander(f"{ch}", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                decay = st.slider(f"{ch}: decay", 0.0, 0.95, 0.5, 0.01, key=f"decay_{ch}")
            with c2:
                hs = st.number_input(
                    f"{ch}: half-saturation (scale)",
                    min_value=1e-3,
                    value=1.0,
                    step=0.1,
                    key=f"hs_{ch}",
                )
            with c3:
                alpha = st.slider(f"{ch}: hill alpha", 0.2, 3.0, 1.3, 0.1, key=f"alpha_{ch}")
            ch_specs.append(ChannelSpec(name=ch, decay=float(decay), half_saturation=float(hs), hill_alpha=float(alpha)))

    run_fit = st.button("Fit model", type="primary", disabled=len(ch_specs) < 1)
    if run_fit:
        try:
            fit = fit_mmm(
                df,
                channels=ch_specs,
                control_columns=control_cols,
                ridge_alpha=float(ridge_alpha),
                add_trend=bool(add_trend),
                add_seasonality=bool(add_seasonality),
                seasonality_K=int(seasonality_K),
            )
            contrib = contributions_over_time(
                df,
                fit=fit,
                control_columns=control_cols,
                add_trend=bool(add_trend),
                add_seasonality=bool(add_seasonality),
                seasonality_K=int(seasonality_K),
            )
            st.session_state["fit"] = fit
            st.session_state["contrib"] = contrib
        except Exception as e:
            st.error(str(e))

    fit = st.session_state.get("fit")
    contrib = st.session_state.get("contrib")

    if fit is not None and contrib is not None:
        st.markdown("**Diagnostics**")
        d1, d2, d3 = st.columns(3)
        d1.metric("RMSE", f"{fit.diagnostics['rmse']:.2f}")
        d2.metric("MAE", f"{fit.diagnostics['mae']:.2f}")
        d3.metric("R² (centered)", f"{fit.diagnostics['r2_centered']:.3f}")

        st.markdown("**Actual vs predicted**")
        chart_df = contrib[["date", "kpi_actual", "kpi_pred"]].set_index("date")
        st.line_chart(chart_df, use_container_width=True)

        st.markdown("**Channel contribution totals**")
        summary = channel_summary(contrib, fit.channels)
        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download contributions (CSV)",
            data=contrib.to_csv(index=False).encode("utf-8"),
            file_name="mmm_contributions.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("AI executive summary")
        if not use_ai:
            st.info("Enable AI features in the sidebar to generate an executive summary.")
        else:
            key = _get_api_key()
            if not key:
                st.info("Add `OPENAI_API_KEY` in Streamlit secrets to enable the executive summary.")
            else:
                summary_payload: Dict[str, Any] = {
                    "app": "MMM Budget Optimizer",
                    "diagnostics": fit.diagnostics,
                    "channels": [c.model_dump() for c in fit.channels],
                    "channel_contribution_totals": channel_summary(contrib, fit.channels).to_dict(orient="records"),
                    "notes": (ai_plan.get("notes") if isinstance(ai_plan, dict) else "") or "",
                }

                cache_key = json.dumps(summary_payload, sort_keys=True)
                if st.session_state.get("ai_summary_cache_key") != cache_key:
                    st.session_state["ai_summary_cache_key"] = cache_key
                    st.session_state["ai_summary_text"] = None

                if auto_ai_summary and not st.session_state.get("ai_summary_text"):
                    with st.spinner("Generating executive summary…"):
                        try:
                            st.session_state["ai_summary_text"] = executive_summary_with_ai(
                                api_key=key,
                                model_name=model_name,
                                payload=summary_payload,
                            )
                        except Exception as e:
                            st.error(f"Executive summary failed: {e}")

                if st.session_state.get("ai_summary_text"):
                    st.markdown(st.session_state["ai_summary_text"])
                else:
                    if st.button("Generate executive summary"):
                        with st.spinner("Generating executive summary…"):
                            try:
                                st.session_state["ai_summary_text"] = executive_summary_with_ai(
                                    api_key=key,
                                    model_name=model_name,
                                    payload=summary_payload,
                                )
                            except Exception as e:
                                st.error(f"Executive summary failed: {e}")
                        if st.session_state.get("ai_summary_text"):
                            st.markdown(st.session_state["ai_summary_text"])


with tab_opt:
    st.subheader("Budget optimizer (next period)")

    fit = st.session_state.get("fit")
    contrib = st.session_state.get("contrib")
    if fit is None or contrib is None or df is None:
        st.info("Fit a model first in the Model + Insights tab.")
        st.stop()

    channels = [c.name for c in fit.channels]
    latest = df.tail(1)
    current = {ch: float(latest.get(f"spend_{ch}", pd.Series([0.0])).iloc[0]) for ch in channels}

    c1, c2, c3 = st.columns(3)
    with c1:
        total_budget = st.number_input("Total budget", min_value=0.0, value=float(sum(current.values())), step=100.0)
    with c2:
        horizon = st.selectbox("Horizon (weeks)", options=[1, 2, 4], index=0)
    with c3:
        step = st.number_input("Allocation step", min_value=1.0, value=50.0, step=25.0)

    st.markdown("**Constraints (optional)**")
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for ch in channels:
        row1, row2 = st.columns(2)
        with row1:
            mins[ch] = st.number_input(f"{ch}: min", min_value=0.0, value=0.0, step=50.0, key=f"min_{ch}")
        with row2:
            maxs[ch] = st.number_input(
                f"{ch}: max",
                min_value=0.0,
                value=float(total_budget),
                step=50.0,
                key=f"max_{ch}",
            )

    if st.button("Optimize budget", type="primary"):
        try:
            alloc = optimize_budget(
                fit=fit,
                current_spend=current,
                total_budget=float(total_budget),
                min_spend=mins,
                max_spend=maxs,
                horizon_weeks=int(horizon),
                step=float(step),
            )
            st.session_state["alloc"] = alloc
        except Exception as e:
            st.error(str(e))

    alloc = st.session_state.get("alloc")
    if isinstance(alloc, dict):
        alloc_df = (
            pd.DataFrame(
                {
                    "channel": list(alloc.keys()),
                    "current": [current.get(k, 0.0) for k in alloc.keys()],
                    "recommended": [float(v) for v in alloc.values()],
                }
            )
            .assign(delta=lambda d: d["recommended"] - d["current"])
            .sort_values("recommended", ascending=False)
        )
        st.dataframe(alloc_df, use_container_width=True)
        st.bar_chart(alloc_df.set_index("channel")[["current", "recommended"]], use_container_width=True)

        st.download_button(
            "Download recommended budget (CSV)",
            data=alloc_df.to_csv(index=False).encode("utf-8"),
            file_name="recommended_budget.csv",
            mime="text/csv",
        )

