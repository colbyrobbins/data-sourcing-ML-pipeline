"""Streamlit web interface for the Pokémon Card Fair-Value Model.

Allows collectors and buyers to search cards semantically, estimate fair market values,
and judge whether an asking price is a deal, fair, or overpriced.
"""

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Pokémon Card Fair-Value Finder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Configuration & API Helpers
# -----------------------------------------------------------------------------

DEFAULT_API_URL = os.environ.get("API_URL", "https://pokemon-fair-value-api-3tplqbtcoa-uc.a.run.app")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_health(base_url: str):
    # Cloud Run can take ~10 s to cold-start, so allow more than a warm request needs.
    resp = requests.get(f"{base_url}/health", timeout=20)
    return resp.status_code, resp.json() if resp.status_code == 200 else {}

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
        width=120,
    )
    st.title("Settings")
    api_url = st.text_input(
        "API Base URL",
        value=DEFAULT_API_URL,
        help="URL of the deployed FastAPI backend (e.g. Google Cloud Run or local uvicorn server).",
    ).rstrip("/")

    st.markdown("---")
    st.subheader("API Status")

    api_healthy = False
    health_data = {}
    try:
        status_code, health_json = fetch_health(api_url)
        if status_code == 200:
            api_healthy = True
            health_data = health_json
            st.success(f"🟢 **Connected** ({health_data.get('model_name', 'Model')})")
            st.caption(f"Snapshot: `{health_data.get('snapshot_date', 'N/A')}`")
            if health_data.get("search_enabled"):
                st.caption("🔍 Semantic search: **Enabled**")
            else:
                st.caption("ℹ️ Semantic search: **Disabled** (Prediction only)")
        else:
            st.warning(f"⚠️ API returned HTTP {status_code}")
    except Exception as exc:
        st.error("🔴 **API Disconnected**")
        st.caption(f"Cannot reach `{api_url}`. Make sure your API service is running.")

    st.markdown("---")
    st.markdown(
        """
        **About this project**:
        - **Pipeline**: TCGCSV snapshot → Google Cloud Storage → scikit-learn + MLflow
        - **Deployment**: Google Cloud Run + Streamlit Cloud
        - **Team**: AIPI 510 Module Project 3
        """
    )


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

st.title("⚡ Pokémon Card Fair-Value Finder")
st.markdown(
    "Estimate the fair market price of any Pokémon card and determine whether an asking price "
    "is a **deal**, **fair**, or **overpriced**."
)

tab_search, tab_manual, tab_about = st.tabs(
    ["🔍 Search & Evaluate Deal", "📝 Manual Card Estimator", "ℹ️ Model & Architecture"]
)

# -----------------------------------------------------------------------------
# Tab 1: Semantic Card Search & Deal Verdict
# -----------------------------------------------------------------------------
with tab_search:
    st.subheader("1. Find Your Card")
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search by name, set, collector number, or description",
            value="Charizard ex 199/165",
            placeholder="e.g. 'charizard 151', 'umbreon vmax alt art', or 'fiery flying dragon'",
        )
    with col_k:
        top_k = st.number_input("Max results", min_value=1, max_value=15, value=5)

    search_btn = st.button("Search Cards", type="primary", width="stretch")

    search_key = (query.strip(), int(top_k))
    if query.strip() and (search_btn or st.session_state.get("last_search") != search_key):
        st.session_state["last_search"] = search_key
        st.session_state["search_results"] = []
        if not api_healthy:
            st.warning(
                f"Please ensure the API backend is running at `{api_url}` to execute card searches."
            )
        else:
            with st.spinner("Searching card catalog..."):
                try:
                    search_resp = requests.get(
                        f"{api_url}/search",
                        params={"q": query, "k": top_k},
                        timeout=30,
                    )
                    if search_resp.status_code == 200:
                        results = search_resp.json().get("results", [])
                        if not results:
                            st.info("No matching cards found. Try adjusting your query terms.")
                        else:
                            st.write(f"Found **{len(results)}** matching card(s):")
                            st.session_state["search_results"] = results
                    elif search_resp.status_code == 503:
                        st.info("Semantic search index is not loaded on this API instance.")
                    else:
                        st.error(f"Search failed with code {search_resp.status_code}: {search_resp.text}")
                except Exception as err:
                    st.error(f"Failed to query search API: {err}")

    # Display results if available in session
    if "search_results" in st.session_state and st.session_state["search_results"]:
        results = st.session_state["search_results"]

        card_options = {
            f"{r['name']} — {r.get('set_name', 'Unknown Set')} ({r.get('rarity', 'Unknown Rarity')})": idx
            for idx, r in enumerate(results)
        }

        selected_label = st.selectbox(
            "Select Card to Evaluate:",
            options=list(card_options.keys()),
        )
        selected_card_data = results[card_options[selected_label]]

        col_img, col_info, col_eval = st.columns([1, 2, 2])

        with col_img:
            img_url = selected_card_data.get("image_url")
            if img_url:
                st.image(img_url, width="stretch")
            else:
                st.caption("No image preview available")

        with col_info:
            st.markdown(f"### {selected_card_data['name']}")
            st.markdown(f"**Set:** {selected_card_data.get('set_name', 'N/A')}")
            st.markdown(f"**Rarity:** {selected_card_data.get('rarity', 'N/A')}")
            if selected_card_data.get("card", {}).get("number"):
                st.markdown(f"**Collector Number:** `{selected_card_data['card']['number']}`")

            printings = selected_card_data.get("printings", {})
            st.markdown("**Snapshot Market Prices by Printing:**")
            for p_name, p_price in printings.items():
                st.markdown(f"- **{p_name}**: `${p_price:.2f}`")

        with col_eval:
            st.markdown("### 2. Evaluate Asking Price")
            available_printings = list(printings.keys()) or ["Normal"]
            selected_printing = st.selectbox("Choose Printing:", options=available_printings)

            # Default asking price to the snapshot price if known
            default_asking = float(printings.get(selected_printing, 20.0))
            user_asking = st.number_input(
                "Asking Price ($)",
                min_value=0.01,
                value=round(default_asking, 2),
                step=1.0,
                format="%.2f",
            )

            evaluate_btn = st.button("Calculate Fair Value & Verdict", type="primary")

            if evaluate_btn:
                predict_payload = dict(selected_card_data["card"])
                predict_payload["printing"] = selected_printing
                predict_payload["asking_price"] = user_asking

                try:
                    with st.spinner("Calculating fair market value..."):
                        pred_resp = requests.post(
                            f"{api_url}/predict",
                            json=predict_payload,
                            timeout=30,
                        )
                    if pred_resp.status_code == 200:
                        pred_data = pred_resp.json()
                        fair_val = pred_data["fair_value"]
                        verdict = pred_data.get("verdict")

                        st.markdown("---")
                        st.metric(
                            label="Predicted Fair Market Value",
                            value=f"${fair_val:,.2f}",
                            delta=f"{((fair_val - user_asking) / fair_val) * 100:+.1f}% vs Asking",
                        )

                        if verdict == "deal":
                            discount = ((fair_val - user_asking) / fair_val) * 100
                            st.success(
                                f"🎉 **GREAT DEAL!** Asking price is **{discount:.1f}% below** estimated fair value."
                            )
                        elif verdict == "overpriced":
                            premium = ((user_asking - fair_val) / fair_val) * 100
                            st.error(
                                f"⚠️ **OVERPRICED!** Asking price is **{premium:.1f}% above** estimated fair value."
                            )
                        else:
                            st.info("⚖️ **FAIR PRICE.** Asking price is within 15% of estimated fair market value.")

                        st.caption(
                            f"Model: `{pred_data.get('model_name')}` | Baseline Data Snapshot: `{pred_data.get('snapshot_date')}`"
                        )
                    else:
                        st.error(f"Prediction failed ({pred_resp.status_code}): {pred_resp.text}")
                except Exception as exc:
                    st.error(f"Failed to communicate with prediction API: {exc}")


# -----------------------------------------------------------------------------
# Tab 2: Manual Card Estimator
# -----------------------------------------------------------------------------
RARITIES = [
    "Common", "Uncommon", "Rare", "Holo Rare", "Double Rare", "Ultra Rare",
    "Illustration Rare", "Special Illustration Rare", "Hyper Rare", "Secret Rare",
    "Shiny Holo Rare", "Shiny Rare", "Rainbow Rare", "ACE SPEC Rare", "Radiant Rare",
    "Amazing Rare", "Classic Collection", "Promo",
]
PRINTINGS = [
    "Normal", "Holofoil", "Reverse Holofoil", "1st Edition", "1st Edition Holofoil",
    "Unlimited", "Unlimited Holofoil",
]
CARD_TYPES = [
    "Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting", "Darkness", "Metal",
    "Fairy", "Dragon", "Colorless", "Supporter", "Item", "Stadium", "Tool", "Basic Energy",
    "Special Energy",
]
STAGES = ["Basic", "Stage 1", "Stage 2", "VMAX", "VSTAR", "Mega", "BREAK Evolution", "Level Up"]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
with tab_manual:
    st.subheader("Manual Card Attribute Estimator")
    st.markdown(
        "For custom cards or cards not found in the search index, input their visible attributes below:"
    )

    with st.form("manual_predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            m_name = st.text_input(
                "Card Name *",
                value="Umbreon VMAX (Alternate Art Secret)",
                help="Words like ex, VMAX, Full Art or Alternate Art in the name are model features.",
            )
            # Values as they appear in the TCGplayer data the model was trained on.
            m_rarity = st.selectbox("Rarity *", RARITIES, index=RARITIES.index("Secret Rare"))
            m_printing = st.selectbox("Printing *", PRINTINGS, index=PRINTINGS.index("Holofoil"))
            m_card_type = st.selectbox("Card Type", CARD_TYPES, index=CARD_TYPES.index("Darkness"))
            m_stage = st.selectbox("Stage", STAGES, index=STAGES.index("VMAX"))

        with col2:
            m_hp = st.number_input("HP", min_value=0.0, max_value=400.0, value=310.0, step=10.0)
            m_retreat = st.number_input("Retreat Cost", min_value=0.0, max_value=5.0, value=2.0, step=1.0)
            m_number = st.text_input("Collector Number (e.g. 215/203)", value="215/203")
            m_published = st.text_input("Set Published Date (YYYY-MM-DD)", value="2021-08-27")
            m_attack2 = st.checkbox("Card Has Attack 2", value=False)
            m_asking = st.number_input("Asking Price ($) [Optional]", min_value=0.0, value=650.0, step=5.0)

        submit_manual = st.form_submit_button("Predict Fair Market Value", type="primary")

    if submit_manual:
        manual_payload = {
            "name": m_name,
            "rarity": m_rarity,
            "printing": m_printing,
            "card_type": m_card_type,
            "stage": m_stage,
            "hp": m_hp,
            "retreat_cost": m_retreat,
            "number": m_number,
            "set_published": m_published,
            "has_attack2": m_attack2,
            "asking_price": m_asking if m_asking > 0 else None,
        }

        try:
            with st.spinner("Scoring card attributes..."):
                resp = requests.post(f"{api_url}/predict", json=manual_payload, timeout=30)
            if resp.status_code == 200:
                res = resp.json()
                st.success(f"### Estimated Fair Value: **\\${res['fair_value']:,.2f}**")
                if res.get("verdict"):
                    v = res["verdict"]
                    if v == "deal":
                        st.balloons()
                        st.success(f"🎉 **DEAL!** Asking price (\\${m_asking:,.2f}) is a bargain compared to \\${res['fair_value']:,.2f} fair value.")
                    elif v == "overpriced":
                        st.error(f"⚠️ **OVERPRICED!** Asking price (\\${m_asking:,.2f}) exceeds fair market value (\\${res['fair_value']:,.2f}).")
                    else:
                        st.info(f"⚖️ **FAIR!** Asking price (\\${m_asking:,.2f}) matches fair market expectations.")
            else:
                st.error(f"Prediction failed ({resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"Could not connect to API at {api_url}: {exc}")


# -----------------------------------------------------------------------------
# Tab 3: Model & Architecture
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("System Architecture & MLOps Pipeline")
    st.markdown(
        """
        This project implements an end-to-end, reproducible Machine Learning pipeline:

        1. **Data Ingestion & Versioning**:
           - Cleaned dataset snapshot generated from **TCGCSV** nightly mirror.
           - SHA-256 pinned and hosted in **Google Cloud Storage (GCS)**.
           - GroupShuffleSplit by `product_id` guarantees that all printings of a card stay strictly in the same split.

        2. **Model Selection**:
           - Four candidates are compared on the validation split: histogram gradient boosting,
             random forest, ridge regression, and a two-stage "hurdle" model (a classifier for
             cards worth \\$25 or more, plus separate regressors for cheap and expensive cards).
           - Gradient boosting won (validation RMSE 0.621 on log price vs 0.644 for the hurdle
             model), so it is the deployed model. Test-set R² on log price: 0.82.
           - Feature engineering lives inside the scikit-learn pipeline, so the API passes raw card fields.

        3. **Cloud Deployment**:
           - **Container**: one Docker image with scikit-learn, CPU-only PyTorch, ChromaDB and FastAPI.
           - **API Endpoint**: Hosted on **Google Cloud Run** with auto-scaling and public accessibility.
           - **Front-End**: Deployed on **Streamlit Community Cloud**.

        4. **Limitations & Ethical Considerations**:
           - The model estimates near-mint raw market value; physical grading (PSA/BGS/CGC) and condition are not captured.
           - High-value chase cards follow hype cycles that card attributes can't capture, so the model under-values them. Compare with the snapshot market price shown next to each search result.
           - Deal indicators should be viewed as algorithmic guidance, not financial advice.
        """
    )
