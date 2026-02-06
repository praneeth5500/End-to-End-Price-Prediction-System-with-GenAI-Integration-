import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os

# SHAP for model interpretability (waterfall plot)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

# PandasAI for generative AI querying in Market Insights (supports v2 and v3)
HAS_PANDASAI = False
SmartDataframe = None
_pandasai_llm_builder = None  # callable(api_key) -> (llm, use_global_config)
_llm_backend_error = None  # reason LLM backend not available

try:
    import pandasai as pai
    from pandasai import SmartDataframe
    HAS_PANDASAI = True
    # PandasAI v3: try pandasai-litellm (multiple possible import paths)
    for _ in range(1):
        try:
            from pandasai_litellm.litellm import LiteLLM
        except ImportError:
            try:
                from pandasai.llms import LiteLLM
            except ImportError:
                try:
                    from pandasai.llm import LiteLLM
                except ImportError:
                    _llm_backend_error = "No module named 'pandasai_litellm'. Run: pip install pandasai-litellm"
                    break
        def _build_llm(api_key):
            llm = LiteLLM(model="gpt-4o-mini", api_key=api_key)
            pai.config.set({"llm": llm})
            return (llm, True)  # use_global_config
        _pandasai_llm_builder = _build_llm
        _llm_backend_error = None
        break
    # PandasAI v2 fallback if v3 LiteLLM not available
    if _pandasai_llm_builder is None:
        try:
            from pandasai.llm import OpenAI as PandasAIOpenAI
            def _build_llm(api_key):
                return (PandasAIOpenAI(api_token=api_key), False)
            _pandasai_llm_builder = _build_llm
            _llm_backend_error = None
        except ImportError as e:
            _llm_backend_error = _llm_backend_error or str(e)
except ImportError as e:
    _pandasai_import_error = str(e)

# Placeholder: set your OpenAI API key here, or set env var OPENAI_API_KEY
OPENAI_API_KEY = ""  # e.g. "sk-..."

# Paths (run from project root; app.py in project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "used_cars.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "car_model.pkl")


@st.cache_data
def load_market_data():
    """Load and clean used cars dataset for Market Insights."""
    df = pd.read_csv(DATA_PATH)
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] > 0) & (df["price"] < 500000)].dropna(subset=["price", "brand"])
    return df


def answer_ask_fallback(df: pd.DataFrame, query: str):
    """
    Answer simple 'compare average price' questions using pandas when LLM is not available.
    Returns (success: bool, message or None, dataframe or None).
    """
    if "brand" not in df.columns or "price" not in df.columns:
        return False, None, None
    q = query.lower().strip()
    brand_series = df["brand"].dropna().astype(str).str.strip().str.lower()
    brands_in_data = set(brand_series.unique())
    # Find brand names mentioned in the query (e.g. "acura", "bmw")
    mentioned = [b for b in brands_in_data if b in q]
    # Also try original casing from dataset (Acura, BMW)
    for orig in df["brand"].dropna().unique():
        if orig and orig.lower() in q and orig.lower() not in mentioned:
            mentioned.append(orig.lower())
    if not mentioned:
        return False, None, None
    # Compute average price per mentioned brand
    results = []
    for brand_val in df["brand"].dropna().unique():
        if str(brand_val).strip().lower() in mentioned:
            subset = df[df["brand"].astype(str).str.strip().str.lower() == str(brand_val).strip().lower()]
            avg = subset["price"].mean()
            cnt = len(subset)
            results.append({"Brand": brand_val, "Average Price ($)": round(avg, 2), "Listings": cnt})
    if not results:
        return False, None, None
    result_df = pd.DataFrame(results).sort_values("Average Price ($)", ascending=False)
    msg = "**Average price comparison** (from dataset):\n\n"
    for _, row in result_df.iterrows():
        msg += f"- **{row['Brand']}**: ${row['Average Price ($)']:,.2f} ({int(row['Listings'])} listings)\n"
    return True, msg, result_df


def price_histogram_by_brand(df: pd.DataFrame, brand: str):
    """Filter by brand and return a Plotly histogram of price distribution."""
    subset = df.loc[df["brand"] == brand, "price"]
    if subset.empty:
        return None
    fig = px.histogram(
        x=subset,
        nbins=min(40, max(15, len(subset) // 20)),
        labels={"x": "Price ($)", "y": "Count"},
        title=f"Price distribution — {brand}",
    )
    fig.update_layout(
        xaxis_tickformat="$,.0f",
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=30),
    )
    return fig


# 1. Load the trained model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    _model_error = str(e)

# 2. Define the lists of options (Copied from your training data)
brands = ['Acura', 'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari', 'Ford', 'GMC', 'Genesis', 'Honda', 'Hummer', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Karma', 'Kia', 'Lamborghini', 'Land', 'Lexus', 'Lincoln', 'Lotus', 'Lucid', 'MINI', 'Maserati', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'Mercury', 'Mitsubishi', 'Nissan', 'Plymouth', 'Polestar', 'Pontiac', 'Porsche', 'RAM', 'Rivian', 'Rolls-Royce', 'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']

# Top 50 models (as defined in your training step)
models = ['M3 Base', 'F-150 XLT', 'Corvette Base', '1500 Laramie', 'Wrangler Sport', 'Camaro 2SS', 'Model Y Long Range', '911 Carrera', 'Mustang GT Premium', 'M4 Base', 'Explorer XLT', 'F-250 Lariat', 'F-150 Lariat', 'E-Class E 350 4MATIC', 'M5 Base', 'E-Class E 350', '911 Carrera S', 'F-250 XLT', 'R1S Adventure Package', 'Land Cruiser Base', 'Macan S', 'ES 350 Base', 'Wrangler Unlimited Sport', 'Model 3 Long Range', 'GX 460 Base', 'Mustang GT', 'Corvette Stingray w/2LT', '1500 Big Horn', 'Model Y Performance', 'X7 xDrive40i', 'Cooper S Base', 'X5 xDrive35i', 'Highlander XLE', 'H2 Base', '911 Carrera 4S', 'Suburban Premier', 'Wrangler Unlimited Sahara', 'SL-Class SL 550', 'RX 350 Base', 'G-Class G 550 4MATIC', 'Suburban LT', 'C-Class C 300 4MATIC', 'GT-R Premium', 'X6 M Base', 'Camaro 1SS', 'Rover Range Rover Sport HSE', 'Tahoe LT', 'LX 570 Three-Row', 'CLA-Class CLA 250', '2500 Big Horn', 'Other']

fuel_types = ['Gasoline', 'Diesel', 'E85 Flex Fuel', 'Hybrid', 'Plug-In Hybrid', 'not supported']
transmissions = ['A/T', '8-Speed A/T', 'Transmission w/Dual Shift Mode', '6-Speed A/T', '6-Speed M/T', 'Automatic', '7-Speed A/T', '8-Speed Automatic', '10-Speed A/T', '5-Speed A/T', '9-Speed Automatic']

# 3. App Title & Tabs
st.title("🚗 Used Car Price Valuation Tool")
tab_valuation, tab_insights = st.tabs(["Valuation Tool", "Market Insights"])

# ---------- Tab 1: Valuation Tool ----------
with tab_valuation:
    if model is None:
        st.error(f"Could not load the model. Place `car_model.pkl` in the **model** folder.")
        st.code(MODEL_PATH, language=None)
        try:
            st.caption(_model_error)
        except NameError:
            pass
        st.stop()

    st.markdown("Enter the car details below to check if it's a **Fair Price**, **Good Deal**, or **Overpriced**.")

    # 4. User Inputs (Layout with columns)
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brands)
        model_choice = st.selectbox("Model", models)
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2019)
        mileage = st.number_input("Mileage (miles)", min_value=0, value=50000)

    with col2:
        fuel = st.selectbox("Fuel Type", fuel_types)
        trans = st.selectbox("Transmission", transmissions)
        accident = st.radio("Has it been in an accident?", ["No", "Yes"])
        title = st.radio("Does it have a clean title?", ["Yes", "No"])

    # Listing Price (For Comparison)
    listing_price = st.number_input("Seller's Asking Price ($)", min_value=1000, value=25000)

    # 5. Prediction Logic
    if st.button("Analyze Deal"):
        # Preprocess inputs
        acc_val = 1 if accident == "Yes" else 0
        title_val = 1 if title == "Yes" else 0

        car_age = 2025 - year
        input_data = pd.DataFrame({
            "brand": [brand],
            "model_simplified": [model_choice],
            "car_age": [car_age],
            "milage": [mileage],
            "fuel_type": [fuel],
            "transmission": [trans],
            "accident": [acc_val],
            "clean_title": [title_val],
        })

        predicted_price = model.predict(input_data)[0]
        st.subheader(f"Estimated Market Value: ${predicted_price:,.2f}")

        diff = listing_price - predicted_price
        if diff < -1000:
            st.success(f" GREAT DEAL! You are saving ${abs(diff):,.2f} below market value.")
        elif diff > 1000:
            st.error(f"OVERPRICED. The seller is asking ${diff:,.2f} too much.")
        else:
            st.warning("FAIR PRICE. The listing is close to market value.")

        # SHAP interpretability: waterfall plot directly below the prediction result
        if HAS_SHAP:
            st.markdown("**How each feature pushed the price up or down (from baseline):**")
            # Use the same preprocessing as the pipeline so features match what the model sees
            preprocessor = model.named_steps["preprocessor"]
            regressor = model.named_steps["regressor"]
            X_trans = preprocessor.transform(input_data)  # single row, model-ready features
            
            # --- FIX: FORCE CAST TO FLOAT TO PREVENTS NUMPY OBJECT ERROR ---
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.array(X_trans, dtype=np.float64)
            # -------------------------------------------------------------

            feature_names = preprocessor.get_feature_names_out()  # names expected by the model

            # Initialize SHAP explainer for the trained Random Forest
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(X_trans)
            base_value = explainer.expected_value
            
            # Handle list output (common in older shap versions or classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            vals = np.asarray(shap_values[0]).flatten()
            data_row = np.asarray(X_trans[0]).flatten()

            explanation = shap.Explanation(
                values=vals,
                base_values=base_value,
                data=data_row,
                feature_names=feature_names.tolist(),
            )
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close()
        else:
            st.info("Install **shap** to see how each feature (Brand, Mileage, Year, etc.) pushed the price up or down: `pip install shap`")

# ---------- Tab 2: Market Insights ----------
with tab_insights:
    st.subheader("Market Insights")
    st.markdown("Select a **Car Brand** to see the price distribution, or **ask the data** a question in natural language.")

    try:
        df_market = load_market_data()
        brands_in_data = sorted(df_market["brand"].dropna().unique().tolist())
        brand_insight = st.selectbox("Car Brand", options=brands_in_data, key="insights_brand")

        fig = price_histogram_by_brand(df_market, brand_insight)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            n_cars = len(df_market[df_market["brand"] == brand_insight])
            st.caption(f"Based on {n_cars:,} listings for **{brand_insight}**.")
        else:
            st.info(f"No listings found for **{brand_insight}**.")

        # ----- Generative AI: Ask the Data -----
        st.divider()
        st.markdown("**Ask the Data** — ask questions in natural language (e.g. *Compare average price of Acura vs BMW*).")
        api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        ask_query = st.text_input(
            "Ask the Data",
            placeholder="e.g., Compare average price of Acura vs BMW",
            key="ask_the_data_input",
        )

        if ask_query.strip():
            # Try built-in fallback first for "compare average price" style questions (works without LLM)
            fallback_ok, fallback_msg, fallback_df = answer_ask_fallback(df_market, ask_query)

            if fallback_ok:
                st.markdown(fallback_msg)
                if fallback_df is not None:
                    st.dataframe(fallback_df, use_container_width=True, hide_index=True)
                st.caption("Tip: Install `pandasai-litellm` and set your OpenAI API key for more flexible natural-language questions.")

            elif not HAS_PANDASAI:
                st.warning(
                    "**PandasAI** is not available. Install it with: `pip install pandasai` — "
                    "then run the app with the **same** Python: `python -m streamlit run app.py`"
                )
                try:
                    st.caption(f"Import error: {_pandasai_import_error}")
                except NameError:
                    pass
            elif _pandasai_llm_builder is None:
                st.warning(
                    "**PandasAI** is installed but no **LLM backend** was found. "
                    "Install it in the **same Python** you use to run this app, then restart:"
                )
                st.code("pip install pandasai-litellm\npython -m streamlit run app.py", language="bash")
                if _llm_backend_error:
                    st.caption(f"Reason: {_llm_backend_error}")
            elif not api_key:
                st.warning("Set your **OpenAI API key** in the code (`OPENAI_API_KEY = \"sk-...\"`) or in the `OPENAI_API_KEY` environment variable.")
            else:
                with st.spinner("Querying the data..."):
                    try:
                        llm, use_global_config = _pandasai_llm_builder(api_key)
                        if use_global_config:
                            smart_df = SmartDataframe(df_market)
                        else:
                            smart_df = SmartDataframe(df_market, config={"llm": llm})
                        response = smart_df.chat(ask_query)
                        if response is None:
                            st.write("No response returned.")
                        elif isinstance(response, pd.DataFrame):
                            st.dataframe(response, use_container_width=True)
                        elif hasattr(response, "savefig"):
                            st.pyplot(response)
                            plt.close(response)
                        else:
                            st.write(response)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. Run the app from the project root.")
    except Exception as e:
        st.error(f"Could not load market data: {e}")