import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HomeVal · House Price Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e1;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(212,175,55,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(139,90,43,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

h1, h2, h3 { font-family: 'Playfair Display', serif; }

.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0 20px;
    border-bottom: 1px solid rgba(212,175,55,0.2);
    margin-bottom: 40px;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 900;
    color: #e8e6e1;
    letter-spacing: -0.5px;
}
.nav-logo span { color: #d4af37; }
.nav-badge {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #d4af37;
    border: 1px solid rgba(212,175,55,0.4);
    padding: 4px 12px;
    border-radius: 2px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -2px;
    margin-bottom: 12px;
}
.hero-accent { color: #d4af37; }
.hero-sub {
    font-size: 14px;
    color: #6b6860;
    margin-bottom: 40px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #d4af37;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    margin-top: 28px;
}

.result-card {
    background: linear-gradient(135deg, rgba(212,175,55,0.08), rgba(212,175,55,0.02));
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 4px;
    padding: 32px;
    text-align: center;
    margin-bottom: 20px;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    color: #d4af37;
    letter-spacing: -2px;
    margin: 8px 0;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #6b6860;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.metric-card {
    background: #13161d;
    border: 1px solid #1e2330;
    border-radius: 4px;
    padding: 16px 20px;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #d4af37;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #4a4840;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.insight-row {
    background: #13161d;
    border: 1px solid #1e2330;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #d4af37, #b8962e) !important;
    border: none !important;
    border-radius: 3px !important;
    color: #0d0f14 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 16px !important;
}

.range-bar-bg {
    background: #1e2330;
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# NAV
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">Home<span>Val</span></div>
  <div class="nav-badge">XGBoost · Ames Housing · R²=0.9291</div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero-title">
  Predict Your<br>
  <span class="hero-accent">Home's Value</span>
</div>
<div class="hero-sub">
  ML-POWERED · 1,460 HOMES · XGBOOST + GRIDSEARCHCV · LASSO BASELINE
</div>
""", unsafe_allow_html=True)

# LOAD MODEL
# Make sure joblib is imported at the top of your file, or inside the function:
import joblib

@st.cache_resource
def load_model():
    try:
        model_path = "Projects/house_prices_prediction/ames_housing_xgb_model.pkl"
        features_path = "Projects/house_prices_prediction/model_features.pkl"
        
        # ACTUALLY LOAD THE FILES HERE:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        
        return model, features
if model is None:
    st.markdown("""
    <div style="background:rgba(212,175,55,0.08);border:1px solid rgba(212,175,55,0.3);
         border-radius:4px;padding:20px;margin-bottom:24px;">
      <div style="font-family:DM Mono,monospace;font-size:11px;color:#d4af37;
           letter-spacing:2px;margin-bottom:8px;">⚠ MODEL FILES NOT FOUND</div>
      <div style="font-size:13px;color:#6b6860;">
        Place <code style="color:#d4af37;">ames_housing_xgb_model.pkl</code> and
        <code style="color:#d4af37;">model_features.pkl</code> in the same folder as this app.
      </div>
    </div>
    """, unsafe_allow_html=True)

col_form, col_result = st.columns([1.2, 0.8], gap="large")

with col_form:
    st.markdown('<div class="section-label">// Core Property Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    overall_qual = c1.selectbox(
        "Overall Quality (1-10)", 
        options=list(range(1, 11)), 
        index=6,
        format_func=lambda x: f"{x} — {'Poor' if x <= 3 else 'Fair' if x <= 5 else 'Good' if x <= 7 else 'Excellent'}"
    )
    overall_cond = c2.selectbox("Overall Condition (1-10)", options=list(range(1, 11)), index=4)
    c3, c4 = st.columns(2)
    year_built = c3.number_input("Year Built", 1872, 2010, 2000)
    year_remod = c4.number_input("Year Remodeled", 1950, 2010, 2005)

    st.markdown('<div class="section-label">// Size & Space</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    gr_liv_area = c5.number_input("Living Area (sq ft)", 300, 6000, 1500, 50)
    total_bsmt_sf = c6.number_input("Basement Area (sq ft)", 0, 3000, 800, 50)
    c7, c8 = st.columns(2)
    first_flr_sf = c7.number_input("1st Floor (sq ft)", 300, 4000, 1000, 50)
    lot_area = c8.number_input("Lot Area (sq ft)", 1000, 200000, 9000, 500)

    st.markdown('<div class="section-label">// Rooms & Bathrooms</div>', unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)
    full_bath = c9.number_input("Full Bath", 0, 4, 2)
    half_bath = c10.number_input("Half Bath", 0, 3, 0)
    tot_rms = c11.number_input("Total Rooms", 2, 14, 7)
    bedrooms = c12.number_input("Bedrooms", 0, 8, 3)

    st.markdown('<div class="section-label">// Garage</div>', unsafe_allow_html=True)
    c13, c14 = st.columns(2)
    garage_cars = c13.number_input("Garage Cars", 0, 4, 2)
    garage_area = c14.number_input("Garage Area (sq ft)", 0, 1500, 480, 20)

    st.markdown('<div class="section-label">// Property Extras</div>', unsafe_allow_html=True)
    c15, c16 = st.columns(2)
    fireplaces = c15.number_input("Fireplaces", 0, 4, 1)
    mas_vnr_area = c16.number_input("Masonry Veneer (sq ft)", 0, 1600, 100, 10)
    c17, c18 = st.columns(2)
    wood_deck_sf = c17.number_input("Wood Deck (sq ft)", 0, 900, 0, 10)
    open_porch_sf = c18.number_input("Open Porch (sq ft)", 0, 600, 0, 10)

    st.markdown('<div class="section-label">// Location & Zoning</div>', unsafe_allow_html=True)
    c19, c20 = st.columns(2)
    ms_zoning = c19.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0)
    lot_shape = c20.selectbox("Lot Shape", ["Reg", "IR1", "IR2", "IR3"], index=0)
    c21, c22 = st.columns(2)
    lot_frontage = c21.number_input("Lot Frontage (ft)", 0, 300, 70, 5)
    paved_drive = c22.selectbox("Paved Drive", ["Y", "P", "N"], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🏡  ESTIMATE PROPERTY VALUE")

with col_result:
    if predict_btn and model is not None:
        input_data = {col: 0 for col in feature_cols}

        numeric_map = {
            "Overall Qual": overall_qual,
            "Overall Cond": overall_cond,
            "Year Built": year_built,
            "Year Remod/Add": year_remod,
            "Gr Liv Area": gr_liv_area,
            "Total Bsmt SF": total_bsmt_sf,
            "1st Flr SF": first_flr_sf,
            "Lot Area": lot_area,
            "Full Bath": full_bath,
            "Half Bath": half_bath,
            "TotRms AbvGrd": tot_rms,
            "Bedroom AbvGr": bedrooms,
            "Garage Cars": garage_cars,
            "Garage Area": garage_area,
            "Fireplaces": fireplaces,
            "Mas Vnr Area": mas_vnr_area,
            "Wood Deck SF": wood_deck_sf,
            "Open Porch SF": open_porch_sf,
            "Lot Frontage": lot_frontage,
            "Bsmt Full Bath": 1,
            "Bsmt Half Bath": 0,
            "Kitchen AbvGr": 1,
            "Garage Yr Blt": year_built,
            "BsmtFin SF 1": int(total_bsmt_sf * 0.6),
            "BsmtFin SF 2": 0,
            "Bsmt Unf SF": int(total_bsmt_sf * 0.4),
            "2nd Flr SF": max(0, gr_liv_area - first_flr_sf),
            "Low Qual Fin SF": 0,
        }
        for k, v in numeric_map.items():
            if k in input_data:
                input_data[k] = v

        for col in [f"MS Zoning_{ms_zoning}", f"Lot Shape_{lot_shape}", f"Paved Drive_{paved_drive}"]:
            if col in input_data:
                input_data[col] = 1

        for col in ["Garage Type_Attchd", "Kitchen Qual_TA", "Bsmt Qual_Gd",
                    "Exter Qual_TA", "Foundation_PConc", "Heating QC_Ex"]:
            if col in input_data:
                input_data[col] = 1

        input_df = pd.DataFrame([input_data])[feature_cols]
        predicted = model.predict(input_df)[0]
        pred_k = predicted / 1000
        low = predicted * 0.92
        high = predicted * 1.08

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Estimated Market Value</div>
          <div class="result-price">${pred_k:,.0f}K</div>
          <div style="font-family:DM Mono,monospace;font-size:11px;color:#4a4840;margin-top:8px;">
            Confidence Range · ${low/1000:,.0f}K — ${high/1000:,.0f}K
          </div>
        </div>
        """, unsafe_allow_html=True)

        pct = min(max((predicted - 50000) / (750000 - 50000), 0), 1)
        st.markdown(f"""
        <div style="margin-bottom:24px;">
          <div style="display:flex;justify-content:space-between;
               font-family:DM Mono,monospace;font-size:10px;color:#4a4840;margin-bottom:4px;">
            <span>$50K</span><span>Market Range</span><span>$750K+</span>
          </div>
          <div class="range-bar-bg">
            <div style="width:{pct*100:.1f}%;height:100%;
                 background:linear-gradient(90deg,#8b5a2b,#d4af37);border-radius:2px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        price_per_sqft = predicted / max(gr_liv_area, 1)
        m1.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">${price_per_sqft:,.0f}</div>
          <div class="metric-lbl">Per Sq Ft</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{overall_qual}/10</div>
          <div class="metric-lbl">Quality Score</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">// Key Value Drivers</div>', unsafe_allow_html=True)

        age_note = 2024 - year_built
        size_note = "Above avg" if gr_liv_area > 1500 else "Below avg"

        insights = [
            ("Overall Quality", f"{overall_qual}/10", "#d4af37" if overall_qual >= 7 else "#8b5a2b"),
            ("Living Area", f"{gr_liv_area:,} sqft · {size_note}", "#d4af37" if gr_liv_area > 1500 else "#8b5a2b"),
            ("Property Age", f"{age_note} years old", "#d4af37" if age_note < 30 else "#8b5a2b"),
            ("Garage Capacity", f"{garage_cars} car{'s' if garage_cars != 1 else ''}", "#d4af37" if garage_cars >= 2 else "#8b5a2b"),
            ("Basement", f"{total_bsmt_sf:,} sq ft", "#d4af37" if total_bsmt_sf > 600 else "#8b5a2b"),
        ]
        for lbl, val, clr in insights:
            st.markdown(f"""
            <div class="insight-row">
              <span style="font-size:12px;color:#6b6860;">{lbl}</span>
              <span style="font-family:DM Mono,monospace;font-size:12px;color:{clr};">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <br>
        <div style="background:#13161d;border:1px solid #1e2330;border-radius:4px;padding:16px 20px;margin-top:8px;">
          <div style="font-family:DM Mono,monospace;font-size:10px;color:#d4af37;
               letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">// Model Info</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#6b6860;">Algorithm</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:#e8e6e1;">XGBoost</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#6b6860;">R² Score</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:#d4af37;">0.9291</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
            <span style="font-size:12px;color:#6b6860;">MAE</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:#d4af37;">$13,317</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:12px;color:#6b6860;">vs Lasso</span>
            <span style="font-family:DM Mono,monospace;font-size:12px;color:#d4af37;">↑ $1,716 better</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif predict_btn and model is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:48px;margin-bottom:16px;">⚠️</div>
          <div style="font-family:Playfair Display,serif;font-size:20px;color:#e8e6e1;margin-bottom:8px;">
            Model files missing
          </div>
          <div style="font-size:13px;color:#4a4840;">Add .pkl files to run predictions</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:70px 20px;">
          <div style="font-size:60px;margin-bottom:20px;">🏡</div>
          <div style="font-family:Playfair Display,serif;font-size:22px;
               font-weight:700;color:#e8e6e1;margin-bottom:10px;">
            Configure your property
          </div>
          <div style="font-size:13px;line-height:1.8;color:#4a4840;">
            Fill in the details on the left<br>
            and click <strong style="color:#d4af37;">Estimate Property Value</strong>
          </div>
        </div>""", unsafe_allow_html=True)

# MODEL STATS
st.markdown("<hr style='border:none;border-top:1px solid #1e2330;margin:32px 0;'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">// Model Performance Dashboard</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
for col, lbl, val in [
    (s1, "XGBoost R²", "0.9291"),
    (s2, "XGBoost MAE", "$13,317"),
    (s3, "Lasso R²", "0.9192"),
    (s4, "Features", "261"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="font-size:1.5rem;">{val}</div>
      <div class="metric-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">// Top Predictors (XGBoost Feature Importance)</div>', unsafe_allow_html=True)

features_imp = {
    "Overall Qual": 0.213,
    "Garage Cars": 0.148,
    "Gr Liv Area": 0.112,
    "Bsmt Qual_Gd": 0.089,
    "Full Bath": 0.071,
    "Total Bsmt SF": 0.068,
    "Kitchen Qual_TA": 0.055,
    "MS Zoning_RM": 0.048,
    "Garage Type_Attchd": 0.041,
    "Fireplaces": 0.038,
}
for feat, imp in features_imp.items():
    bar_w = imp / 0.213 * 100
    st.markdown(f"""
    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
        <span style="font-size:12px;color:#e8e6e1;">{feat}</span>
        <span style="font-family:DM Mono,monospace;font-size:11px;color:#d4af37;">{imp:.3f}</span>
      </div>
      <div class="range-bar-bg">
        <div style="width:{bar_w:.1f}%;height:100%;
             background:linear-gradient(90deg,#8b5a2b,#d4af37);border-radius:2px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div style="border-top:1px solid #1e2330;padding-top:20px;margin-top:40px;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
  <span style="font-family:DM Mono,monospace;font-size:11px;color:#4a4840;">
    HomeVal · XGBoost · GridSearchCV · Ames Housing · 1,460 samples · 261 features
  </span>
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
    <span style="font-family:Playfair Display,serif;font-size:13px;color:#6b6860;">
      Built by <strong style="color:#d4af37;">Mehran Mushtaq</strong>
    </span>
    <a href="https://github.com/mehranmushtaq" target="_blank"
       style="font-family:DM Mono,monospace;font-size:11px;color:#d4af37;
              text-decoration:none;border:1px solid rgba(212,175,55,0.3);
              padding:4px 12px;border-radius:2px;">
      ⌥ github.com/mehranmushtaq
    </a>
    <span style="font-family:DM Mono,monospace;font-size:11px;color:#4a4840;">📍 Kashmir, India</span>
  </div>
</div>
""", unsafe_allow_html=True)
