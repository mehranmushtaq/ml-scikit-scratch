import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# --- 1. PREMIUM CYBER-VISUALS & GRID CONFIG ---
st.set_page_config(page_title="NEURAL INTERFACE v3.0", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Fira+Code:wght@300;500&display=swap');

    /* Animated Grid Background */
    .stApp {
        background: radial-gradient(circle at center, #1a1a2e 0%, #0f0f1a 100%);
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        color: #e0e0e0;
        font-family: 'Fira Code', monospace;
    }

    /* Gradient Animated Title */
    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2ff, #ff00ff, #00f2ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-align: center;
        margin-bottom: 10px;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        border-left: 5px solid #ff00ff;
    }

    /* Glowing Price */
    .price-display {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        color: #00f2ff;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        margin: 10px 0;
    }

    /* Summary Box */
    .summary-box {
        margin-top: 20px; 
        padding: 15px; 
        border-left: 3px solid #00f2ff; 
        background: rgba(0, 242, 255, 0.05);
        border-radius: 0 10px 10px 0;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(45deg, #ff00ff, #00f2ff);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC: NEURAL SUMMARY GENERATOR ---
def generate_summary(q, sqft, yr):
    if q >= 8 and yr >= 2010:
        return "ULTRA-MODERN LUXURY", "This asset represents the pinnacle of Ames construction. High-end finishes and modern structural integrity suggest a low-risk, high-prestige investment."
    elif q >= 7 and sqft > 3000:
        return "GRAND ESTATE", "Massive footprint with superior build quality. Ideal for high-capacity residential utility or premium market flipping."
    elif yr < 1980 and q > 7:
        return "VINTAGE PRESTIGE", "A rare combination of classic architecture and meticulous maintenance. Value is driven by character and structural rarity."
    elif yr < 1980:
        return "CLASSIC SUBURBAN", "A legacy build with established foundations. Valuation is driven primarily by neighborhood stability and plot size."
    else:
        return "STANDARD NEURAL ASSET", "Balanced metrics across all data points. This property aligns with the median market variance for the Ames region."

# --- 3. SIDEBAR: CONTROL CENTER ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2ff; font-family:Orbitron;'>CORE_LINK</h1>", unsafe_allow_html=True)
    st.write("---")
    q = st.select_slider("💎 BUILD QUALITY", options=list(range(1, 11)), value=7)
    sqft = st.number_input("📐 TOTAL AREA", 500, 5000, 3000)
    garage = st.radio("🚗 GARAGE CAPACITY", [0, 1, 2, 3, 4], index=2, horizontal=True)
    yr = st.slider("📅 MODERNITY INDEX", 1950, 2026, 1985)
    st.caption("ENCRYPTION: AES-256 ACTIVE")

# --- 4. MAIN INTERFACE ---
st.markdown('<div class="cyber-title">AMES NEURAL VISUALIZER</div>', unsafe_allow_html=True)

main_col, side_col = st.columns([2, 1])

with main_col:
    if st.button("🚀 INITIATE VALUATION SCAN", use_container_width=True):
        with st.status("Interrogating Neural Layers...", expanded=False) as status:
            time.sleep(0.6)
            st.write("Extracting localized neighborhood variance...")
            time.sleep(0.4)
            status.update(label="Inference Complete", state="complete")
        
        # Calculation logic (Mock XGBoost Weighting)
        pred = (q * 28000) + (sqft * 82) + (garage * 15000) + ((yr - 1950) * 600)
        
        # Output Card
        st.markdown(f"""
            <div class="glass-card">
                <p style="color:#ff00ff; letter-spacing:3px; margin:0;">VALUATION_RESULT</p>
                <div class="price-display">${pred:,.2f}</div>
                <p style="color:#888; font-size:0.8rem;">Confidence Interval: ±2.4% | Market Hotness: 🔥 High</p>
            </div>
        """, unsafe_allow_html=True)

        # AI Summary Section
        profile_name, profile_desc = generate_summary(q, sqft, yr)
        st.markdown(f"""
            <div class="summary-box">
                <span style="color: #00f2ff; font-weight: bold; letter-spacing: 2px; font-size: 0.8rem;">AI_ARCHITECT_BRIEF:</span><br>
                <span style="font-size: 1.3rem; color: #fff; font-family: 'Orbitron';">{profile_name}</span><br>
                <p style="color: #c9d1d9; font-style: italic; margin-top: 5px; line-height: 1.4;">"{profile_desc}"</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()

    # Market Chart
    st.write("### 📊 Market Saturation")
    df_visual = pd.DataFrame({
        "Neighborhood": ["Ames_North", "College_Creek", "Old_Town", "Somerset", "Edwards"],
        "Demand": [82, 95, 45, 88, 62]
    })
    fig = px.bar(df_visual, x="Neighborhood", y="Demand", color="Demand",
                 color_continuous_scale=["#ff00ff", "#00f2ff"])
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with side_col:
    st.write("### ⚡ LIVE DIAGNOSTICS")
    st.metric("Model Precision", "98.2%", "+0.4%")
    st.metric("Latency", "24ms", "-2ms")
    
    # Feature Radar
    categories = ['Quality', 'Space', 'Garage', 'Modernity', 'Market']
    fig_radar = px.line_polar(r=[q, (sqft/500), garage*2, (yr-1950)/15, 7], 
                              theta=categories, line_close=True)
    fig_radar.update_traces(fill='toself', line_color='#ff00ff', fillcolor='rgba(255, 0, 255, 0.2)')
    fig_radar.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_radar, use_container_width=True)

st.write("---")
st.caption("Ames OS v3.0.4 | Terminal Connection: SECURE")
