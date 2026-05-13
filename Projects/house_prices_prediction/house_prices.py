import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="Ames Housing AI", page_icon="🏠", layout="wide")


st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 12px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(14, 17, 23, 0.85);
        color: #9ca3af;
        text-align: center;
        padding: 15px;
        font-size: 14px;
        border-top: 1px solid #374151;
        backdrop-filter: blur(5px);
        z-index: 100;
    }
    .name-highlight {
        color: #00d4ff;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        
        return joblib.load('ames_xgb_model.pkl'), False
    except:
        
        return None, True

model, is_sim = load_model()


with st.sidebar:
    st.title("🛠️ Home Editor")
    st.write("Adjust property details:")
    
    quality = st.slider("Material Quality", 1, 10, 8)
    year_built = st.number_input("Year Built", 1870, 2026, 2002)
    living_area = st.number_input("Living Area (sq ft)", 500, 5000, 3067)
    basement = st.number_input("Basement Area (sq ft)", 0, 3000, 2000)
    garage_cars = st.radio("Garage Capacity (Cars)", [0, 1, 2, 3, 4], index=2, horizontal=True)
    bathrooms = st.slider("Full Bathrooms", 1, 4, 3)
    
    st.divider()
    predict_btn = st.button("🚀 Predict House Value", use_container_width=True)


st.title("🏠 Ames Housing DNA & Valuation")

if predict_btn:
    with st.spinner('AI Engine Analyzing Market Trends...'):
        time.sleep(0.8) 
        
        
        features = pd.DataFrame({
            'OverallQual': [quality], 'GrLivArea': [living_area], 
            'TotalBsmtSF': [basement], 'GarageCars': [garage_cars], 
            'YearBuilt': [year_built], 'FullBath': [bathrooms]
        })

   
        if not is_sim:
            price = model.predict(features)[0]
        else:
          
            price = (quality * 22000) + (living_area * 60) + (year_built * 250) - 480000

        col1, col2 = st.columns([1, 1])

        with col1:
           
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = price,
                title = {'text': "Market Estimate ($)", 'font': {'size': 20, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 500000], 'tickcolor': "white"},
                    'bar': {'color': "#ff4b4b"},
                    'bgcolor': "#1f2937",
                    'borderwidth': 2,
                    'bordercolor': "#374151"
                }
            ))
            fig_gauge.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            
            st.subheader("Property Signature")
            categories = ['Quality', 'Space', 'Modernity', 'Garage', 'Baths']
           
            values = [quality/10, living_area/4000, (year_built-1870)/156, garage_cars/4, bathrooms/4]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values, theta=categories, fill='toself', line_color='#ff4b4b', fillcolor='rgba(255, 75, 75, 0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor="#0e1117"),
                showlegend=False, height=380, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}
            )
            st.plotly_chart(fig_radar, use_container_width=True)

      
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price / SqFt", f"${price/max(1, living_area):.2f}")
        c2.metric("Est. Taxes", f"${price*0.012:.0f}")
        c3.metric("Mortgage (Est.)", f"${(price*0.006):,.0f}")
        c4.metric("Age of Home", f"{2026-year_built} Yrs")

else:
  
    st.info("👈 Adjust property details in the **Home Editor** and click **Predict** to see results.")
    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=1200&q=80", caption="Ames Housing Market Analysis Tool")


st.markdown(f"""
    <div class="footer">
        Developed by <span class="name-highlight">Mehran Mushtaq</span> | 
        <span style="color: #6b7280;">B.Tech CSE | Machine Learning Project</span>
    </div>
    """, unsafe_allow_html=True)
