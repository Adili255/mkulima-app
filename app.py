import streamlit as st

st.set_page_config(
    page_title="Mkulima Consultation System",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #ffffff;
    }
    .header {
        background-color: #1E3F1E;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .description {
        background-color: #1A2E1A;
        color: #e0e0e0;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background-color: #1E1E1E;
        color: #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border-left: 4px solid #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #388E3C;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4CAF50 !important;
    }
    p, li, ol {
        color: #e0e0e0 !important;
    }
    [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #1E1E1E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

from utils.navigation import display_navigation

page = display_navigation()

with st.container():
    st.markdown("""
    <div class="header">
        <h1 style='text-align: center; margin: 0;'>Mkulima Consultation System</h1>
        <p style='text-align: center; margin: 0.5rem 0 0;'>Empowering Tanzania's Agricultural Sector</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Home":
    st.markdown("""
    <div class="description">
        <h3 style='margin-top: 0;'>About the System</h3>
        <p>Our platform helps farmers and agriculture traders analyze crop prices across Tanzania's markets from all regions. 
        Users can perform detailed price analysis across different markets, regions, and time periods to make informed 
        decisions about crop production and trading.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Market Analysis</h4>
            <p>Compare prices across different markets and regions in Tanzania</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Price Trends</h4>
            <p>Visualize historical price trends for various crops</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>Data Export</h4>
            <p>Download analysis results for offline use</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <h3>How to Use the System</h3>
        <ol>
            <li>Select <strong style='color: #4CAF50;'>Analysis</strong> to explore crop price data</li>
            <li>Use filters to narrow down by region, market, or crop</li>
            <li>View interactive charts and tables</li>
            <li>Download data for further analysis</li>
            <li>Use <strong style='color: #4CAF50;'>Prediction</strong> for future price forecasts</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif page == "Analysis":
    from all_pages.analysis import create_analysis_page
    create_analysis_page()

elif page == "Prediction":
    from all_pages.prediction import create_prediction_model
    create_prediction_model()

st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding: 1rem; color: #4CAF50;'>
    <p>© 2025 Mkulima Consultation System </p>
</div>
""", unsafe_allow_html=True)