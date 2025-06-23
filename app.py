import streamlit as st

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #bbf7d0;
        color: #121212;
    }
    .header {
        background-color: #ffffff;
        color: #121212;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .description {
        background-color: #ffffff;
        color: #121212;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background-color: #ffffff;
        color: #121212;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border-left: 4px solid #4CAF50;
    }
    
    /* General button style */
    .stButton>button {
        border: 1px solid #4CAF50;
        color: #4CAF50;
        background-color: white;
    }
    .stButton>button:hover {
        border-color: #388E3C;
        color: #388E3C;
    }
    
    /* Specific style for prediction button */
    div[data-testid="stButton"] > button[kind="secondary"][title="Click to make a prediction"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        width: 100%;
    }
    div[data-testid="stButton"] > button[kind="secondary"][title="Click to make a prediction"]:hover {
        background-color: #45a049 !important;
        color: white !important;
    }
    
    /* Blue prediction result text */
    .prediction-result {
        color: #2196F3 !important;
        font-size: 18px;
        font-weight: bold;
    }
    
    /* Red disclaimer/NOTE text */
    .warning-text {
        color: #F44336 !important;
        font-weight: bold;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1E3F1E !important;
    }
    p, li, ol {
        color: #121212 !important;
    }
    
    /* Dropdown Navigation Styles */
    div[data-baseweb="select"] {
        background-color: white !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    div[role="listbox"] {
        background-color: white !important;
        color: black !important;
        border-radius: 0 0 8px 8px !important;
        margin-top: 4px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        border: 1px solid #e0e0e0 !important;
    }
    div[role="listbox"] div {
        background-color: white !important;
        color: black !important;
    }
    div[role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: #1E3F1E !important;
        border-radius: 4px !important;
    }
    div[role="option"][aria-selected="true"] {
        background-color: #22c55e !important;
        color: white !important;
        border-radius: 4px !important;
    }
    
    /* List Styles */
    ol, ul {
        background-color: white !important;
        padding: 1rem 2rem !important;
        border-radius: 0.5rem;
    }
    li {
        background-color: white !important;
        padding: 0.25rem 0 !important;
    }
    
    /* Input Fields */
    [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #121212 !important;
    }
    
    /* Centered contact info */
    .centered-contact {
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

from utils.navigation import display_navigation

page = display_navigation()

# Only show the header on the Home page
if page == "Home":
    with st.container():
        st.markdown("""
        <div class="header">
            <h1 style='margin-top: 0;'>Mkulima Consultation System</h1>
            <p style='margin: 0.5rem 0 0;'>Empowering Tanzania's Agricultural Sector</p>
        </div>
        """, unsafe_allow_html=True)

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
            <li>Select <strong style='color: #1E3F1E;'>Analysis</strong> to explore crop price data</li>
            <li>Use filters to narrow down by region, market, or crop</li>
            <li>View interactive charts and tables</li>
            <li>Download data for further analysis</li>
            <li>Use <strong style='color: #1E3F1E;'>Prediction</strong> for future price forecasts</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif page == "Analysis":
    from all_pages.analysis import create_analysis_page
    create_analysis_page()

elif page == "Prediction":
    from all_pages.prediction import create_prediction_model
    create_prediction_model()
