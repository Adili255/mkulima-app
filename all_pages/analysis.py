import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_analysis_page():
    st.markdown("""
    <style>
        /* Main styling improvements */
        body {
            color: #333333;
        }
        
        /* Sidebar styling - Compact design */
        [data-testid=stSidebar] {
            background-color: #E8F5E9 !important;
            padding: 0.5rem 0.75rem !important;
            min-width: 220px !important;
            max-width: 280px !important;
        }
        
        /* Sidebar widget styling - changed background to #bbf7d0 */
        .stMultiSelect, .stSelectbox {
            background-color: #bbf7d0 !important;
            border-radius: 5px;
            padding: 6px;
            margin-bottom: 8px;
            border: 1px solid #C8E6C9;
        }
        
        /* Sidebar labels - smaller font */
        .stMultiSelect label, .stSelectbox label {
            color: #1B5E20 !important;
            font-weight: bold !important;
            font-size: 13px !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Dropdown menu styling */
        .stMultiSelect [data-baseweb=select] > div:first-child {
            background-color: white !important;
            color: #333333 !important;
            border-radius: 4px;
            padding: 0.25rem 0.5rem !important;
        }
        
        /* Selected items in multiselect */
        .stMultiSelect [data-baseweb=tag] {
            background-color: #f0fdf4 !important;
            color: #166534 !important;
            border: 1px solid #166534 !important;
            padding: 0.15rem 0.4rem !important;
            margin: 0.1rem !important;
            font-size: 12px !important;
        }
        
        /* Dropdown options */
        [role="listbox"] div {
            color: #333333 !important;
            padding: 0.25rem 0.5rem !important;
            font-size: 13px !important;
        }
        
        /* Sidebar header - more compact */
        .sidebar-header {
            background-color: #C8E6C9;
            padding: 8px 10px;
            border-radius: 5px; 
            border-left: 5px solid #2E7D32;
            margin-bottom: 12px;
        }
        
        .sidebar-header h3 {
            font-size: 16px !important;
            margin: 0 !important;
        }
        
        /* Expander styling - changed button colors */
        .compact-expander {
            border: 1px solid #4CAF50 !important;
            border-radius: 5px !important;
            margin: 5px 0 !important;
            background-color: white !important;
        }
        
        .compact-expander .st-emotion-cache-1hynsf2 {
            padding: 0.5rem 1rem !important;
            background-color: white !important;
            color: #1B5E20 !important;
            font-weight: bold !important;
        }
        
        .compact-data {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 10px;
            background-color: white;
        }
        
        /* Chart containers - improved fitting */
        .stPlotlyChart {
            background-color: black;
            border-radius: 8px;
            padding: 10px;
            margin: 0 auto;
            width: 100% !important;
        }
        
        /* Section header styling - unified design */
        .section-header {
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 5px solid #2E7D32;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .section-header h2 {
            text-align: center;
            color: #1B5E20 !important;
            margin: 0;
        }
        
        /* Text elements */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #1B5E20 !important;
        }
        
        .stMarkdown p {
            color: #333333 !important;
        }
        
        /* Button styling */
        .stDownloadButton button {
            background-color: #2E7D32 !important;
            color: white !important;
            border: none !important;
        }
        
        .stDownloadButton button:hover {
            background-color: #1B5E20 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main header with unified design
    st.markdown("""
    <div class="section-header">
        <h1 >Crop Price Analysis Dashboard</h1>
        <p style='text-align: center; margin: 0;'>Analyze crop price trends across Tanzania's regions and markets</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_csv("filtered_data.csv", encoding="ISO-8859-1")
        df["date"] = pd.to_datetime(df["date"])
    except FileNotFoundError:
        st.error("Dataset not found at: filtered_data.csv")
        st.stop()

    # Compact sidebar with light green background
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3>Analysis Filters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter widgets with #bbf7d0 background
    region = st.sidebar.multiselect("Select region", df["region"].unique())
    district = st.sidebar.multiselect("Select district", df[df["region"].isin(region)]["district"].unique()) if region else st.sidebar.multiselect("Select district", df["district"].unique())
    market = st.sidebar.multiselect("Select market", df[df["district"].isin(district)]["market"].unique()) if district else st.sidebar.multiselect("Select market", df["market"].unique())
    commodity = st.sidebar.multiselect("Select commodity", df[df["market"].isin(market)]["commodity"].unique()) if market else st.sidebar.multiselect("Select commodity", df["commodity"].unique())

    filtered_df = df.copy()
    if region: filtered_df = filtered_df[filtered_df["region"].isin(region)]
    if district: filtered_df = filtered_df[filtered_df["district"].isin(district)]
    if market: filtered_df = filtered_df[filtered_df["market"].isin(market)]
    if commodity: filtered_df = filtered_df[filtered_df["commodity"].isin(commodity)]

    current_year = pd.to_datetime("today").year
    filtered_df["year"] = pd.to_datetime(filtered_df["date"]).dt.year
    display_df = filtered_df[filtered_df["year"] >= (current_year - 5)]

    # Regional Market Price Analysis section
    st.markdown("""
    <div class="section-header">
        <h2>Regional Market Price Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #333333 !important;'>This section shows average prices by region for selected crops. Compare how different regions perform for each commodity.</p>", unsafe_allow_html=True)
    
    crops_to_analyze = commodity if commodity else display_df["commodity"].unique().tolist()
    region_crop_avg = display_df.groupby(["region", "commodity"])["TSh"].mean().reset_index().round(2)
    
    for crop in crops_to_analyze:
        crop_data = region_crop_avg[region_crop_avg["commodity"] == crop].sort_values("TSh", ascending=False)
        if not crop_data.empty:
            fig = px.bar(crop_data, x="region", y="TSh", title=f"Average {crop} Prices by Region",
                        labels={"TSh": "Average Price (TSh)", "region": "Region"}, 
                        template="plotly_dark",
                        color="TSh", color_continuous_scale=["#C8E6C9", "#81C784", "#4CAF50", "#2E7D32", "#1B5E20"])
            fig.update_layout(
                xaxis_tickangle=-45, 
                coloraxis_showscale=False, 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander(f"View data for {crop}", expanded=False):
                st.markdown('<div class="compact-data">', unsafe_allow_html=True)
                st.dataframe(crop_data, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                csv = crop_data.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_regional_prices.csv", mime='text/csv')

    filtered_df["month_year"] = pd.to_datetime(filtered_df["date"]).dt.to_period('M').astype(str)
    
    # Time Series Analysis section
    st.markdown("""
    <div class="section-header">
        <h2>Time Series Analysis by Crop</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #333333 !important;'>Track monthly price movements for selected crops. The line charts show how prices have changed over time.</p>", unsafe_allow_html=True)
    
    selected_crops = filtered_df["commodity"].unique().tolist() if not commodity else [c for c in ["Beans", "Maize", "Rice"] if c in commodity]
    
    for crop in selected_crops:
        crop_df = filtered_df[filtered_df["commodity"] == crop]
        linechart = crop_df.groupby("month_year")["TSh"].mean().reset_index()
        linechart["month_year"] = pd.to_datetime(linechart["month_year"]).dt.strftime("%Y : %b")
        fig = px.line(linechart, x="month_year", y="TSh", 
                     labels={"TSh": "Average Price (TSh)", "month_year": "Month-Year"},
                     title=f"{crop} Prices Over Time", height=400, 
                     template="plotly_dark")
        fig.update_traces(
            line_color="#4CAF50", 
            line_width=3, 
            mode="lines+markers", 
            marker=dict(color="#2E7D32")
        )
        fig.update_layout(
            font=dict(color='white'),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(linechart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = linechart.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_time_series.csv", mime='text/csv')

    # Rolling Average section with fixed x-axis label
    st.markdown("""
    <div class="section-header">
        <h2>3-Month Rolling Average Price Trends</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #333333 !important;'>The rolling average smooths out short-term fluctuations to show longer-term trends. Blue line shows actual prices while orange shows the smoothed trend.</p>", unsafe_allow_html=True)
    
    for crop in selected_crops:
        crop_df = filtered_df[filtered_df["commodity"] == crop]
        linechart = crop_df.groupby("month_year")["TSh"].mean().reset_index()
        linechart["month_year"] = pd.to_datetime(linechart["month_year"])
        linechart["Rolling_Avg"] = linechart["TSh"].rolling(window=3).mean()
        fig = px.line(linechart, x=linechart["month_year"].dt.strftime("%Y : %b"), y=["TSh", "Rolling_Avg"],
                     labels={"value": "Price (TSh)", "variable": "Metric", "x": "Month-Year"}, 
                     title=f"{crop} Price Trends", height=400,
                     template="plotly_dark")
        fig.update_traces(line_color="#4CAF50", selector=dict(name="TSh"))
        fig.update_traces(line_color="#FFA500", selector=dict(name="Rolling_Avg"))
        fig.update_layout(
            font=dict(color='white'),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Month-Year"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(linechart[["month_year", "TSh", "Rolling_Avg"]], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = linechart[["month_year", "TSh", "Rolling_Avg"]].to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_rolling_avg.csv", mime='text/csv')

    # Yearly Analysis section
    st.markdown("""
    <div class="section-header">
        <h2>Yearly Average Price Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #333333 !important;'>Annual price trends show how crop prices have changed year-over-year. Useful for identifying long-term patterns.</p>", unsafe_allow_html=True)
    
    for crop in selected_crops:
        yearly_avg = filtered_df[filtered_df["commodity"] == crop].groupby("year")["TSh"].mean().reset_index().round(2)
        fig = px.line(yearly_avg, x="year", y="TSh", markers=True, 
                     title=f"{crop} Yearly Average Prices", height=400,
                     template="plotly_dark")
        fig.update_traces(
            line_color="#4CAF50", 
            line_width=3, 
            mode="lines+markers", 
            marker=dict(color="#2E7D32", size=10)
        )
        fig.update_layout(
            font=dict(color='white'),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(yearly_avg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = yearly_avg.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_yearly_avg.csv", mime='text/csv')

    # Footer
    st.markdown("""
    <div style='background-color: #2E7D32; color: white; padding: 10px; border-radius: 5px; 
    text-align: center; margin-top: 30px; box-shadow: 0px -3px 10px rgba(0, 0, 0, 0.1);'>
        <p style='color: white !important;'>Crop Price Analysis Dashboard © 2023</p>
    </div>
    """, unsafe_allow_html=True)
