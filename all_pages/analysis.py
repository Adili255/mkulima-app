import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_analysis_page():
    st.markdown("""
    <style>
        .compact-expander {
            border: 1px solid #4CAF50 !important;
            border-radius: 5px !important;
            margin: 5px 0 !important;
        }
        .compact-expander .st-emotion-cache-1hynsf2 {
            padding: 0.5rem 1rem !important;
        }
        .compact-data {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #2E7D32; color: white; padding: 15px; border-radius: 5px; 
    box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.2); text-align: center; margin-bottom: 20px;'>
        <h1>Crop Price Analysis Dashboard</h1>
        <p>Analyze crop price trends across Tanzania's regions and markets</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_csv("filtered_data.csv", encoding="ISO-8859-1")
        df["date"] = pd.to_datetime(df["date"])
    except FileNotFoundError:
        st.error("Dataset not found at: filtered_data.csv")
        st.stop()

    st.sidebar.markdown("""
    <div style='background-color: #C8E6C9; padding: 10px; border-radius: 5px; border-left: 5px solid #2E7D32;'>
        <h3 style='text-align: center; color: #1B5E20;'>Analysis Filters</h3>
    </div>
    """, unsafe_allow_html=True)
    
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

    st.markdown("""
    <div style='background-color: #E8F5E9; padding: 15px; border-radius: 8px; margin-top: 20px; 
    border-left: 5px solid #2E7D32; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
        <h2 style='text-align: center; color: #1B5E20;'>Regional Market Price Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("This section shows average prices by region for selected crops. Compare how different regions perform for each commodity.")
    crops_to_analyze = commodity if commodity else display_df["commodity"].unique().tolist()
    region_crop_avg = display_df.groupby(["region", "commodity"])["TSh"].mean().reset_index().round(2)
    
    for crop in crops_to_analyze:
        crop_data = region_crop_avg[region_crop_avg["commodity"] == crop].sort_values("TSh", ascending=False)
        if not crop_data.empty:
            fig = px.bar(crop_data, x="region", y="TSh", title=f"Average {crop} Prices by Region",
                        labels={"TSh": "Average Price (TSh)", "region": "Region"}, template="plotly_white",
                        color="TSh", color_continuous_scale=["#C8E6C9", "#81C784", "#4CAF50", "#2E7D32", "#1B5E20"])
            fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander(f"View data for {crop}", expanded=False):
                st.markdown('<div class="compact-data">', unsafe_allow_html=True)
                st.dataframe(crop_data, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                csv = crop_data.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_regional_prices.csv", mime='text/csv')

    filtered_df["month_year"] = pd.to_datetime(filtered_df["date"]).dt.to_period('M').astype(str)
    st.markdown("""
    <div style='background-color: #E8F5E9; padding: 15px; border-radius: 8px; margin-top: 20px; 
    border-left: 5px solid #2E7D32; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
        <h2 style='text-align: center; color: #1B5E20;'>Time Series Analysis by Crop</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("Track monthly price movements for selected crops. The line charts show how prices have changed over time.")
    selected_crops = filtered_df["commodity"].unique().tolist() if not commodity else [c for c in ["Beans", "Maize", "Rice"] if c in commodity]
    
    for crop in selected_crops:
        crop_df = filtered_df[filtered_df["commodity"] == crop]
        linechart = crop_df.groupby("month_year")["TSh"].mean().reset_index()
        linechart["month_year"] = pd.to_datetime(linechart["month_year"]).dt.strftime("%Y : %b")
        fig = px.line(linechart, x="month_year", y="TSh", labels={"TSh": "Average Price (TSh)", "month_year": "Month-Year"},
                     title=f"{crop} Prices Over Time", height=400, template="plotly_white")
        fig.update_traces(line_color="#4CAF50", line_width=3, mode="lines+markers", marker=dict(color="#2E7D32"))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(linechart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = linechart.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_time_series.csv", mime='text/csv')

    st.markdown("""
    <div style='background-color: #E8F5E9; padding: 15px; border-radius: 8px; margin-top: 20px; 
    border-left: 5px solid #2E7D32; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
        <h2 style='text-align: center; color: #1B5E20;'>3-Month Rolling Average Price Trends</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("The rolling average smooths out short-term fluctuations to show longer-term trends. Blue line shows actual prices while orange shows the smoothed trend.")
    for crop in selected_crops:
        crop_df = filtered_df[filtered_df["commodity"] == crop]
        linechart = crop_df.groupby("month_year")["TSh"].mean().reset_index()
        linechart["month_year"] = pd.to_datetime(linechart["month_year"])
        linechart["Rolling_Avg"] = linechart["TSh"].rolling(window=3).mean()
        fig = px.line(linechart, x=linechart["month_year"].dt.strftime("%Y : %b"), y=["TSh", "Rolling_Avg"],
                     labels={"value": "Price (TSh)", "variable": "Metric"}, title=f"{crop} Price Trends", height=400)
        fig.update_traces(line_color="#4CAF50", selector=dict(name="TSh"))
        fig.update_traces(line_color="#FFA500", selector=dict(name="Rolling_Avg"))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(linechart[["month_year", "TSh", "Rolling_Avg"]], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = linechart[["month_year", "TSh", "Rolling_Avg"]].to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_rolling_avg.csv", mime='text/csv')

    st.markdown("""
    <div style='background-color: #E8F5E9; padding: 15px; border-radius: 8px; margin-top: 20px; 
    border-left: 5px solid #2E7D32; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
        <h2 style='text-align: center; color: #1B5E20;'>Yearly Average Price Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("Annual price trends show how crop prices have changed year-over-year. Useful for identifying long-term patterns.")
    for crop in selected_crops:
        yearly_avg = filtered_df[filtered_df["commodity"] == crop].groupby("year")["TSh"].mean().reset_index().round(2)
        fig = px.line(yearly_avg, x="year", y="TSh", markers=True, title=f"{crop} Yearly Average Prices", height=400)
        fig.update_traces(line_color="#4CAF50", line_width=3, mode="lines+markers", marker=dict(color="#2E7D32", size=10))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"View data for {crop}", expanded=False):
            st.markdown('<div class="compact-data">', unsafe_allow_html=True)
            st.dataframe(yearly_avg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            csv = yearly_avg.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {crop} data", data=csv, file_name=f"{crop}_yearly_avg.csv", mime='text/csv')

    st.markdown("""
    <div style='background-color: #2E7D32; color: white; padding: 10px; border-radius: 5px; 
    text-align: center; margin-top: 30px; box-shadow: 0px -3px 10px rgba(0, 0, 0, 0.1);'>
    </div>
    """, unsafe_allow_html=True)