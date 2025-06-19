import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(page_title="analysisPage", page_icon=":bar_chart:", layout="wide")

# Title and layout tweaks
st.title("📊 Sample Superstore EDA")
st.markdown('<style>div.block-container{padding-top:1rem}</style>', unsafe_allow_html=True)

# ✅ Load your dataset directly without user upload
try:
    # Adjust path as needed
    file_path = r"filtered_data.csv"
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
except FileNotFoundError:
    st.error(f"Dataset not found at: {file_path}")
    st.stop()


col1, col2 = st.columns((2))
df["date"] = pd.to_datetime(df["date"])

# max and min date
startDate = pd.to_datetime(df["date"]).min()
endDate = pd.to_datetime(df["date"]).max()

col1, col2 = st.columns(2)
with col1:
    start_date = pd.to_datetime(st.date_input("Start Date", startDate))
with col2:
    end_date = pd.to_datetime(st.date_input("End Date", endDate))



st.sidebar.header("Choose your filter: ")

# Region filter
region = st.sidebar.multiselect("Pick your region", df["region"].unique())
df_region = df[df["region"].isin(region)] if region else df.copy()

# District filter based on region
district = st.sidebar.multiselect("Pick your district", df_region["district"].unique())
df_district = df_region[df_region["district"].isin(district)] if district else df_region.copy()

# Market filter based on district
market = st.sidebar.multiselect("Pick your market", df_district["market"].unique())
df_market = df_district[df_district["market"].isin(market)] if market else df_district.copy()

# Commodity filter based on market
commodity = st.sidebar.multiselect("Pick your commodity", df_market["commodity"].unique())
filtered_df = df_market[df_market["commodity"].isin(commodity)] if commodity else df_market.copy()




# Chart 1 - Commodity-wise Average Prices
category_avg = (
    filtered_df.groupby("commodity", as_index=False)["TSh"]
    .mean()
    .round(2)
    .sort_values(by="TSh", ascending=False)
)

with col1:
    st.subheader("🧺 Average Price by Commodity")
    fig = px.bar(
        category_avg,
        x="commodity",
        y="TSh",
        color="commodity",
        template="plotly",
        labels={"TSh": "Average Price (TSh)", "commodity": "Commodity"},
        title="Average Crop Prices"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📥 Download Average Commodity Prices"):
        st.write(category_avg.style.background_gradient(cmap="Blues"))
        csv = category_avg.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="average_commodity_prices.csv",
            mime="text/csv"
        )

# Chart 2 - Region-wise Contribution to Market Value
region_value = (
    filtered_df.groupby("region", as_index=False)["TSh"]
    .sum()
    .round(2)
    .sort_values(by="TSh", ascending=False)
)

with col2:
    st.subheader("🌍 Market Value by Region")
    fig = px.pie(
        region_value,
        values="TSh",
        names="region",
        hole=0.4,
        title="Regional Market Price Contribution",
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📥 Download Regional Market Data"):
        st.write(region_value.style.background_gradient(cmap="Oranges"))
        csv = region_value.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="regional_market_contribution.csv",
            mime="text/csv"
        )


# Safely create "month_year" column
filtered_df["month_year"] = pd.to_datetime(filtered_df["date"]).dt.to_period('M').astype(str)

st.subheader('📈 Time Series Analysis by Crop')

# Define crops you care about (can be a subset of all available)
tracked_crops = ["Beans", "Maize", "Rice"]

# Determine which crops to show based on filter
selected_crops = (
    filtered_df["commodity"].unique().tolist()
    if not commodity or set(commodity) == set(df["commodity"].unique())
    else [crop for crop in tracked_crops if crop in commodity]
)

# Now, loop and show charts only for selected crops
for crop in selected_crops:
    st.markdown(f"### 📌 {crop} Price Trend")

    crop_df = filtered_df[filtered_df["commodity"] == crop]

    linechart = (
        crop_df.groupby("month_year")["TSh"]
        .mean()
        .reset_index()
    )

    # Format and sort time column
    linechart["month_year"] = pd.to_datetime(linechart["month_year"])
    linechart = linechart.sort_values("month_year")
    linechart["month_year"] = linechart["month_year"].dt.strftime("%Y : %b")

    fig = px.line(
        linechart,
        x="month_year",
        y="TSh",
        labels={"TSh": "Average Price (TSh)", "month_year": "Month-Year"},
        title=f"{crop} Prices Over Time",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"View and Download {crop} Time Series Data"):
        st.write(linechart.style.background_gradient(cmap="Blues"))
        csv = linechart.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{crop}_TimeSeries.csv",
            mime="text/csv",
            help=f"Download {crop} time series data"
        )

# Ensure 'month_year' is parsed correctly
filtered_df["month_year"] = pd.to_datetime(filtered_df["date"]).dt.to_period('M').astype(str)

st.subheader("📊 3-Month Rolling Average Price Trends")

# Define target crops
tracked_crops = ["Beans", "Maize", "Rice"]

# Filter selected crops
selected_crops = (
    filtered_df["commodity"].unique().tolist()
    if not commodity or set(commodity) == set(df["commodity"].unique())
    else [crop for crop in tracked_crops if crop in commodity]
)

# Loop over selected crops
for crop in selected_crops:
    st.markdown(f"### 📌 {crop} (3-Month Rolling Average)")

    crop_df = filtered_df[filtered_df["commodity"] == crop]

    # Group by month and get average prices
    linechart = (
        crop_df.groupby("month_year")["TSh"]
        .mean()
        .reset_index()
    )

    # Convert to datetime and sort
    linechart["month_year"] = pd.to_datetime(linechart["month_year"])
    linechart = linechart.sort_values("month_year")

    # Compute 3-month rolling average
    linechart["Rolling_Avg"] = linechart["TSh"].rolling(window=3).mean()

    # Format for display
    linechart["month_year_label"] = linechart["month_year"].dt.strftime("%Y : %b")

    # Plot both original and rolling average
    fig = px.line(
        linechart,
        x="month_year_label",
        y=["TSh", "Rolling_Avg"],
        labels={"value": "Price (TSh)", "month_year_label": "Month-Year"},
        title=f"{crop} Price Trend with 3-Month Rolling Average",
        height=400,
        template="plotly_white"
    )
    fig.update_traces(mode="lines+markers")

    st.plotly_chart(fig, use_container_width=True)

    # Add download option
    with st.expander(f"📥 View and Download {crop} Rolling Average Data"):
        download_df = linechart[["month_year_label", "TSh", "Rolling_Avg"]]
        st.write(download_df.style.background_gradient(cmap="Greens"))
        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{crop}_RollingAvg.csv",
            mime="text/csv",
            help=f"Download {crop} rolling average data"
        )
