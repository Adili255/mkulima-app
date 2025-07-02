import xgboost as xgb
from PIL import Image
import time
import pandas as pd
import streamlit as st
import joblib
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64
import tempfile
import os

def create_prediction_model():
    # Load the dataset
    try:
        df = pd.read_csv('filtered_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Ensure price column is named as TSh (Tanzania Shillings)
        if 'price' in df.columns:
            df = df.rename(columns={'price': 'TSh'})
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return

    # Generate a unique session identifier to ensure unique keys
    if 'prediction_session_id' not in st.session_state:
        st.session_state.prediction_session_id = str(uuid.uuid4())[:8]
    
    session_id = st.session_state.prediction_session_id
    
    xgb_model = joblib.load('xgb_model.pkl')

    regional_mapping = {
            'Arusha': 0,
            'Dar-es-salaam': 1,
            'Dodoma': 2,
            'Iringa': 3,
            'Kagera': 4,
            'Kigoma': 5,
            'Kilimanjaro': 6,
            'Lindi': 7,
            'Manyara': 8,
            'Mara': 9,
            'Mbeya': 10,
            'Morogoro': 11,
            'Mtwara': 12,
            'Mwanza': 13,
            'Rukwa': 14,
            'Ruvuma': 15,
            'Shinyanga': 16,
            'Singida': 17,
            'Tabora': 18,
            'Tanga': 19,
            'Katavi': 20,
            'Njombe': 21,
            'Geita': 22
        }
        
    district_mapping = {
            'Arusha Urban': 0,
            'Ngorongoro': 1,
            'Ilala': 2,
            'Mpwapwa': 3,
            'Iringa Urban': 4,
            'Bukoba Urban': 5,
            'Kigoma Municipal-Ujiji': 6,
            'Moshi Municipal': 7,
            'Lindi Urban': 8,
            'Babati Urban': 9,
            'Musoma Municipal': 10,
            'Mbeya Urban': 11,
            'Morogoro Urban': 12,
            'Mtwara Urban': 13,
            'Nyamagana': 14,
            'Sumbawanga Urban': 15,
            'Songea Urban': 16,
            'Shinyanga Urban': 17,
            'Singida Urban': 18,
            'Tabora Urban': 19,
            'Tanga': 20,
            'Mpanda Urban': 21,
            "Wanging'ombe": 22,
            'Geita': 23,
            'Kinondoni': 24,
            'Temeke': 25,
            'Kongwa': 26
        }  
        
    market_mapping = {
            'Arusha Urban': 0,
            'Ngorongoro': 1,
            'Ilala Market': 2,
            'Dar Es Salaam': 3,
            'Dodoma (Majengo)': 4,
            'Iringa Urban': 5,
            'Bukoba': 6,
            'Kigoma': 7,
            'Moshi': 8,
            'Lindi': 9,
            'Babati': 10,
            'Musoma': 11,
            'Mwanjelwa': 12,
            'Morogoro': 13,
            'Mtwara DC': 14,
            'Mwanza': 15,
            'Sumbawanga': 16,
            'Songea': 17,
            'Shinyanga': 18,
            'Singida': 19,
            'Tabora': 20,
            'Tanga / Mgandini': 21,
            'Mpanda': 22,
            'Njombe': 23,
            'Geita': 24,
            'Ilala (Buguruni)': 25,
            'Kinondoni (Tandale)': 26,
            'Temeke (Tandika)': 27,
            'Kibaigwa': 28,
            'Mbeya (SIDO)': 29,
            'Tanga': 30,
            'Sumbawanga (Katumba)': 31
        }
        
    
    commodity_mapping = {
        'Maize': 0,
        'Rice': 1,
        'Beans': 2,
    }      
    
    # Region to districts mapping
    region_to_districts = {
        'Arusha': ['Arusha Urban', 'Ngorongoro'],
        'Dar-es-salaam': ['Ilala'],
        'Dodoma': ['Mpwapwa'],
        'Iringa': ['Iringa Urban'],
        'Kagera': ['Bukoba Urban'],
        'Kigoma': ['Kigoma Municipal-Ujiji'],
        'Kilimanjaro': ['Moshi Municipal'],
        'Lindi': ['Lindi Urban'],
        'Manyara': ['Babati Urban'],
        'Mara': ['Musoma Municipal'],
        'Mbeya': ['Mbeya Urban'],
        'Morogoro': ['Morogoro Urban'],
        'Mtwara': ['Mtwara Urban'],
        'Mwanza': ['Nyamagana'],
        'Rukwa': ['Sumbawanga Urban'],
        'Ruvuma': ['Songea Urban'],
        'Shinyanga': ['Shinyanga Urban'],
        'Singida': ['Singida Urban'],
        'Tabora': ['Tabora Urban'],
        'Tanga': ['Tanga'],
        'Katavi': ['Mpanda Urban'],
        'Njombe': ["Wanging'ombe"],
        'Geita': ['Geita']
    }
    
    # Region to markets mapping
    region_to_markets = {
        'Arusha': ['Arusha Urban', 'Ngorongoro'],
        'Dar-es-salaam': ['Ilala Market'],
        'Dodoma': ['Dodoma (Majengo)'],
        'Iringa': ['Iringa Urban'],
        'Kagera': ['Bukoba'],
        'Kigoma': ['Kigoma'],
        'Kilimanjaro': ['Moshi'],
        'Lindi': ['Lindi'],
        'Manyara': ['Babati'],
        'Mara': ['Musoma'],
        'Mbeya': ['Mwanjelwa', 'Mbeya (SIDO)'],
        'Morogoro': ['Morogoro'],
        'Mtwara': ['Mtwara DC'],
        'Mwanza': ['Mwanza'],
        'Rukwa': ['Sumbawanga', 'Sumbawanga (Katumba)'],
        'Ruvuma': ['Songea'],
        'Shinyanga': ['Shinyanga'],
        'Singida': ['Singida'],
        'Tabora': ['Tabora'],
        'Tanga': ['Tanga / Mgandini', 'Tanga'],
        'Katavi': ['Mpanda'],
        'Njombe': ['Njombe'],
        'Geita': ['Geita']
    }
    
    # Function to generate synthetic data for future years using the prediction model
    def generate_future_data(df, model, start_year=2024, end_year=2025):
        future_data = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for market in df['market'].unique():
                    for commodity in df['commodity'].unique():
                        # Get corresponding regional and district for the market
                        market_data = df[df['market'] == market].iloc[0]
                        regional = market_data['region']
                        district = market_data['district']
                        
                        input_data = pd.DataFrame({
                            'regional': [regional_mapping.get(regional, 0)],
                            'district': [district_mapping.get(district, 0)],
                            'market': [market_mapping.get(market, 0)],
                            'commodity': [commodity_mapping.get(commodity, 0)],
                            'year': [year],
                            'month': [month]
                        })
                        
                        try:
                            pred_price = model.predict(input_data)[0]
                        except:
                            # Fallback to average price if prediction fails
                            commodity_data = df[df['commodity'] == commodity]
                            pred_price = commodity_data['TSh'].mean() * np.random.uniform(0.9, 1.1)
                        
                        future_data.append({
                            'date': pd.to_datetime(f'{year}-{month}-01'),
                            'market': market,
                            'commodity': commodity,
                            'TSh': pred_price,
                            'region': regional,
                            'district': district,
                            'year': year,
                            'month': month
                        })
        
        return pd.DataFrame(future_data)
    
    # Generate future data (2024-2025) and combine with historical data
    future_df = generate_future_data(df, xgb_model)
    combined_df = pd.concat([df, future_df], ignore_index=True)
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs([" Price Prediction", " Report Generation"])
    
    with tab1:
        st.markdown('<div style="background-color: #4CAF50; color: white; padding: 10px 0; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1); display: flex; align-items: center; justify-content: center;">', unsafe_allow_html=True)
        st.title('Crop Price Prediction')
        st.write("Welcome to the Mkulima consultation system. Enter your details and we'll predict the crop price for you!")
        st.markdown("""
            <div style='background-color: #FFEBEE; padding: 10px; border-radius: 5px; border-left: 4px solid #F44336; margin: 10px 0;'>
                <p style='color: #D32F2F; font-size: 14px;'>
                    <span style='color: #F44336; font-weight: 600;'>For optimal prediction accuracy</span>, 
                    <span style='color: #D32F2F;'>we recommend using this model for forecasts up to year 2026.</span><br>
                    <span style='color: #FF5722; font-weight: 600;'>Predictions beyond this timeframe may be less reliable</span> 
                    <span style='color: #D32F2F;'>due to market volatility and economic factors.</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.subheader('Input Details')

        selected_regional = st.selectbox(
            'Select Regional', 
            sorted(regional_mapping.keys()),
            key=f"pred_regional_select_{session_id}"
        )
        
        selected_district = st.selectbox(
            'Select District', 
            region_to_districts[selected_regional],
            key=f"pred_district_select_{selected_regional}_{session_id}"
        )
        
        selected_market = st.selectbox(
            'Select Market', 
            region_to_markets[selected_regional],
            key=f"pred_market_select_{selected_regional}_{session_id}"
        )

        commodity = st.selectbox(
            'Select Commodity', 
            sorted(commodity_mapping.keys()),
            key=f"pred_commodity_select_{session_id}"
        )
        
        year = st.number_input(
            'Enter Year', 
            min_value=2025, 
            max_value=2026, 
            value=2025,
            key=f"pred_year_input_{session_id}"
        )
        
        month = st.number_input(
            'Enter Month', 
            min_value=1, 
            max_value=12, 
            value=6,
            key=f"pred_month_input_{session_id}"
        )

        st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #4CAF50 !important;
                color: white !important;
                font-weight: bold !important;
                border: none !important;
                padding: 10px 24px !important;
                width: 100% !important;
                border-radius: 4px !important;
            }
            
            div.stButton > button:first-child:hover {
                background-color: #45a049 !important;
            }
            
            button[kind="secondary"] {
                background-color: #4CAF50 !important;
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)

        predict_button = st.button(
            'Predict', 
            key=f'pred_predict_button_{session_id}',
            help='Click to make a prediction'
        )

        st.subheader('Prediction Result')
        st.markdown("""
            <div style='color: #F44336; font-weight: bold; display: inline;'>Disclaimer</div>: 
            <span style='color: inherit;'>The prediction is for informational purposes only and may not reflect real-world prices accurately.</span>
        """, unsafe_allow_html=True)

        def predict_crop_price(regional, district, market, commodity, year, month):
            regional = regional_mapping[regional]
            district = district_mapping[district]
            market = market_mapping[market]
            commodity = commodity_mapping[commodity]

            input_data = pd.DataFrame({
                'regional': [regional],
                'district': [district],
                'market': [market],
                'commodity': [commodity],
                'year': [year],
                'month': [month]
            })

            time.sleep(2)

            prediction = xgb_model.predict(input_data)[0]
            formatted_prediction = '{:,.2f} TSh'.format(prediction)

            return formatted_prediction

        if predict_button:
            with st.spinner('Predicting...'):
                formatted_prediction = predict_crop_price(selected_regional, selected_district, selected_market, commodity, year, month)
                st.success('Prediction complete')
                st.markdown(f'<p style="color: #2196F3; font-size: 18px; font-weight: bold;">Predicted Price: {formatted_prediction}</p>', unsafe_allow_html=True)
                st.markdown("""
                    <div style='color: #F44336; font-weight: bold; display: inline;'>NOTE</div>: 
                    <span style='color: inherit;'>Prediction is for 100kg, typically considered as a wholesale quantity.</span>
                    <br>
                    <span style='color: inherit;'>Prices for 1kg may vary and are often different, especially in retail markets.</span>
                """, unsafe_allow_html=True)

    # Report Generation Tab
    with tab2:
        st.title("Professional Analytical Report Generator")
        st.write("Generate professional analytical report of crop prices for Tanzanian markets")
        
        # Report Configuration Section
        st.subheader("Report inputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique markets from the dataset
            available_markets = combined_df['market'].unique().tolist()
            report_markets = st.multiselect(
                "Select Markets for Analysis",
                options=available_markets,
                default=available_markets[:2] if len(available_markets) > 1 else available_markets,
                key=f"report_markets_{session_id}"
            )
            
            # Get unique commodities from the dataset
            available_commodities = combined_df['commodity'].unique().tolist()
            report_crops = st.multiselect(
                "Select Crops for Analysis",
                options=available_commodities,
                default=available_commodities,
                key=f"report_crops_{session_id}"
            )
        
        with col2:
            # Get min and max years from the dataset
            min_year = int(combined_df['year'].min())
            max_year = int(combined_df['year'].max())
            
            start_year = st.number_input(
                "Start Year",
                min_value=min_year,
                max_value=max_year,
                value=max_year-5,
                key=f"start_year_{session_id}"
            )
            
            end_year = st.number_input(
                "End Year",
                min_value=min_year,
                max_value=max_year,
                value=max_year,
                key=f"end_year_{session_id}"
            )
        
        # Report Generation Functions
        def filter_data(df, markets, crops, start_year, end_year):
            """Filter the dataset based on user selections"""
            return df[
                (df['market'].isin(markets)) & 
                (df['commodity'].isin(crops)) & 
                (df['year'] >= start_year) & 
                (df['year'] <= end_year)
            ]
        
        def create_trend_analysis_chart(df, crop):
            """Create trend analysis chart for a specific crop using actual data"""
            crop_data = df[df['commodity'] == crop]
            monthly_avg = crop_data.groupby(['year', 'month'])['TSh'].mean().reset_index()
            monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
            
            # Separate historical and predicted data
            historical = monthly_avg[monthly_avg['date'] < pd.to_datetime('2024-01-01')]
            predicted = monthly_avg[monthly_avg['date'] >= pd.to_datetime('2024-01-01')]
            
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['TSh'],
                mode='lines+markers',
                name=f'{crop} Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add predicted data
            if not predicted.empty:
                fig.add_trace(go.Scatter(
                    x=predicted['date'],
                    y=predicted['TSh'],
                    mode='lines+markers',
                    name=f'{crop} Predicted Price',
                    line=dict(color='red', width=2, dash='dot')
                ))
            
            fig.update_layout(
                title=f'{crop} Price Trend Analysis (TSh)',
                xaxis_title='Date',
                yaxis_title='Price (TSh per 100kg)',
                template='plotly_white',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        
        def create_market_comparison_chart(df, crop):
            """Create market comparison chart using actual data"""
            crop_data = df[df['commodity'] == crop]
            market_avg = crop_data.groupby('market')['TSh'].mean().reset_index()
            
            # Sort by price for better visualization
            market_avg = market_avg.sort_values('TSh', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(x=market_avg['market'], y=market_avg['TSh'],
                       marker_color='lightblue')
            ])
            
            fig.update_layout(
                title=f'{crop} - Average Price by Market (TSh)',
                xaxis_title='Market',
                yaxis_title='Average Price (TSh per 100kg)',
                template='plotly_white',
                height=400
            )
            
            return fig
        
        def generate_predictions(markets, crops, model):
            """Generate future predictions"""
            predictions = []
            current_date = datetime.now()
            
            for i in range(5):  # Next 5 months
                pred_date = current_date + timedelta(days=30*i)
                year = pred_date.year
                month = pred_date.month
                
                for market in markets:
                    for crop in crops:
                        # Find corresponding regional and district for the market
                        market_data = df[df['market'] == market].iloc[0]
                        regional = market_data['region']
                        district = market_data['district']
                        
                        input_data = pd.DataFrame({
                            'regional': [regional_mapping.get(regional, 0)],
                            'district': [district_mapping.get(district, 0)],
                            'market': [market_mapping.get(market, 0)],
                            'commodity': [commodity_mapping.get(crop, 0)],
                            'year': [year],
                            'month': [month]
                        })
                        
                        try:
                            pred_price = model.predict(input_data)[0]
                        except:
                            # Fallback to average price if prediction fails
                            crop_data = df[df['commodity'] == crop]
                            pred_price = crop_data['TSh'].mean() * np.random.uniform(0.9, 1.2)
                            
                        predictions.append({
                            'date': pred_date,
                            'market': market,
                            'crop': crop,
                            'predicted_price': pred_price
                        })
            
            return pd.DataFrame(predictions)
        
        def calculate_statistics(df):
            """Calculate statistical summary from actual data"""
            stats = []
            
            for crop in df['commodity'].unique():
                for market in df['market'].unique():
                    subset = df[(df['commodity'] == crop) & (df['market'] == market)]
                    if not subset.empty:
                        stats.append({
                            'Crop': crop,
                            'Market': market,
                            'Min Price (TSh)': f"{subset['TSh'].min():,.2f}",
                            'Max Price (TSh)': f"{subset['TSh'].max():,.2f}",
                            'Avg Price (TSh)': f"{subset['TSh'].mean():,.2f}",
                            'Std Dev (TSh)': f"{subset['TSh'].std():,.2f}",
                            'Latest Price (TSh)': f"{subset['TSh'].iloc[-1]:,.2f}"
                        })
            
            return pd.DataFrame(stats)
        
        # Report Generation Button
        generate_report_btn = st.button(
            "Generate Report",
            key=f"generate_report_{session_id}",
            help="Click to generate a detailed analytical report"
        )
        
        if generate_report_btn:
            if not report_markets or not report_crops:
                st.error("Please select at least one market and one crop for analysis.")
            else:
                with st.spinner("Generating comprehensive report... This may take a few moments."):
                    
                    # Filter the dataset
                    filtered_data = filter_data(combined_df, report_markets, report_crops, start_year, end_year)
                    
                    if filtered_data.empty:
                        st.error("No data available for the selected filters. Please adjust your selections.")
                        return
                    
                    # Generate predictions
                    prediction_data = generate_predictions(report_markets, report_crops, xgb_model)
                    
                    # Calculate statistics
                    stats_summary = calculate_statistics(filtered_data)
                    
                    # Display Report Sections
                    st.success("Report generated successfully!")
                    
                    # 1. Executive Summary
                    st.header("Executive Summary")
                    
                    key_insights = []
                    for crop in report_crops:
                        crop_data = filtered_data[filtered_data['commodity'] == crop]
                        monthly_trend = crop_data.groupby('month')['TSh'].mean()
                        peak_month = monthly_trend.idxmax()
                        peak_price = monthly_trend.max()
                        
                        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                                     5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                     9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                        
                        key_insights.append(f"**{crop}** prices peaked in {month_names[peak_month]} with average price of {peak_price:,.2f} TSh per 100kg")
                    
                    st.markdown("**Key Findings:**")
                    for insight in key_insights:
                        st.markdown(f"• {insight}")
                    
                    st.markdown(f"""
                    **Analysis Overview:**
                    - **Crops Analyzed:** {', '.join(report_crops)}
                    - **Markets Covered:** {', '.join(report_markets)}
                    - **Analysis Period:** {start_year} - {end_year}
                    - **Total Records:** {len(filtered_data):,}
                    """)
                    
                    # 2. Data Overview
                    st.header("Data Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Markets Analyzed", len(report_markets))
                    with col2:
                        st.metric("Crops Analyzed", len(report_crops))
                    with col3:
                        st.metric("Data Points", f"{len(filtered_data):,}")
                    
                    st.markdown("""
                    **Data Sources:** Tanzania Ministry of Agriculture Official Data  
                    **Data Period:** 2006-2023 (Historical), 2024-2025 (Predicted)  
                    **Model Used:** XGBoost Machine Learning Algorithm
                    """)
                    
                    # 3. Trend Analysis
                    st.header("Trend Analysis")
                    
                    for crop in report_crops:
                        st.subheader(f"{crop} price Analysis")
                        
                        # Time series chart
                        trend_fig = create_trend_analysis_chart(filtered_data, crop)
                        st.plotly_chart(trend_fig, use_container_width=True)
                        
                        # Market comparison
                        market_fig = create_market_comparison_chart(filtered_data, crop)
                        st.plotly_chart(market_fig, use_container_width=True)
                        
                        # Monthly averages
                        crop_data = filtered_data[filtered_data['commodity'] == crop]
                        monthly_avg = crop_data.groupby('month')['TSh'].mean()
                        
                        st.markdown("**Monthly Average Prices (TSh):**")
                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        monthly_df = pd.DataFrame({
                            'Month': months,
                            'Average Price (TSh)': [f"{monthly_avg.get(i+1, 0):,.2f}" for i in range(12)]
                        })
                        st.dataframe(monthly_df, use_container_width=True)
                        
                        # Price insights
                        min_price = crop_data['TSh'].min()
                        max_price = crop_data['TSh'].max()
                        avg_price = crop_data['TSh'].mean()
                        
                        st.markdown(f"""
                        **{crop} Price Insights:**
                        - **Highest Price:** {max_price:,.2f} TSh per 100kg
                        - **Lowest Price:** {min_price:,.2f} TSh per 100kg
                        - **Average Price:** {avg_price:,.2f} TSh per 100kg
                        - **Price Volatility:** {((max_price - min_price) / avg_price * 100):,.1f}%
                        """)
                    
                    # 4. Prediction Summary
                    st.header("Price Predictions for the next 5 months")
                    
                    # Create prediction visualization
                    fig_pred = go.Figure()
                    
                    for crop in report_crops:
                        crop_predictions = prediction_data[prediction_data['crop'] == crop]
                        avg_predictions = crop_predictions.groupby('date')['predicted_price'].mean().reset_index()
                        
                        fig_pred.add_trace(go.Scatter(
                            x=avg_predictions['date'],
                            y=avg_predictions['predicted_price'],
                            mode='lines+markers',
                            name=f'{crop} Forecast',
                            line=dict(width=3)
                        ))
                    
                    fig_pred.update_layout(
                        title='Price Forecast - Next 5 Months (TSh)',
                        xaxis_title='Date',
                        yaxis_title='Predicted Price (TSh per 100kg)',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Display prediction table
                    st.subheader("Detailed Prediction Values")
                    prediction_table = prediction_data.copy()
                    prediction_table['date'] = prediction_table['date'].dt.strftime('%Y-%m-%d')
                    prediction_table['predicted_price'] = prediction_table['predicted_price'].apply(lambda x: f"{x:,.2f} TSh")
                    prediction_table = prediction_table.rename(columns={
                        'date': 'Date',
                        'market': 'Market',
                        'crop': 'Crop',
                        'predicted_price': 'Predicted Price'
                    })
                    st.dataframe(prediction_table, use_container_width=True)
                    
                    # 5. Statistical Summary
                    st.header("Statistical Summary")
                    st.dataframe(stats_summary, use_container_width=True)
                    
                    # 6. Market-Specific Analysis
                    st.header("Markets Analysis")
                    
                    # Calculate market volatility rankings
                    volatility_data = []
                    for market in report_markets:
                        market_data = filtered_data[filtered_data['market'] == market]
                        if not market_data.empty:
                            volatility = market_data.groupby('commodity')['TSh'].std().mean()
                            volatility_data.append({
                                'Market': market,
                                'Average Volatility': volatility
                            })
                    
                    volatility_df = pd.DataFrame(volatility_data)
                    if not volatility_df.empty:
                        volatility_df = volatility_df.sort_values('Average Volatility', ascending=False)
                        volatility_df['Volatility Level'] = pd.qcut(volatility_df['Average Volatility'], 
                                                                   q=3, 
                                                                   labels=['Low Volatility', 'Medium Volatility', 'High Volatility'])
                    
                    for market in report_markets:
                        st.subheader(f" {market} Market")
                        
                        market_data = filtered_data[filtered_data['market'] == market]
                        
                        # Create market overview chart
                        fig_market = go.Figure()
                        
                        for crop in report_crops:
                            crop_market_data = market_data[market_data['commodity'] == crop]
                            monthly_avg = crop_market_data.groupby(['year', 'month'])['TSh'].mean().reset_index()
                            monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))
                            
                            fig_market.add_trace(go.Scatter(
                                x=monthly_avg['date'],
                                y=monthly_avg['TSh'],
                                mode='lines+markers',
                                name=f'{crop}',
                                line=dict(width=2)
                            ))
                        
                        fig_market.update_layout(
                            title=f'{market} - All Crops Price Trends (TSh)',
                            xaxis_title='Date',
                            yaxis_title='Price (TSh per 100kg)',
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_market, use_container_width=True)
                        
                        # Market insights
                        market_stats = []
                        for crop in report_crops:
                            crop_data = market_data[market_data['commodity'] == crop]
                            if not crop_data.empty:
                                volatility = crop_data['TSh'].std()
                                avg_price = crop_data['TSh'].mean()
                                market_stats.append({
                                    'Crop': crop,
                                    'Average Price (TSh)': f"{avg_price:,.2f}",
                                    'Volatility (Std Dev)': f"{volatility:,.2f}",
                                    'Coefficient of Variation': f"{(volatility/avg_price*100):,.1f}%"
                                })
                        
                        if market_stats:
                            st.dataframe(pd.DataFrame(market_stats), use_container_width=True)
                    
                    # Market volatility comparison
                    if not volatility_df.empty:
                        st.subheader("Market Volatility Comparison")
                        
                        # Create volatility comparison chart
                        fig_volatility = go.Figure()
                        
                        fig_volatility.add_trace(go.Bar(
                            x=volatility_df['Market'],
                            y=volatility_df['Average Volatility'],
                            marker_color=['red' if level == 'High Volatility' else 
                                         'orange' if level == 'Medium Volatility' else 
                                         'green' for level in volatility_df['Volatility Level']],
                            text=volatility_df['Volatility Level'],
                            textposition='auto'
                        ))
                        
                        fig_volatility.update_layout(
                            title='Market Volatility Comparison (Average Standard Deviation)',
                            xaxis_title='Market',
                            yaxis_title='Average Volatility (Standard Deviation)',
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_volatility, use_container_width=True)
                        
                        # Display volatility rankings
                        st.markdown("**Market Volatility Rankings:**")
                        st.dataframe(volatility_df[['Market', 'Average Volatility', 'Volatility Level']]
                                    .sort_values('Average Volatility', ascending=False)
                                    .reset_index(drop=True), 
                                   use_container_width=True)
                        
                        # Volatility insights
                        high_vol_markets = volatility_df[volatility_df['Volatility Level'] == 'High Volatility']['Market'].tolist()
                        low_vol_markets = volatility_df[volatility_df['Volatility Level'] == 'Low Volatility']['Market'].tolist()
                        
                        if high_vol_markets:
                            st.markdown(f"**High Volatility Markets:** {', '.join(high_vol_markets)} - These markets show significant price fluctuations, which may present both higher risks and opportunities.")
                        
                        if low_vol_markets:
                            st.markdown(f"**Low Volatility Markets:** {', '.join(low_vol_markets)} - These markets show more stable prices, which may be preferable for risk-averse traders.")
                    
                    # 7. Recommendations
                    st.header("market Insights and Recommendations")
                    
                    recommendations = []
                    
                    for crop in report_crops:
                        crop_data = filtered_data[filtered_data['commodity'] == crop]
                        monthly_avg = crop_data.groupby('month')['TSh'].mean()
                        lowest_month = monthly_avg.idxmin()
                        highest_month = monthly_avg.idxmax()
                        
                        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                                     5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                     9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                        
                        recommendations.append(f"**{crop}**: Buy in {month_names[lowest_month]} when prices are lowest, sell in {month_names[highest_month]} when prices peak")
                    
                    # Market-specific recommendations
                    market_recommendations = []
                    for market in report_markets:
                        market_data = filtered_data[filtered_data['market'] == market]
                        if not market_data.empty:
                            most_stable_crop = None
                            min_volatility = float('inf')
                            
                            for crop in report_crops:
                                crop_data = market_data[market_data['commodity'] == crop]
                                if not crop_data.empty:
                                    volatility = crop_data['TSh'].std()
                                    if volatility < min_volatility:
                                        min_volatility = volatility
                                        most_stable_crop = crop
                            
                            if most_stable_crop:
                                market_recommendations.append(f"**{market}**: {most_stable_crop} shows the most stable prices (lowest volatility)")
                    
                    st.markdown("**Seasonal Buying/Selling Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                    
                    st.markdown("**Market-Specific Insights:**")
                    for market_rec in market_recommendations:
                        st.markdown(f"• {market_rec}")
                    
                    # 8. PDF Report Generation
                    st.header("Generate Report")
                    
                    def create_pdf_report():
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=A4)
                        styles = getSampleStyleSheet()
                        
                        # Custom styles
                        styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
                        styles.add(ParagraphStyle(name='Right', alignment=TA_RIGHT))
                        
                        # Story holds all the elements
                        story = []
                        
                        # Title
                        title_style = ParagraphStyle(
                            name='Title',
                            parent=styles['Heading1'],
                            fontSize=18,
                            alignment=TA_CENTER,
                            spaceAfter=20
                        )
                        
                        story.append(Paragraph("CROP PRICE ANALYSIS REPORT", title_style))
                        story.append(Spacer(1, 12))
                        
                        # Report metadata
                        meta_style = styles['BodyText']
                        meta_style.alignment = TA_CENTER
                        
                        story.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
                        story.append(Paragraph(f"<b>Markets:</b> {', '.join(report_markets)}", meta_style))
                        story.append(Paragraph(f"<b>Crops:</b> {', '.join(report_crops)}", meta_style))
                        story.append(Paragraph(f"<b>Period:</b> {start_year} - {end_year}", meta_style))
                        
                        story.append(Spacer(1, 24))
                        
                        # Executive Summary
                        story.append(Paragraph("1. Executive Summary", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        summary_text = f"""
                        This report provides a comprehensive analysis of crop prices across {len(report_markets)} markets in Tanzania, 
                        covering {len(report_crops)} major crops. The analysis period spans from {start_year} to {end_year}, 
                        with predictive insights for the coming months.
                        """
                        story.append(Paragraph(summary_text, styles['BodyText']))
                        story.append(Spacer(1, 12))
                        
                        # Key Findings
                        story.append(Paragraph("<b>Key Findings:</b>", styles['BodyText']))
                        for insight in key_insights:
                            story.append(Paragraph(f"• {insight.replace('**', '')}", styles['BodyText']))
                        
                        story.append(PageBreak())
                        
                        # Trend Analysis
                        story.append(Paragraph("2. Trend Analysis", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        for crop in report_crops:
                            story.append(Paragraph(f"{crop} Price Trends", styles['Heading3']))
                            
                            # Create temporary file for trend chart
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                trend_fig = create_trend_analysis_chart(filtered_data, crop)
                                trend_fig.write_image(tmpfile.name)
                                img = RLImage(tmpfile.name, width=6*inch, height=3*inch)
                                story.append(img)
                                story.append(Spacer(1, 12))
                                os.unlink(tmpfile.name)  # Clean up
                            
                            # Create temporary file for market comparison chart
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                market_fig = create_market_comparison_chart(filtered_data, crop)
                                market_fig.write_image(tmpfile.name)
                                img = RLImage(tmpfile.name, width=6*inch, height=3*inch)
                                story.append(img)
                                story.append(Spacer(1, 12))
                                os.unlink(tmpfile.name)  # Clean up
                        
                        story.append(PageBreak())
                        
                        # Predictions
                        story.append(Paragraph("3. Price Predictions", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        # Create temporary file for prediction chart
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            fig_pred = go.Figure()
                            for crop in report_crops:
                                crop_predictions = prediction_data[prediction_data['crop'] == crop]
                                avg_predictions = crop_predictions.groupby('date')['predicted_price'].mean().reset_index()
                                
                                fig_pred.add_trace(go.Scatter(
                                    x=avg_predictions['date'],
                                    y=avg_predictions['predicted_price'],
                                    mode='lines+markers',
                                    name=f'{crop} Forecast',
                                    line=dict(width=3)
                                ))
                            
                            fig_pred.update_layout(
                                title='Price Forecast - Next 5 Months (TSh)',
                                xaxis_title='Date',
                                yaxis_title='Predicted Price (TSh per 100kg)',
                                template='plotly_white',
                                height=500
                            )
                            
                            fig_pred.write_image(tmpfile.name)
                            img = RLImage(tmpfile.name, width=6*inch, height=4*inch)
                            story.append(img)
                            story.append(Spacer(1, 12))
                            os.unlink(tmpfile.name)  # Clean up
                        
                        # Add prediction table
                        story.append(Paragraph("<b>Detailed Prediction Values:</b>", styles['Heading3']))
                        story.append(Spacer(1, 6))
                        
                        # Prepare table data
                        table_data = [['Date', 'Market', 'Crop', 'Predicted Price (TSh)']]
                        for _, row in prediction_data.iterrows():
                            table_data.append([
                                row['date'].strftime('%Y-%m-%d'),
                                row['market'],
                                row['crop'],
                                f"{row['predicted_price']:,.2f}"
                            ])
                        
                        # Create table
                        pred_table = Table(table_data)
                        pred_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ]))
                        
                        story.append(pred_table)
                        story.append(PageBreak())
                        
                        # Market Volatility Analysis
                        story.append(Paragraph("4. Market Volatility Analysis", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        if not volatility_df.empty:
                            # Create temporary file for volatility chart
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                                fig_volatility = go.Figure()
                                fig_volatility.add_trace(go.Bar(
                                    x=volatility_df['Market'],
                                    y=volatility_df['Average Volatility'],
                                    marker_color=['red' if level == 'High Volatility' else 
                                                 'orange' if level == 'Medium Volatility' else 
                                                 'green' for level in volatility_df['Volatility Level']]
                                ))
                                
                                fig_volatility.update_layout(
                                    title='Market Volatility Comparison (Average Standard Deviation)',
                                    xaxis_title='Market',
                                    yaxis_title='Average Volatility (Standard Deviation)',
                                    template='plotly_white',
                                    height=400
                                )
                                
                                fig_volatility.write_image(tmpfile.name)
                                img = RLImage(tmpfile.name, width=6*inch, height=4*inch)
                                story.append(img)
                                story.append(Spacer(1, 12))
                                os.unlink(tmpfile.name)  # Clean up
                            
                            # Add volatility table
                            story.append(Paragraph("<b>Market Volatility Rankings:</b>", styles['Heading3']))
                            story.append(Spacer(1, 6))
                            
                            # Prepare table data
                            vol_table_data = [['Market', 'Average Volatility', 'Volatility Level']]
                            for _, row in volatility_df.iterrows():
                                vol_table_data.append([
                                    row['Market'],
                                    f"{row['Average Volatility']:,.2f}",
                                    row['Volatility Level']
                                ])
                            
                            # Create table
                            vol_table = Table(vol_table_data)
                            vol_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ]))
                            
                            story.append(vol_table)
                            story.append(Spacer(1, 12))
                            
                            # Add volatility insights
                            if high_vol_markets:
                                story.append(Paragraph(f"<b>High Volatility Markets:</b> {', '.join(high_vol_markets)}", styles['BodyText']))
                                story.append(Paragraph("These markets show significant price fluctuations, which may present both higher risks and opportunities.", styles['BodyText']))
                                story.append(Spacer(1, 6))
                            
                            if low_vol_markets:
                                story.append(Paragraph(f"<b>Low Volatility Markets:</b> {', '.join(low_vol_markets)}", styles['BodyText']))
                                story.append(Paragraph("These markets show more stable prices, which may be preferable for risk-averse traders.", styles['BodyText']))
                                story.append(Spacer(1, 6))
                        
                        story.append(PageBreak())
                        
                        # Recommendations
                        story.append(Paragraph("5. Recommendations", styles['Heading2']))
                        story.append(Spacer(1, 12))
                        
                        story.append(Paragraph("<b>Seasonal Buying/Selling Recommendations:</b>", styles['BodyText']))
                        for rec in recommendations:
                            story.append(Paragraph(f"• {rec.replace('**', '')}", styles['BodyText']))
                        
                        story.append(Spacer(1, 12))
                        
                        story.append(Paragraph("<b>Market-Specific Insights:</b>", styles['BodyText']))
                        for market_rec in market_recommendations:
                            story.append(Paragraph(f"• {market_rec.replace('**', '')}", styles['BodyText']))
                        
                        # Build the PDF
                        doc.build(story)
                        buffer.seek(0)
                        return buffer
                    
                    # PDF Download Button
                    pdf_buffer = create_pdf_report()
                    st.download_button(
                        label="Download Full Report",
                        data=pdf_buffer,
                        file_name=f"crop_price_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_download_{session_id}"
                    )

# Run the app
if __name__ == "__main__":
    create_prediction_model()
