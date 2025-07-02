import xgboost as xgb
from PIL import Image
import time
import pandas as pd
import streamlit as st
import joblib
import uuid

def create_prediction_model():
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
        'Ilala': 1,
        'Mpwapwa': 2,
        'Iringa Urban': 3,
        'Bukoba Urban': 4,
        'Kigoma Municipal-Ujiji': 5,
        'Moshi Municipal': 6,
        'Lindi Urban': 7,
        'Babati Urban': 8,
        'Musoma Municipal': 9,
        'Mbeya Urban': 10,
        'Morogoro Urban': 11,
        'Mtwara Urban': 12,
        'Nyamagana': 13,
        'Sumbawanga Urban': 14,
        'Songea Urban': 15,
        'Shinyanga Urban': 16,
        'Singida Urban': 17,
        'Tabora Urban': 18,
        'Tanga': 19,
        'Mpanda Urban': 20,
        "Wanging'ombe": 21,
        'Geita': 22,
        'Kinondoni': 23,
        'Temeke': 24,
        'Kongwa': 25
    }  
    
    market_mapping = {
        'Arusha (urban)': 0,
        'Dar Es Salaam': 1,
        'Dodoma (Majengo)': 2,
        'Iringa Urban': 3,
        'Bukoba': 4,
        'Kigoma': 5,
        'Moshi': 6,
        'Lindi': 7,
        'Babati': 8,
        'Musoma': 9,
        'Mwanjelwa': 10,
        'Morogoro': 11,
        'Mtwara DC': 12,
        'Mwanza': 13,
        'Sumbawanga': 14,
        'Songea': 15,
        'Shinyanga': 16,
        'Singida': 17,
        'Tabora': 18,
        'Tanga / Mgandini': 19,
        'Mpanda': 20,
        'Njombe': 21,
        'Geita': 22,
        'Ilala (Buguruni)': 23,
        'Kinondoni (Tandale)': 24,
        'Temeke (Tandika)': 25,
        'Kibaigwa': 26,
        'Mbeya (SIDO)': 27,
        'Tanga': 28,
        'Sumbawanga (Katumba)': 29
    }
    
    commodity_mapping = {
        'Maize': 0,
        'Rice': 1,
        'Beans': 2,
    }      
    
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

    region_to_districts = {
        'Arusha': ['Arusha Urban'],
        'Dar-es-salaam': ['Ilala', 'Kinondoni', 'Temeke'],
        'Dodoma': ['Mpwapwa', 'Kongwa'],
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
        'Geita': ['Geita'],
    }
    
    region_to_markets = {
        'Arusha': ['Arusha (urban)'],
        'Dar-es-salaam': ['Dar Es Salaam', 'Ilala (Buguruni)', 'Kinondoni (Tandale)', 'Temeke (Tandika)'],
        'Dodoma': ['Dodoma (Majengo)', 'Kibaigwa'],
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
        'Geita': ['Geita'],
    }
    
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

    # Add this RIGHT BEFORE your button definition
    st.markdown("""
    <style>
        /* This will work with 100% certainty */
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
        
        /* Nuclear option - targets ALL buttons */
        button[kind="secondary"] {
            background-color: #4CAF50 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Then your button code with unique key:
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
        formatted_prediction = '{:,.2f} TZS'.format(prediction)

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

    # Centered contact information
    st.markdown("""
        <div style='text-align: center;'>
            <hr style='border: 1px solid #e1e4e8; margin: 20px 0;'>
            <h3>Mkulima Consultation System</h3>
            <p>Address Line: P. O. Box 34675, DSM</p>
            <p>Email Address: mkulimaconsaltation@gmail.com</p>
            <p>Phone Number: +225 672 410 645 / +255 712 410 690</p>
        </div>
    """, unsafe_allow_html=True)

# Only call this if the script is run directly, not when imported
if __name__ == "__main__":
    create_prediction_model()
