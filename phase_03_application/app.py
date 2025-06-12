import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib
import yaml
import streamlit_authenticator as stauth
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from yaml.loader import SafeLoader

DATA_FOLDER = '../phase_01_data_collection/data'
MODELS_FOLDER = '../phase_02_model_building/models'

with open('./auth.yaml') as file:
  config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login(location='main')
name = st.session_state.get('name', '')
authentication_status = st.session_state.get('authentication_status')
username = st.session_state.get('username', '')

class CryptoPredictor:
    def __init__(self, data_path, crypto_name):
        self.data = pd.read_csv(data_path)
        self.crypto_name = crypto_name
        self.scaler = MinMaxScaler()
        
    def ensemble_predict(self, models, X, enabled_models=None):
        if enabled_models is None:
            enabled_models = ['linear', 'random_forest', 'lstm']
            
        X_2d = X.reshape(X.shape[0], -1)
        predictions = []
        
        if 'linear' in enabled_models:
            predictions.append(models['linear'].predict(X_2d))
        if 'random_forest' in enabled_models:
            predictions.append(models['random_forest'].predict(X_2d))
        if 'lstm' in enabled_models:
            predictions.append(models['lstm'].predict(X))
        
        ensemble_pred = sum(predictions) / len(predictions)
        
        return self.scaler.inverse_transform(ensemble_pred)
    
    def load_models_and_forecast(self, crypto_name, enabled_models=None):
        if enabled_models is None:
            enabled_models = ['linear', 'random_forest', 'lstm']
            
        forecasts = {}
        models = {}
        horizons = ['12h', '24h', '36h', '48h']
        
        latest_data = self.data.tail(24)['Close'].values
        latest_data = self.scaler.fit_transform(latest_data.reshape(-1, 1))
        latest_data = latest_data.reshape(1, 24, 1)
        
        for horizon in horizons:
            models[horizon] = {}
            if 'linear' in enabled_models:
                models[horizon]['linear'] = joblib.load(f'{MODELS_FOLDER}/{crypto_name}/linear_{horizon}.joblib')
            if 'random_forest' in enabled_models:
                models[horizon]['random_forest'] = joblib.load(f'{MODELS_FOLDER}/{crypto_name}/rf_{horizon}.joblib')
            if 'lstm' in enabled_models:
                models[horizon]['lstm'] = load_model(f'{MODELS_FOLDER}/{crypto_name}/lstm_{horizon}.h5')
            
            forecast = self.ensemble_predict(models[horizon], latest_data, enabled_models)
            forecasts[horizon] = forecast[0]
            
        return forecasts

class CryptoForecastApp:
    def __init__(self):
        self.currencies = [
          'btcusd', 'adausd', 'dogeusd', 'ethusd', 'solusd', 'xrpusd', 
          'usdtusd', 'usdcusd', 'daiusd', 'algousd', 'ltcusd'
        ]
        self.currencies_names = [
          'Bitcoin', 'Cardano', 'DogeCoin', 'Ethereum', 'Solana', 'XRP',
          'Tether', 'USDC', 'DAI', 'Algorand', 'Litecoin'
        ]
        self.horizons = ['12h', '24h', '36h', '48h']
        
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None
        if 'historical_data' not in st.session_state:
            st.session_state.historical_data = None
        if 'forecasts' not in st.session_state:
            st.session_state.forecasts = None
        if 'forecast_done' not in st.session_state:
            st.session_state.forecast_done = False
        if 'selected_currency' not in st.session_state:
            st.session_state.selected_currency = 'Select'
        if 'selected_horizon' not in st.session_state:
            st.session_state.selected_horizon = 'Select'
        if 'forecast_timestamps' not in st.session_state:
            st.session_state.forecast_timestamps = None
        if 'forecast_values' not in st.session_state:
            st.session_state.forecast_values = None
        if 'compare_mode' not in st.session_state:
            st.session_state.compare_mode = False
        if 'comparison_currencies' not in st.session_state:
            st.session_state.comparison_currencies = []
        if 'comparison_data' not in st.session_state:
            st.session_state.comparison_data = {}

    def setup_sidebar(self):
        st.sidebar.header('Select Parameters')
        
        st.session_state.compare_mode = st.sidebar.checkbox("Price Comparison Mode", value=st.session_state.compare_mode)
        
        if st.session_state.compare_mode:
            st.session_state.comparison_currencies = st.sidebar.multiselect(
                'Select Cryptocurrencies to Compare',
                self.currencies,
                format_func=lambda x: f'{self.currencies_names[self.currencies.index(x)]} ({x.upper()[:3]}/USD)'
            )
            
            compare_button = st.sidebar.button('Compare Prices', disabled=(len(st.session_state.comparison_currencies) == 0))
            
            return compare_button
        else:
            st.session_state.selected_currency = st.sidebar.selectbox(
                'Select Cryptocurrency',
                ['Select'] + self.currencies,
                format_func=lambda x: 'Select' if x == 'Select' else f'{self.currencies_names[self.currencies.index(x)]} ({x.upper()[:3]}/USD)'
            )
            
            st.session_state.selected_horizon = st.sidebar.selectbox(
                'Select Forecast Horizon',
                ['Select'] + self.horizons,
                format_func=lambda x: 'Select' if x == 'Select' else f"{x} hours"
            )
            
            forecast_button = st.sidebar.button('Forecast', 
                disabled=(st.session_state.selected_currency == 'Select' or st.session_state.selected_horizon == 'Select'))
            
            return forecast_button

    def setup_logout(self):
        st.sidebar.markdown("---")
        authenticator.logout("Logout", "sidebar")    

    def setup_additional_config(self):
        if not st.session_state.compare_mode and st.session_state.historical_data is not None:            
            st.sidebar.header('Additional Configuration')
            min_date = st.session_state.historical_data['Timestamp'].min()
            max_date = st.session_state.historical_data['Timestamp'].max()
            default_start = max_date - timedelta(weeks=2)
            
            start_date = st.sidebar.date_input(
                "Select Start Date for Historical Data",
                value=default_start,
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            st.session_state.start_date = start_date
            
        elif st.session_state.compare_mode and st.session_state.comparison_data:
            st.sidebar.header('Additional Configuration')
            
            min_dates = []
            max_dates = []
            for currency, data in st.session_state.comparison_data.items():
                if 'Timestamp' in data.columns and not data.empty:
                    min_dates.append(data['Timestamp'].min())
                    max_dates.append(data['Timestamp'].max())
            
            if min_dates and max_dates:
                normalize = st.sidebar.checkbox("Normalize prices (% change)", value=False, key='normalize_checkbox')
                st.session_state.normalize_prices = normalize
                
                min_date = min(min_dates)
                max_date = max(max_dates)
                default_start = max_date - timedelta(weeks=2)
                
                start_date = st.sidebar.date_input(
                    "Start Date for Comparison",
                    value=default_start,
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                end_date = st.sidebar.date_input(
                    "End Date for Comparison",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                if end_date < start_date:
                    st.sidebar.error("End date must be after start date.")
                    end_date = start_date
                
                st.session_state.comparison_start_date = start_date
                st.session_state.comparison_end_date = end_date

    def display_metrics(self, current_price, last_forecast):
        with st.spinner('Loading metrics...'):
            price_change = ((last_forecast - current_price) / current_price) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Forecasted Price", f"${last_forecast:.2f}")
            col3.metric("Expected Change", f"{price_change:.4f}%")
        
    def plot_chart(self):
        with st.spinner('Generating chart...'):
            start_date = st.session_state.get('start_date', None)
            plot_data = st.session_state.historical_data.copy()
            if start_date:
                plot_data = plot_data[plot_data['Timestamp'].dt.date >= start_date]
                
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=plot_data['Timestamp'],
                y=plot_data['Close'],
                name='Historical',
                line=dict(color='blue')
            ))

            last_historical_timestamp = plot_data['Timestamp'].iloc[-1]
            last_historical_price = plot_data['Close'].iloc[-1]
            
            forecast_x = [last_historical_timestamp] + st.session_state.forecast_timestamps
            forecast_y = [last_historical_price] + st.session_state.forecast_values.tolist()
            
            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                name='Forecast',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{self.currencies_names[self.currencies.index(st.session_state.selected_currency)]} ({st.session_state.selected_currency.upper()[:3]}/USD) Price History and Forecast',
                xaxis_title='Time',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    def display_data_tables(self):
        with st.spinner('Loading historical data...'):
            st.subheader('Historical Data')
            timeframes = {
                '1 Week': 7,
                '2 Weeks': 14,
                '1 Month': 30,
                'All Time': None
            }
            selected_timeframe = st.selectbox('Select Historical Data Range', list(timeframes.keys()))
            
            if timeframes[selected_timeframe]:
                display_data = st.session_state.historical_data.tail(timeframes[selected_timeframe] * 48).copy()
            else:
                display_data = st.session_state.historical_data.copy()
                
            display_data['Timestamp'] = display_data['Timestamp'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
            )
            st.dataframe(display_data.iloc[::-1])
        
        with st.spinner('Loading forecast data...'):
            st.subheader('Forecast Values')
            forecast_df = pd.DataFrame({
                'Timestamp': st.session_state.forecast_timestamps,
                'Predicted Price': np.round(st.session_state.forecast_values, 6)
            })
            forecast_df['Timestamp'] = forecast_df['Timestamp'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
            )
            st.dataframe(forecast_df)
        
    def display_about(self):
        with open('ReadMe.md', 'r') as f: readme = f.read()
        st.markdown(readme)
        
    def make_forecast(self):
        with st.spinner('Loading models and making predictions...'):
            st.session_state.predictor = CryptoPredictor(f'{DATA_FOLDER}/{st.session_state.selected_currency}.csv', st.session_state.selected_currency)
            st.session_state.forecasts = st.session_state.predictor.load_models_and_forecast(st.session_state.selected_currency, ['linear', 'lstm'])
            
            st.session_state.historical_data = st.session_state.predictor.data.copy()
            st.session_state.historical_data['Timestamp'] = pd.to_datetime(st.session_state.historical_data['Timestamp'], unit='s')
            
            last_timestamp = st.session_state.historical_data['Timestamp'].iloc[-1]
            forecast_hours = int(st.session_state.selected_horizon[:-1])
            forecast_intervals = forecast_hours * 2
            
            st.session_state.forecast_timestamps = [
                last_timestamp + timedelta(minutes=30 * (i+1))
                for i in range(forecast_intervals)
            ]
            
            st.session_state.forecast_values = st.session_state.forecasts[st.session_state.selected_horizon]
            st.session_state.forecast_done = True
        
    def load_comparison_data(self):
        with st.spinner('Loading comparison data...'):
            comparison_data = {}
            for currency in st.session_state.comparison_currencies:
                try:
                    data = pd.read_csv(f'{DATA_FOLDER}/{currency}.csv')
                    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
                    comparison_data[currency] = data
                except Exception as e:
                    st.error(f"Error loading data for {currency}: {str(e)}")
            
            st.session_state.comparison_data = comparison_data
            return comparison_data

    def plot_comparison_chart(self):
        with st.spinner('Generating comparison chart...'):
            if not st.session_state.comparison_currencies:
                st.warning("Please select at least one cryptocurrency to compare")
                return
            
            normalize = st.session_state.get('normalize_prices', False)
            
            start_date = st.session_state.get('comparison_start_date', None)
            end_date = st.session_state.get('comparison_end_date', None)
            
            fig = go.Figure()
            
            for currency in st.session_state.comparison_currencies:
                if currency in st.session_state.comparison_data:
                    plot_data = st.session_state.comparison_data[currency].copy()
                    
                    if start_date:
                        plot_data = plot_data[plot_data['Timestamp'].dt.date >= start_date]
                    
                    if end_date:
                        plot_data = plot_data[plot_data['Timestamp'].dt.date <= end_date]
                    
                    if plot_data.empty:
                        continue
                        
                    if currency in self.currencies:
                        currency_index = self.currencies.index(currency)
                        display_name = f'{self.currencies_names[currency_index]} ({currency.upper()[:3]})'
                    else:
                        display_name = currency.upper()
                    
                    y_values = plot_data['Close']
                    
                    if normalize and len(y_values) > 0:
                        base_value = y_values.iloc[0]
                        if base_value != 0:
                            y_values = [(val / base_value - 1) * 100 for val in y_values]

                    fig.add_trace(go.Scatter(
                        x=plot_data['Timestamp'],
                        y=y_values,
                        name=display_name,
                        mode='lines'
                    ))
            
            if len(fig.data) > 0:
                y_axis_title = "Percentage Change (%)" if normalize else "Price (USD)"
                
                title = 'Cryptocurrency Price Comparison'
                if start_date and end_date:
                    title += f' ({start_date} to {end_date})'
                elif start_date:
                    title += f' (from {start_date})'
                
                fig.update_layout(
                    title=title,
                    xaxis_title='Time',
                    yaxis_title=y_axis_title,
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected cryptocurrencies and date range")
        
    def run(self):
        if authentication_status is False:
            st.error("Username/password is incorrect")
        elif authentication_status is None:
            st.warning("Please enter your username and password")
        elif authentication_status:
            st.title('CryptoTrend Forecasting App')
            
            sidebar_button = self.setup_sidebar()
            
            if not st.session_state.compare_mode:
                st.session_state.forecast_done = False
                if sidebar_button and st.session_state.selected_currency != 'Select' and st.session_state.selected_horizon != 'Select':
                    try:
                        self.make_forecast()
                    except Exception as e:
                        st.error(f"Error loading models or making predictions: {str(e)}")
                        st.info("Please make sure you have trained the models first by running the training script.")
                        st.session_state.forecast_done = False
                
                if not st.session_state.forecast_done:
                    self.setup_logout()
                    tab1 = st.tabs(["ℹ️ About"])[0]
                    with tab1:
                        self.display_about()
                else:
                    self.setup_additional_config()
                    self.setup_logout()
                    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Data Tables", "ℹ️ About"])
                    
                    with tab1:
                        st.subheader('Current Statistics')
                        current_price = st.session_state.historical_data['Close'].iloc[-1]
                        last_forecast = st.session_state.forecast_values[-1]
                        
                        self.display_metrics(current_price, last_forecast)
                        self.plot_chart()
                    
                    with tab2:
                        self.display_data_tables()
                        
                    with tab3:
                        self.display_about()
            else:                
                if sidebar_button and st.session_state.comparison_currencies:
                    self.load_comparison_data()
                    st.session_state.forecast_done = True
                
                if not st.session_state.forecast_done or not st.session_state.comparison_currencies:
                    self.setup_logout()
                    tab1 = st.tabs(["ℹ️ About"])[0]
                    with tab1:
                        st.info("Select cryptocurrencies from the sidebar and click 'Compare Prices' to visualize price comparisons")
                        self.display_about()
                else:
                    self.setup_additional_config()
                    self.setup_logout()
                    tab1, tab2 = st.tabs(["📈 Comparison Chart", "ℹ️ About"])
                    with tab1:
                        st.subheader('Cryptocurrency Price Comparison')
                        self.plot_comparison_chart()
                    with tab2:
                        self.display_about()

if __name__ == "__main__":
    app = CryptoForecastApp()
    app.run()