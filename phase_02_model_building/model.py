import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import numpy as np
import argparse
from typing import List, Optional

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

DATA_FOLDER = '../phase_01_data_collection/data'

class CryptoPredictor:
    def __init__(self, data_path, crypto_name):
        self.data = pd.read_csv(data_path)
        self.crypto_name = crypto_name
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, lookback=24, forecast_horizon=24):
        # to convert timestamps and sort data
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'], unit='s')
        self.data = self.data.sort_values('Timestamp')
        
        # to create features and target
        X = []
        y = []
        
        scaled_data = self.scaler.fit_transform(self.data[['Close']])
        
        for i in range(len(scaled_data) - lookback - forecast_horizon + 1):
            X.append(scaled_data[i:(i + lookback)])
            y.append(scaled_data[i + lookback:i + lookback + forecast_horizon])
            
        return np.array(X), np.array(y)
    
    def train_linear_regression(self, X_train, y_train):
        # to reshape data for linear regression
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        model = LinearRegression()
        model.fit(X_train_2d, y_train.reshape(y_train.shape[0], -1))
        return model
    
    def train_random_forest(self, X_train, y_train):
        # to reshape data for random forest
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor()
        model = GridSearchCV(rf, param_grid, cv=3, verbose=3)
        model.fit(X_train_2d, y_train.reshape(y_train.shape[0], -1))
        return model
    
    def create_lstm_model(self, input_shape, output_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_lstm(self, X_train, y_train):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model = self.create_lstm_model(
            (X_train.shape[1], X_train.shape[2]),
            y_train.shape[1]
        )
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=2
        )
        return model
    
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
            
        # to take average only the enabled models
        ensemble_pred = sum(predictions) / len(predictions)
        
        return self.scaler.inverse_transform(ensemble_pred)
    
    def train_and_predict(self, forecast_horizons=[24, 48, 72, 96], enabled_models=None):  # 12h, 24h, 36h, 48h
        if enabled_models is None:
            enabled_models = ['linear', 'random_forest', 'lstm']
            
        results = {}
        models = {}
        
        for horizon in forecast_horizons:
            X, y = self.prepare_data(lookback=24, forecast_horizon=horizon)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # to initialize models dictionary for this horizon
            models[f'{horizon//2}h'] = {}
            
            # to train only enabled models
            if 'linear' in enabled_models:
                models[f'{horizon//2}h']['linear'] = self.train_linear_regression(X_train, y_train)
            if 'random_forest' in enabled_models:
                models[f'{horizon//2}h']['random_forest'] = self.train_random_forest(X_train, y_train)
            if 'lstm' in enabled_models:
                models[f'{horizon//2}h']['lstm'] = self.train_lstm(X_train, y_train)
            
            # to make predictions using only enabled models
            predictions = self.ensemble_predict(models[f'{horizon//2}h'], X_test, enabled_models)
            results[f'{horizon//2}h'] = predictions
            
        return models, results

    def load_models_and_forecast(self, crypto_name, enabled_models=None):
        """Load saved models and make forecasts for all time horizons"""
        if enabled_models is None:
            enabled_models = ['linear', 'random_forest', 'lstm']
            
        forecasts = {}
        models = {}
        horizons = ['12h', '24h', '36h', '48h']
        
        # to prepare latest data for prediction
        latest_data = self.data.tail(24)['Close'].values
        latest_data = self.scaler.fit_transform(latest_data.reshape(-1, 1))
        latest_data = latest_data.reshape(1, 24, 1)  # reshaped for LSTM input
        
        for horizon in horizons:
            # to load only enabled models for this horizon
            models[horizon] = {}
            if 'linear' in enabled_models:
                models[horizon]['linear'] = joblib.load(f'models/{crypto_name}/linear_{horizon}.joblib')
            if 'random_forest' in enabled_models:
                models[horizon]['random_forest'] = joblib.load(f'models/{crypto_name}/rf_{horizon}.joblib')
            if 'lstm' in enabled_models:
                models[horizon]['lstm'] = load_model(f'models/{crypto_name}/lstm_{horizon}.h5')
            
            # to make ensemble prediction using only enabled models
            forecast = self.ensemble_predict(models[horizon], latest_data, enabled_models)
            forecasts[horizon] = forecast[0]
            
        return forecasts

def main(enabled_models: Optional[List[str]] = None):
    """
    Main function to train and save models for all cryptocurrencies
    Args:
        enabled_models: List of models to train/use. Options: ['linear', 'random_forest', 'lstm']
                        If None, all models will be used
    """
    # to validate enabled_models
    if enabled_models is None:
        enabled_models = ['linear', 'random_forest', 'lstm']
    else:
        valid_models = ['linear', 'random_forest', 'lstm']
        enabled_models = [model for model in enabled_models if model in valid_models]
        if not enabled_models:
            raise ValueError("No valid models specified. Choose from: 'linear', 'random_forest', 'lstm'")

    # to create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    cryptocurrencies = [
      'btcusd', 'adausd', 'dogeusd', 'ethusd', 'solusd', 'xrpusd', 
      'usdtusd', 'usdcusd', 'daiusd', 'algousd', 'ltcusd'
    ]
    
    # the training phase
    for crypto in cryptocurrencies:
        print(f"\nTraining models for {crypto}...")
        print(f"Enabled models: {enabled_models}")
        
        # to initialize predictor
        predictor = CryptoPredictor(f'{DATA_FOLDER}/{crypto}.csv', crypto)
        
        # to train models and get predictions
        models, predictions = predictor.train_and_predict(enabled_models=enabled_models)
        
        # to create directory for this cryptocurrency
        crypto_dir = f'models/{crypto}'
        if not os.path.exists(crypto_dir):
            os.makedirs(crypto_dir)
        
        # to save only enabled models
        for horizon, horizon_models in models.items():
            if 'lstm' in enabled_models and 'lstm' in horizon_models:
                save_model(horizon_models['lstm'], f'{crypto_dir}/lstm_{horizon}.h5')
            
            if 'linear' in enabled_models and 'linear' in horizon_models:
                joblib.dump(horizon_models['linear'], f'{crypto_dir}/linear_{horizon}.joblib')
                
            if 'random_forest' in enabled_models and 'random_forest' in horizon_models:
                joblib.dump(horizon_models['random_forest'], f'{crypto_dir}/rf_{horizon}.joblib')
            
        print(f"Models for {crypto} saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save models for all cryptocurrencies")
    parser.add_argument(
        "--enabled_models",
        nargs="+",
        help="List of models to train/use. Options: ['linear', 'random_forest', 'lstm']"
    )
    args = parser.parse_args()
    
    main([model for model in args.enabled_models if model != 'random_forest'])