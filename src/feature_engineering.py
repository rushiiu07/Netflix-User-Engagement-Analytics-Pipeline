"""
Feature engineering module for Netflix User Engagement data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from datetime import datetime
import logging
from pathlib import Path

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()
        self.scalers = {}

    def _setup_logger(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def create_temporal_features(self, df):
        """
        Create time-based features from date columns
        """
        try:
            df_copy = df.copy()
            
            # Convert date columns to datetime
            df_copy['join_date'] = pd.to_datetime(df_copy['join_date'])
            df_copy['last_payment_date'] = pd.to_datetime(df_copy['last_payment_date'])
            current_date = datetime.now()

            # Calculate subscription length
            df_copy['subscription_length_days'] = (
                df_copy['last_payment_date'] - df_copy['join_date']
            ).dt.days

            # Calculate days since last payment
            df_copy['days_since_last_payment'] = (
                pd.Timestamp.now() - df_copy['last_payment_date']
            ).dt.days

            # Extract additional temporal features
            df_copy['join_month'] = df_copy['join_date'].dt.month
            df_copy['join_day_of_week'] = df_copy['join_date'].dt.dayofweek
            df_copy['join_quarter'] = df_copy['join_date'].dt.quarter

            return df_copy

        except Exception as e:
            self.logger.error(f"Error creating temporal features: {str(e)}")
            raise

    def encode_categorical_features(self, df):
        """
        Encode categorical variables using the specified method
        """
        try:
            df_copy = df.copy()
            
            for field in self.config['VALIDATION_RULES']['categorical_fields']:
                if field in df_copy.columns:
                    if self.config['FEATURE_CONFIG']['categorical_encoding'] == 'one_hot':
                        # One-hot encoding
                        encoded = pd.get_dummies(
                            df_copy[field], 
                            prefix=field, 
                            drop_first=True
                        )
                        df_copy = pd.concat([df_copy, encoded], axis=1)
                        df_copy.drop(columns=[field], inplace=True)
                    
                    elif self.config['FEATURE_CONFIG']['categorical_encoding'] == 'label':
                        # Label encoding
                        df_copy[field] = pd.Categorical(df_copy[field]).codes

            return df_copy

        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            raise

    def scale_numeric_features(self, df):
        """
        Scale numeric features using the specified method
        """
        try:
            df_copy = df.copy()
            numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns

            # Choose scaler based on configuration
            if self.config['FEATURE_CONFIG']['scaling_method'] == 'standard':
                scaler = StandardScaler()
            elif self.config['FEATURE_CONFIG']['scaling_method'] == 'minmax':
                scaler = MinMaxScaler()
            elif self.config['FEATURE_CONFIG']['scaling_method'] == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.config['FEATURE_CONFIG']['scaling_method']}")

            # Scale numeric features
            scaled_features = scaler.fit_transform(df_copy[numeric_columns])
            df_copy[numeric_columns] = scaled_features
            
            # Store scaler for future use
            self.scalers['numeric'] = scaler

            return df_copy

        except Exception as e:
            self.logger.error(f"Error scaling numeric features: {str(e)}")
            raise

    def create_interaction_features(self, df):
        """
        Create interaction features between numeric columns
        """
        try:
            df_copy = df.copy()
            numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns

            # Create interactions between numeric features
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    interaction_name = f"{col1}_x_{col2}"
                    df_copy[interaction_name] = df_copy[col1] * df_copy[col2]

            return df_copy

        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            raise

    def engineer_features(self, df):
        """
        Apply full feature engineering pipeline
        """
        try:
            # Create temporal features
            df_transformed = self.create_temporal_features(df)
            
            # Encode categorical features
            df_transformed = self.encode_categorical_features(df_transformed)
            
            # Scale numeric features
            df_transformed = self.scale_numeric_features(df_transformed)
            
            # Create interaction features
            df_transformed = self.create_interaction_features(df_transformed)
            
            return df_transformed

        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise

    def save_engineered_features(self, df):
        """
        Save engineered features and feature engineering artifacts
        """
        try:
            # Create feature store directory if it doesn't exist
            feature_store_dir = Path(self.config['DATA_CONFIG']['feature_store_path'])
            feature_store_dir.mkdir(parents=True, exist_ok=True)

            # Save engineered features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            features_path = feature_store_dir / f"engineered_features_{timestamp}.parquet"
            df.to_parquet(features_path)

            # Save scalers and other artifacts
            artifacts_path = feature_store_dir / f"engineering_artifacts_{timestamp}.pkl"
            pd.to_pickle(self.scalers, artifacts_path)

            self.logger.info(f"Saved engineered features to {features_path}")
            self.logger.info(f"Saved engineering artifacts to {artifacts_path}")

            return features_path, artifacts_path

        except Exception as e:
            self.logger.error(f"Error saving engineered features: {str(e)}")
            raise

if __name__ == "__main__":
    from config import DATA_CONFIG, VALIDATION_RULES, FEATURE_CONFIG
    
    # Initialize feature engineer
    config = {
        'DATA_CONFIG': DATA_CONFIG,
        'VALIDATION_RULES': VALIDATION_RULES,
        'FEATURE_CONFIG': FEATURE_CONFIG
    }
    feature_engineer = FeatureEngineer(config)

    # Read validated data
    df = pd.read_parquet(DATA_CONFIG['processed_data_path'] + "/validated_data_latest.parquet")

    # Apply feature engineering
    df_engineered = feature_engineer.engineer_features(df)
    
    # Save engineered features
    features_path, artifacts_path = feature_engineer.save_engineered_features(df_engineered)