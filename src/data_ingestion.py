"""
Data ingestion module for Netflix User Engagement data
"""
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def read_csv_data(self, file_path):
        """
        Read CSV data from the specified path
        """
        try:
            self.logger.info(f"Reading CSV file from {file_path}")
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully read {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {str(e)}")
            raise

    def read_json_data(self, file_path):
        """
        Read JSON data from the specified path
        """
        try:
            self.logger.info(f"Reading JSON file from {file_path}")
            df = pd.read_json(file_path)
            self.logger.info(f"Successfully read {len(df)} records")
            return df
        except Exception as e:
            self.logger.error(f"Error reading JSON file: {str(e)}")
            raise

    def validate_raw_data(self, df):
        """
        Perform initial validation on raw data
        """
        validation_results = {
            'total_records': len(df),
            'missing_fields': {},
            'invalid_formats': []
        }

        # Check required fields
        for field in self.config['VALIDATION_RULES']['required_fields']:
            if field not in df.columns:
                validation_results['invalid_formats'].append(
                    f"Missing required field: {field}"
                )
            else:
                missing_count = df[field].isnull().sum()
                if missing_count > 0:
                    validation_results['missing_fields'][field] = missing_count

        return validation_results

    def save_validated_data(self, df, validation_results):
        """
        Save validated data and validation results
        """
        try:
            # Create processed data directory if it doesn't exist
            processed_dir = Path(self.config['DATA_CONFIG']['processed_data_path'])
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = processed_dir / f"validated_data_{timestamp}.parquet"
            df.to_parquet(output_path)

            # Save validation results
            validation_path = processed_dir / f"validation_results_{timestamp}.json"
            pd.Series(validation_results).to_json(validation_path)

            self.logger.info(f"Saved validated data to {output_path}")
            self.logger.info(f"Saved validation results to {validation_path}")

            return output_path, validation_path

        except Exception as e:
            self.logger.error(f"Error saving validated data: {str(e)}")
            raise

if __name__ == "__main__":
    from config import DATA_CONFIG, VALIDATION_RULES

    # Initialize data ingestion
    config = {'DATA_CONFIG': DATA_CONFIG, 'VALIDATION_RULES': VALIDATION_RULES}
    ingestion = DataIngestion(config)

    # Read and validate data
    df = ingestion.read_csv_data(DATA_CONFIG['raw_data_path'])
    validation_results = ingestion.validate_raw_data(df)
    
    # Save validated data
    output_path, validation_path = ingestion.save_validated_data(
        df, validation_results
    )