"""
Data quality assessment module for Netflix User Engagement data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Configure warnings
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class DataQualityChecker:
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

    def check_missing_values(self, df):
        """
        Check for missing values in each column
        """
        missing_stats = {
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'columns_above_threshold': []
        }

        threshold = self.config['QUALITY_THRESHOLDS']['missing_threshold']
        for col, pct in missing_stats['missing_percentages'].items():
            if pct > threshold * 100:
                missing_stats['columns_above_threshold'].append({
                    'column': col,
                    'missing_percentage': pct
                })

        return missing_stats

    def check_duplicates(self, df):
        """
        Check for duplicate records
        """
        duplicates = df.duplicated()
        duplicate_stats = {
            'total_duplicates': duplicates.sum(),
            'duplicate_percentage': (duplicates.sum() / len(df) * 100),
            'above_threshold': False
        }

        threshold = self.config['QUALITY_THRESHOLDS']['duplicate_threshold']
        if duplicate_stats['duplicate_percentage'] > threshold * 100:
            duplicate_stats['above_threshold'] = True

        return duplicate_stats

    def _identify_data_source(self, df):
        """
        Identify the data source based on column structure
        """
        # First try to match by required fields
        for source, rules in self.config['VALIDATION_RULES'].items():
            required_fields = set(rules['required_fields'])
            if required_fields.issubset(set(df.columns)):
                return source
        
        # If no match by required fields, try to match by column pattern
        column_patterns = {
            'users': ['user_id', 'country', 'age'],
            'movies': ['movie_id', 'title'],
            'watch_history': ['user_id', 'movie_id', 'watch_date'],
            'reviews': ['user_id', 'movie_id', 'rating'],
            'search_logs': ['user_id', 'search_query'],
            'recommendation_logs': ['user_id', 'movie_id', 'recommendation_date']
        }
        
        for source, pattern in column_patterns.items():
            if any(col in df.columns for col in pattern):
                return source
                
        return None
        
    def enrich_missing_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Enrich missing data using available information
        """
        df = df.copy()
        
        if source == 'users':
            # Enrich subscription_type if missing
            if 'subscription_type' not in df.columns:
                df['subscription_type'] = 'basic'
                self.logger.info("Added missing subscription_type with default 'basic'")
                
            # Enrich join_date if missing
            if 'join_date' not in df.columns:
                # Try to get earliest activity date from watch_history
                watch_history_path = Path(self.config['DATA_CONFIG']['raw_data']['watch_history'])
                if watch_history_path.exists():
                    try:
                        watch_df = pd.read_csv(watch_history_path)
                        watch_df['watch_date'] = pd.to_datetime(watch_df['watch_date'])
                        first_watches = watch_df.groupby('user_id')['watch_date'].min()
                        df['join_date'] = df['user_id'].map(first_watches)
                        self.logger.info("Enriched join_date using watch_history data")
                    except Exception as e:
                        self.logger.warning(f"Could not enrich join_date from watch_history: {str(e)}")
                        df['join_date'] = datetime.now()
                else:
                    df['join_date'] = datetime.now()
                    self.logger.warning("Added missing join_date with current timestamp")
                    
            # Standardize country codes
            if 'country' in df.columns:
                df['country'] = df['country'].str.upper()
                
            # Clean age data
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
                invalid_age = (df['age'] < 13) | (df['age'] > 100)
                df.loc[invalid_age, 'age'] = np.nan
                
        elif source == 'watch_history':
            # Add completion status if missing
            if 'completed' not in df.columns:
                movies_path = Path(self.config['DATA_CONFIG']['raw_data']['movies'])
                if movies_path.exists():
                    try:
                        movies_df = pd.read_csv(movies_path)
                        df = df.merge(movies_df[['movie_id', 'duration']], on='movie_id', how='left')
                        df['completed'] = (df['watch_duration'] >= df['duration'] * 0.9).fillna(False).astype(int)
                        df.drop('duration', axis=1, inplace=True)
                        self.logger.info("Added completion status based on watch duration")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate completion status: {str(e)}")
                        df['completed'] = 0
                else:
                    df['completed'] = 0
                    
        return df

    def check_outliers(self, df):
        """
        Check for outliers in numeric columns using z-score method
        """
        outlier_stats = {}
        std_threshold = self.config['QUALITY_THRESHOLDS']['outlier_std_threshold']

        # Identify the data source
        source = self._identify_data_source(df)
        
        if source:
            numeric_fields = self.config['VALIDATION_RULES'][source]['numeric_fields']
            for field in numeric_fields:
                if field in df.columns:
                    try:
                        series = pd.to_numeric(df[field], errors='coerce')
                        z_scores = np.abs((series - series.mean()) / series.std())
                        outliers = z_scores > std_threshold
                        outlier_stats[field] = {
                            'outlier_count': outliers.sum(),
                            'outlier_percentage': (outliers.sum() / len(df) * 100),
                            'outlier_indices': outliers[outliers].index.tolist()
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not check outliers for field {field}: {str(e)}")
                        outlier_stats[field] = {'error': str(e)}

        return outlier_stats

    def check_value_distributions(self, df):
        """
        Check value distributions for categorical fields
        """
        distribution_stats = {}
        
        # Identify source using helper method
        source = self._identify_data_source(df)

        if source:
            categorical_fields = self.config['VALIDATION_RULES'][source]['categorical_fields']
            for field in categorical_fields:
                if field in df.columns:
                    value_counts = df[field].value_counts()
                    distribution_stats[field] = {
                        'unique_values': len(value_counts),
                        'top_values': value_counts.head(5).to_dict(),
                        'distribution': (value_counts / len(df) * 100).to_dict()
                    }

        return distribution_stats

    def check_date_ranges(self, df):
        """
        Check date ranges and validity
        """
        date_stats = {}
        
        # Identify source and date fields
        source = self._identify_data_source(df)
        
        if source:
            date_fields = self.config['VALIDATION_RULES'][source]['date_fields']
            for field in date_fields:
                if field in df.columns:
                    try:
                        dates = pd.to_datetime(df[field], errors='coerce')
                        valid_dates = dates.dropna()
                        if len(valid_dates) > 0:
                            date_stats[field] = {
                                'min_date': valid_dates.min().strftime('%Y-%m-%d'),
                                'max_date': valid_dates.max().strftime('%Y-%m-%d'),
                                'invalid_dates': dates.isnull().sum(),
                                'unique_dates': len(valid_dates.unique())
                            }
                        else:
                            date_stats[field] = {
                                'error': 'No valid dates found in the column'
                            }
                    except Exception as e:
                        self.logger.error(f"Error processing dates in {field}: {str(e)}")
                        date_stats[field] = {'error': str(e)}

        return date_stats

    def generate_quality_report(self, df):
        """
        Generate comprehensive data quality report
        """
        quality_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': self.check_missing_values(df),
            'duplicates': self.check_duplicates(df),
            'outliers': self.check_outliers(df),
            'value_distributions': self.check_value_distributions(df),
            'date_ranges': self.check_date_ranges(df)
        }

        return quality_report

    def save_quality_report(self, quality_report):
        """
        Save data quality report
        """
        try:
            # Create quality reports directory if it doesn't exist
            report_dir = Path(self.config['DATA_CONFIG']['quality_reports_path'])
            report_dir.mkdir(parents=True, exist_ok=True)

            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"quality_report_{timestamp}.json"
            pd.Series(quality_report).to_json(report_path)

            self.logger.info(f"Saved quality report to {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Error saving quality report: {str(e)}")
            raise

if __name__ == "__main__":
    from config import DATA_CONFIG, VALIDATION_RULES, QUALITY_THRESHOLDS
    
    # Initialize quality checker
    config = {
        'DATA_CONFIG': DATA_CONFIG,
        'VALIDATION_RULES': VALIDATION_RULES,
        'QUALITY_THRESHOLDS': QUALITY_THRESHOLDS
    }
    quality_checker = DataQualityChecker(config)

    # Read validated data
    df = pd.read_parquet(DATA_CONFIG['processed_data_path'] + "/validated_data_latest.parquet")

    # Generate and save quality report
    quality_report = quality_checker.generate_quality_report(df)
    report_path = quality_checker.save_quality_report(quality_report)