"""
Module for processing data through bronze, silver, and gold layers
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from data_quality import DataQualityChecker
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class DataLakeProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def create_bronze_layer(self):
        """
        Bronze Layer: Raw data ingestion with minimal processing
        - Load raw CSV files
        - Add metadata columns
        - Basic data type conversions
        - Save in parquet format
        """
        bronze_data = {}
        
        for source, file_path in self.config['DATA_CONFIG']['raw_data'].items():
            try:
                # Read raw CSV
                df = pd.read_csv(file_path)
                
                # Validate and enrich data before bronze layer processing
                self.logger.info(f"Starting data validation and enrichment for {source}")
                
                # Add metadata columns
                df['ingestion_timestamp'] = datetime.now()
                df['data_source'] = source
                df['bronze_id'] = np.arange(len(df))
                
                # Apply data quality checks
                quality_checker = DataQualityChecker(self.config)
                quality_report = quality_checker.generate_quality_report(df)
                quality_checker.save_quality_report(quality_report)
                
                # Enrich missing data
                df = quality_checker.enrich_missing_data(df, source)
                
                # Validate and convert data types based on schema
                if source in self.config['SCHEMA_CONFIG']:
                    schema = self.config['SCHEMA_CONFIG'][source]
                    
                    # Convert date fields
                    for col in schema['date']:
                        if col in df.columns:
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                self.logger.info(f"Converted {col} to datetime in {source}")
                            except Exception as e:
                                self.logger.error(f"Error converting {col} to datetime in {source}: {str(e)}")
                                
                    # Convert numeric fields
                    for col in schema['numeric']:
                        if col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                self.logger.info(f"Converted {col} to numeric in {source}")
                            except Exception as e:
                                self.logger.error(f"Error converting {col} to numeric in {source}: {str(e)}")
                                
                    # Handle required and enriched fields
                    missing_required = [col for col in schema['required'] if col not in df.columns]
                    if missing_required:
                        self.logger.error(f"Missing strictly required columns in {source}: {missing_required}")
                        raise ValueError(f"Missing strictly required columns in {source}: {missing_required}")
                    
                    # Handle enriched fields
                    for col in schema.get('enriched', []):
                        if col not in df.columns:
                            if col == 'subscription_type':
                                df[col] = schema['defaults'].get('subscription_type', 'basic')
                            elif col == 'join_date':
                                # Use first watch date as join date if available
                                if 'watch_date' in df.columns:
                                    df[col] = df.groupby('user_id')['watch_date'].transform('min')
                                else:
                                    df[col] = pd.Timestamp.now()
                            self.logger.info(f"Enriched missing column {col} in {source}")
                    
                    # Add missing optional fields with default values
                    for col in schema['optional']:
                        if col not in df.columns:
                            default_value = schema.get('defaults', {}).get(col)
                            if default_value is not None:
                                df[col] = default_value
                            elif col in schema['numeric']:
                                df[col] = 0
                            elif col in schema['date']:
                                df[col] = pd.NaT
                            else:
                                df[col] = 'unknown'
                            self.logger.warning(f"Added missing optional column {col} to {source}")
                
                # Save to bronze layer
                bronze_path = Path(self.config['DATA_CONFIG']['bronze_layer'])
                bronze_path.mkdir(parents=True, exist_ok=True)
                output_path = bronze_path / f"{source}_bronze.parquet"
                df.to_parquet(output_path)
                
                bronze_data[source] = df
                self.logger.info(f"Processed {source} to bronze layer: {len(df)} records")
                
            except Exception as e:
                self.logger.error(f"Error processing {source} to bronze layer: {str(e)}")
                raise
                
        return bronze_data

    def create_silver_layer(self, bronze_data):
        """
        Silver Layer: Cleaned and standardized data
        - Data cleaning and validation
        - Handle missing values
        - Remove duplicates
        - Standardize formats
        - Create relationships between tables
        """
        silver_data = {}
        
        try:
            # Process Users table
            if 'users' in bronze_data:
                users_silver = bronze_data['users'].copy()
                users_silver['age'] = pd.to_numeric(users_silver['age'], errors='coerce')
                users_silver['country'] = users_silver['country'].str.upper()
                users_silver = users_silver.dropna(subset=['user_id'])
                users_silver = users_silver.drop_duplicates(subset=['user_id'])
                silver_data['users'] = users_silver

                # Process Movies table
            if 'movies' in bronze_data:
                movies_silver = bronze_data['movies'].copy()
                
                # Validate required columns exist
                required_columns = ['movie_id', 'title', 'release_year']
                missing_columns = [col for col in required_columns if col not in movies_silver.columns]
                if missing_columns:
                    self.logger.error(f"Missing required columns in movies data: {missing_columns}")
                    raise ValueError(f"Missing required columns in movies data: {missing_columns}")
                
                # Handle numeric conversions safely
                if 'release_year' in movies_silver.columns:
                    movies_silver['release_year'] = pd.to_numeric(movies_silver['release_year'], errors='coerce')
                
                if 'duration' in movies_silver.columns:
                    movies_silver['duration'] = pd.to_numeric(movies_silver['duration'], errors='coerce')
                else:
                    self.logger.warning("Duration column not found in movies data. Adding default duration column.")
                    movies_silver['duration'] = None  # Add column with NA values
                
                movies_silver = movies_silver.dropna(subset=['movie_id'])
                movies_silver = movies_silver.drop_duplicates(subset=['movie_id'])
                silver_data['movies'] = movies_silver            # Process Watch History
            if 'watch_history' in bronze_data:
                watch_silver = bronze_data['watch_history'].copy()
                
                # Filter for valid user and movie relationships
                if 'users' in silver_data and 'movies' in silver_data:
                    watch_silver = watch_silver[watch_silver['user_id'].isin(silver_data['users']['user_id'])]
                    watch_silver = watch_silver[watch_silver['movie_id'].isin(silver_data['movies']['movie_id'])]
                
                # Handle watch_duration field
                if 'watch_duration' not in watch_silver.columns:
                    self.logger.warning("watch_duration column not found in watch_history data. Adding default column.")
                    # Try to get duration from movies if available
                    if 'movies' in silver_data and 'duration' in silver_data['movies'].columns:
                        movie_durations = silver_data['movies'][['movie_id', 'duration']]
                        watch_silver = watch_silver.merge(movie_durations, on='movie_id', how='left')
                        watch_silver['watch_duration'] = watch_silver['duration']
                        watch_silver.drop('duration', axis=1, inplace=True)
                    else:
                        watch_silver['watch_duration'] = None
                else:
                    watch_silver['watch_duration'] = pd.to_numeric(watch_silver['watch_duration'], errors='coerce')
                
                # Handle completed field
                if 'completed' not in watch_silver.columns:
                    self.logger.warning("completed column not found in watch_history data. Adding default column.")
                    # Consider a watch completed if duration matches movie duration (if available)
                    if ('movies' in silver_data and 'duration' in silver_data['movies'].columns 
                        and 'watch_duration' in watch_silver.columns):
                        movie_durations = silver_data['movies'][['movie_id', 'duration']]
                        if not watch_silver['watch_duration'].isna().all():  # Only if we have some valid durations
                            watch_silver = watch_silver.merge(movie_durations, on='movie_id', how='left')
                            watch_silver['completed'] = (
                                watch_silver['watch_duration'] >= watch_silver['duration'] * 0.9
                            ).fillna(False).astype(int)
                            watch_silver.drop('duration', axis=1, inplace=True)
                        else:
                            watch_silver['completed'] = 0
                    else:
                        watch_silver['completed'] = 0
                
                silver_data['watch_history'] = watch_silver

            # Save silver layer data
            silver_path = Path(self.config['DATA_CONFIG']['silver_layer'])
            silver_path.mkdir(parents=True, exist_ok=True)
            
            for source, df in silver_data.items():
                output_path = silver_path / f"{source}_silver.parquet"
                df.to_parquet(output_path)
                self.logger.info(f"Processed {source} to silver layer: {len(df)} records")

        except Exception as e:
            self.logger.error(f"Error processing silver layer: {str(e)}")
            raise
            
        return silver_data

    def create_gold_layer(self, silver_data):
        """
        Gold Layer: Business-level aggregations and KPIs
        - Calculate key metrics
        - Create aggregated views
        - Generate KPI tables
        """
        if not silver_data:
            self.logger.error("No silver layer data provided")
            return {}
            
        gold_data = {}
        try:
            gold_path = Path(self.config['DATA_CONFIG']['gold_layer'])
            gold_path.mkdir(parents=True, exist_ok=True)

            # Dictionary to track KPI generation status
            kpi_status = {
                'user_engagement': False,
                'content_performance': False,
                'recommendation_effectiveness': False,
                'user_retention': False
            }

            # 1. User Engagement KPIs
            try:
                user_engagement = self._calculate_user_engagement_kpis(silver_data)
                if not user_engagement.empty:
                    user_engagement.to_parquet(gold_path / "user_engagement_kpis.parquet")
                    gold_data['user_engagement'] = user_engagement
                    kpi_status['user_engagement'] = True
                    self.logger.info("Successfully generated user engagement KPIs")
            except Exception as e:
                self.logger.error(f"Error generating user engagement KPIs: {str(e)}")

            # 2. Content Performance KPIs
            try:
                content_performance = self._calculate_content_performance_kpis(silver_data)
                if not content_performance.empty:
                    content_performance.to_parquet(gold_path / "content_performance_kpis.parquet")
                    gold_data['content_performance'] = content_performance
                    kpi_status['content_performance'] = True
                    self.logger.info("Successfully generated content performance KPIs")
            except Exception as e:
                self.logger.error(f"Error generating content performance KPIs: {str(e)}")

            # 3. Recommendation Effectiveness KPIs
            try:
                recommendation_kpis = self._calculate_recommendation_kpis(silver_data)
                if not recommendation_kpis.empty:
                    recommendation_kpis.to_parquet(gold_path / "recommendation_kpis.parquet")
                    gold_data['recommendation_kpis'] = recommendation_kpis
                    kpi_status['recommendation_effectiveness'] = True
                    self.logger.info("Successfully generated recommendation KPIs")
            except Exception as e:
                self.logger.error(f"Error generating recommendation KPIs: {str(e)}")

            # 4. User Retention KPIs
            try:
                retention_kpis = self._calculate_retention_kpis(silver_data)
                if not retention_kpis.empty:
                    retention_kpis.to_parquet(gold_path / "retention_kpis.parquet")
                    gold_data['retention_kpis'] = retention_kpis
                    kpi_status['user_retention'] = True
                    self.logger.info("Successfully generated retention KPIs")
            except Exception as e:
                self.logger.error(f"Error generating retention KPIs: {str(e)}")

            # Log overall KPI generation status
            successful_kpis = [k for k, v in kpi_status.items() if v]
            failed_kpis = [k for k, v in kpi_status.items() if not v]
            
            if successful_kpis:
                self.logger.info(f"Successfully generated KPIs for: {', '.join(successful_kpis)}")
            if failed_kpis:
                self.logger.warning(f"Failed to generate KPIs for: {', '.join(failed_kpis)}")
                
            # Create summary statistics
            try:
                self._create_gold_layer_summary(gold_data, gold_path)
            except Exception as e:
                self.logger.error(f"Error creating gold layer summary: {str(e)}")
            
            return gold_data

        except Exception as e:
            self.logger.error(f"Error processing gold layer: {str(e)}")
            raise

    def _calculate_user_engagement_kpis(self, silver_data):
        """Calculate user engagement metrics"""
        watch_history = silver_data.get('watch_history', pd.DataFrame())
        users = silver_data.get('users', pd.DataFrame())
        
        if watch_history.empty or users.empty:
            self.logger.warning("Empty watch history or users data for engagement KPIs")
            return pd.DataFrame()
            
        # Ensure we have user_id column
        if 'user_id' not in watch_history.columns:
            self.logger.error("user_id column missing from watch_history")
            return pd.DataFrame()
            
        # Create base user metrics DataFrame from users table
        user_metrics = users[['user_id']].copy()
        
        # Calculate watch counts for all users
        watch_counts = watch_history.groupby('user_id').size().reset_index(name='total_watches')
        user_metrics = user_metrics.merge(watch_counts, on='user_id', how='left')
        user_metrics['total_watches'] = user_metrics['total_watches'].fillna(0)
        
        # Add duration-based metrics if available
        if 'watch_duration' in watch_history.columns:
            duration_metrics = watch_history.groupby('user_id').agg({
                'watch_duration': ['sum', 'mean']
            }).reset_index()
            duration_metrics.columns = ['user_id', 'total_duration', 'avg_duration']
            user_metrics = user_metrics.merge(duration_metrics, on='user_id', how='left')
            user_metrics[['total_duration', 'avg_duration']] = user_metrics[['total_duration', 'avg_duration']].fillna(0)
            
        # Add completion metrics if available
        if 'completed' in watch_history.columns:
            completion_metrics = watch_history.groupby('user_id')['completed'].sum().reset_index(name='completed_watches')
            user_metrics = user_metrics.merge(completion_metrics, on='user_id', how='left')
            user_metrics['completed_watches'] = user_metrics['completed_watches'].fillna(0)
        
        # Add available user demographics
        for col in ['country', 'age', 'subscription_type']:
            if col in users.columns:
                user_metrics = user_metrics.merge(
                    users[['user_id', col]], 
                    on='user_id', 
                    how='left'
                )
        
        # Calculate engagement scores
        user_metrics['engagement_score'] = 0
        weights = {'watch_weight': 0.3, 'completion_weight': 0.4, 'duration_weight': 0.3}
        
        # Normalize watch counts for scoring
        if 'total_watches' in user_metrics.columns:
            max_watches = user_metrics['total_watches'].max()
            if max_watches > 0:  # Avoid division by zero
                watch_score = (user_metrics['total_watches'] / max_watches) * weights['watch_weight']
                user_metrics['engagement_score'] += watch_score
        
        # Add completion rate component if available
        if all(col in user_metrics.columns for col in ['completed_watches', 'total_watches']):
            completion_rate = user_metrics['completed_watches'] / user_metrics['total_watches'].replace(0, 1)
            user_metrics['engagement_score'] += completion_rate * weights['completion_weight']
        
        # Add duration component if available
        if 'avg_duration' in user_metrics.columns:
            max_duration = user_metrics['avg_duration'].max()
            if max_duration > 0:  # Avoid division by zero
                duration_score = (user_metrics['avg_duration'] / max_duration) * weights['duration_weight']
                user_metrics['engagement_score'] += duration_score
        
        # Ensure all numeric columns have valid values
        numeric_columns = ['total_watches', 'total_duration', 'avg_duration', 
                         'completed_watches', 'engagement_score']
        for col in numeric_columns:
            if col in user_metrics.columns:
                user_metrics[col] = user_metrics[col].fillna(0)
        
        self.logger.info(f"Generated engagement KPIs for {len(user_metrics)} users")
        
        return user_metrics
        
        return user_metrics

    def _calculate_content_performance_kpis(self, silver_data):
        """Calculate content performance metrics"""
        watch_history = silver_data.get('watch_history', pd.DataFrame())
        movies = silver_data.get('movies', pd.DataFrame())
        reviews = silver_data.get('reviews', pd.DataFrame())
        
        if watch_history.empty or movies.empty:
            self.logger.warning("Empty watch history or movies data for content KPIs")
            return pd.DataFrame()
            
        # Start with base movie metrics from movies table
        content_kpis = movies[['movie_id']].copy()
        
        # Add basic watch count metrics
        watch_counts = watch_history.groupby('movie_id').size().reset_index(name='total_watches')
        content_kpis = content_kpis.merge(watch_counts, on='movie_id', how='left')
        content_kpis['total_watches'] = content_kpis['total_watches'].fillna(0)
        
        # Add watch duration metrics if available
        if 'watch_duration' in watch_history.columns:
            duration_metrics = watch_history.groupby('movie_id').agg({
                'watch_duration': ['sum', 'mean']
            }).reset_index()
            duration_metrics.columns = ['movie_id', 'total_duration', 'avg_duration']
            content_kpis = content_kpis.merge(duration_metrics, on='movie_id', how='left')
            content_kpis[['total_duration', 'avg_duration']] = content_kpis[['total_duration', 'avg_duration']].fillna(0)
        
        # Add completion metrics if available
        if 'completed' in watch_history.columns:
            completion_metrics = watch_history.groupby('movie_id')['completed'].agg([
                ('completed_watches', 'sum'),
                ('completion_rate', 'mean')
            ]).reset_index()
            content_kpis = content_kpis.merge(completion_metrics, on='movie_id', how='left')
            content_kpis[['completed_watches', 'completion_rate']] = content_kpis[['completed_watches', 'completion_rate']].fillna(0)
        
        # Add review metrics if available
        if not reviews.empty and 'rating' in reviews.columns:
            review_metrics = reviews.groupby('movie_id').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            review_metrics.columns = ['movie_id', 'total_reviews', 'avg_rating']
            content_kpis = content_kpis.merge(review_metrics, on='movie_id', how='left')
            content_kpis[['total_reviews', 'avg_rating']] = content_kpis[['total_reviews', 'avg_rating']].fillna(0)
        
        # Add movie details
        for col in ['title', 'genre', 'release_year']:
            if col in movies.columns:
                content_kpis = content_kpis.merge(
                    movies[['movie_id', col]], 
                    on='movie_id', 
                    how='left'
                )
        
        # Calculate engagement score
        weights = {'watch_weight': 0.3, 'completion_weight': 0.4, 'rating_weight': 0.3}
        content_kpis['engagement_score'] = 0
        
        # Normalize and add watch count component
        max_watches = content_kpis['total_watches'].max()
        if max_watches > 0:
            watch_score = (content_kpis['total_watches'] / max_watches) * weights['watch_weight']
            content_kpis['engagement_score'] += watch_score
        
        # Add completion rate component if available
        if 'completion_rate' in content_kpis.columns:
            content_kpis['engagement_score'] += content_kpis['completion_rate'] * weights['completion_weight']
        
        # Add rating component if available
        if 'avg_rating' in content_kpis.columns:
            max_rating = content_kpis['avg_rating'].max()
            if max_rating > 0:
                rating_score = (content_kpis['avg_rating'] / max_rating) * weights['rating_weight']
                content_kpis['engagement_score'] += rating_score
        
        self.logger.info(f"Generated content performance KPIs for {len(content_kpis)} movies")
        
        return content_kpis
        
        return content_kpis

    def _calculate_recommendation_kpis(self, silver_data):
        """Calculate recommendation effectiveness metrics"""
        recommendations = silver_data.get('recommendation_logs', pd.DataFrame())
        movies = silver_data.get('movies', pd.DataFrame())
        
        if recommendations.empty or movies.empty:
            self.logger.warning("Empty recommendation logs or movies data")
            return pd.DataFrame()
            
        # Start with base movies DataFrame
        rec_metrics = movies[['movie_id']].copy()
        
        # Basic recommendation counts
        rec_counts = recommendations.groupby('movie_id').size().reset_index(name='times_recommended')
        rec_metrics = rec_metrics.merge(rec_counts, on='movie_id', how='left')
        rec_metrics['times_recommended'] = rec_metrics['times_recommended'].fillna(0)
        
        # Click metrics if available
        if 'clicked' in recommendations.columns:
            click_metrics = recommendations.groupby('movie_id')['clicked'].agg([
                ('times_clicked', 'sum'),
                ('click_through_rate', 'mean')
            ]).reset_index()
            rec_metrics = rec_metrics.merge(click_metrics, on='movie_id', how='left')
            rec_metrics[['times_clicked', 'click_through_rate']] = rec_metrics[['times_clicked', 'click_through_rate']].fillna(0)
        
        # Add relevance score
        rec_metrics['recommendation_relevance'] = 0.0
        if 'click_through_rate' in rec_metrics.columns and 'times_recommended' in rec_metrics.columns:
            # Weight CTR by number of recommendations
            max_recommendations = rec_metrics['times_recommended'].max()
            if max_recommendations > 0:
                recommendation_weight = rec_metrics['times_recommended'] / max_recommendations
                rec_metrics['recommendation_relevance'] = (
                    rec_metrics['click_through_rate'] * recommendation_weight
                ).fillna(0)
        
        # Add movie details for context
        for col in ['title', 'genre']:
            if col in movies.columns:
                rec_metrics = rec_metrics.merge(
                    movies[['movie_id', col]], 
                    on='movie_id', 
                    how='left'
                )
        
        self.logger.info(f"Generated recommendation KPIs for {len(rec_metrics)} movies")
        
        return rec_metrics

    def _calculate_retention_kpis(self, silver_data):
        """Calculate user retention metrics"""
        watch_history = silver_data.get('watch_history', pd.DataFrame())
        users = silver_data.get('users', pd.DataFrame())
        
        if watch_history.empty or users.empty:
            self.logger.warning("Empty watch history or users data for retention KPIs")
            return pd.DataFrame()
            
        # Start with base user metrics
        retention_data = users[['user_id']].copy()
        
        # Verify and convert watch_date column
        if 'watch_date' not in watch_history.columns:
            self.logger.error("watch_date column missing from watch_history")
            return pd.DataFrame()
            
        # Ensure watch_date is datetime
        try:
            watch_history['watch_date'] = pd.to_datetime(watch_history['watch_date'], errors='coerce')
        except Exception as e:
            self.logger.error(f"Error converting watch_date to datetime: {str(e)}")
            return pd.DataFrame()
            
        # Remove rows with invalid dates
        valid_dates = watch_history['watch_date'].notna()
        if not valid_dates.any():
            self.logger.error("No valid watch dates found")
            return pd.DataFrame()
            
        watch_history = watch_history[valid_dates]
        
        # Get first and last watch dates per user
        watch_dates = watch_history.groupby('user_id').agg({
            'watch_date': ['min', 'max']
        }).reset_index()
        
        watch_dates.columns = ['user_id', 'first_watch', 'last_watch']
        retention_data = retention_data.merge(watch_dates, on='user_id', how='left')
        
        # Initialize metrics columns
        retention_data['days_to_first_watch'] = 0
        retention_data['days_between_watches'] = 0
        retention_data['watch_frequency'] = 0
        retention_data['is_active'] = False
        retention_data['subscription_type'] = 'unknown'
        retention_data['join_date'] = pd.NaT
        
        # Add subscription type if available
        if 'subscription_type' in users.columns:
            retention_data = retention_data.merge(
                users[['user_id', 'subscription_type']], 
                on='user_id', 
                how='left'
            )
            retention_data['subscription_type'] = retention_data['subscription_type'].fillna('unknown')
        
        # Add and convert join_date if available
        if 'join_date' in users.columns:
            try:
                users['join_date'] = pd.to_datetime(users['join_date'], errors='coerce')
                retention_data = retention_data.merge(
                    users[['user_id', 'join_date']], 
                    on='user_id', 
                    how='left'
                )
            except Exception as e:
                self.logger.warning(f"Error converting join_date: {str(e)}")
                retention_data['join_date'] = pd.NaT
        
        # Calculate days to first watch where possible
        valid_join_dates = pd.notnull(retention_data['first_watch']) & pd.notnull(retention_data['join_date'])
        if valid_join_dates.any():
            retention_data.loc[valid_join_dates, 'days_to_first_watch'] = (
                retention_data.loc[valid_join_dates, 'first_watch'] - 
                retention_data.loc[valid_join_dates, 'join_date']
            ).dt.days
        
        # Calculate watch history metrics
        valid_watches = pd.notnull(retention_data['first_watch']) & pd.notnull(retention_data['last_watch'])
        retention_data.loc[valid_watches, 'days_between_watches'] = (
            retention_data.loc[valid_watches, 'last_watch'] - 
            retention_data.loc[valid_watches, 'first_watch']
        ).dt.days
        
        # Calculate watch counts and frequency
        watch_counts = watch_history.groupby('user_id').size().reset_index(name='total_watches')
        retention_data = retention_data.merge(watch_counts, on='user_id', how='left')
        retention_data['total_watches'] = retention_data['total_watches'].fillna(0)
        
        # Calculate watch frequency (watches per month) safely
        has_activity = (retention_data['days_between_watches'] > 0) & (retention_data['total_watches'] > 0)
        if has_activity.any():
            retention_data.loc[has_activity, 'watch_frequency'] = (
                retention_data.loc[has_activity, 'total_watches'] * 30 / 
                retention_data.loc[has_activity, 'days_between_watches']
            ).fillna(0)
        
        # Calculate active status based on last watch date
        retention_data['last_watch_days_ago'] = pd.NaT
        valid_last_watch = pd.notnull(retention_data['last_watch'])
        if valid_last_watch.any():
            retention_data.loc[valid_last_watch, 'last_watch_days_ago'] = (
                datetime.now() - retention_data.loc[valid_last_watch, 'last_watch']
            ).dt.days
            retention_data.loc[valid_last_watch, 'is_active'] = retention_data.loc[valid_last_watch, 'last_watch_days_ago'] <= 30
        
        # Calculate retention score components
        retention_data['retention_score'] = 0.0
        weights = {
            'recency_weight': 0.4,
            'frequency_weight': 0.3,
            'longevity_weight': 0.3
        }
        
        # 1. Recency component - based on days since last watch
        if valid_last_watch.any():
            max_days = retention_data['last_watch_days_ago'].max()
            if max_days and max_days > 0:
                recency_score = 1 - (retention_data['last_watch_days_ago'] / max_days)
                retention_data.loc[valid_last_watch, 'retention_score'] += (
                    recency_score * weights['recency_weight']
                )
        
        # 2. Frequency component - based on watch frequency
        max_frequency = retention_data['watch_frequency'].max()
        if max_frequency > 0:
            frequency_score = retention_data['watch_frequency'] / max_frequency
            retention_data['retention_score'] += frequency_score * weights['frequency_weight']
        
        # 3. Longevity component - based on total time span of activity
        max_span = retention_data['days_between_watches'].max()
        if max_span > 0:
            longevity_score = retention_data['days_between_watches'] / max_span
            retention_data['retention_score'] += longevity_score * weights['longevity_weight']
        
        # Clean up temporary columns and ensure all numeric columns have valid values
        retention_data = retention_data.drop('last_watch_days_ago', axis=1)
        numeric_columns = ['days_to_first_watch', 'days_between_watches', 
                         'watch_frequency', 'total_watches', 'retention_score']
        for col in numeric_columns:
            retention_data[col] = retention_data[col].fillna(0).astype(float)
            
        # Log retention metrics summary
        self.logger.info(f"Generated retention KPIs for {len(retention_data)} users")
        self.logger.info(f"Active users: {retention_data['is_active'].sum()}")
        self.logger.info(f"Average retention score: {retention_data['retention_score'].mean():.2f}")
        
        # Log data quality metrics
        missing_counts = retention_data.isnull().sum()
        if missing_counts.any():
            self.logger.warning("Missing values in retention KPIs:")
            for col in missing_counts[missing_counts > 0].index:
                self.logger.warning(f"  {col}: {missing_counts[col]} missing values")
                
        # Validate retention scores
        invalid_scores = retention_data[
            (retention_data['retention_score'] < 0) | 
            (retention_data['retention_score'] > 1)
        ]
        if not invalid_scores.empty:
            self.logger.warning(f"Found {len(invalid_scores)} users with invalid retention scores")
            retention_data.loc[
                retention_data['retention_score'] < 0, 'retention_score'
            ] = 0
            retention_data.loc[
                retention_data['retention_score'] > 1, 'retention_score'
            ] = 1
        
        self.logger.info(f"Generated retention KPIs for {len(retention_data)} users")
        
        return retention_data

    def _create_gold_layer_summary(self, gold_data, gold_path):
        """Create summary statistics for gold layer data"""
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {}
        }
        
        # User Engagement Summary
        if 'user_engagement' in gold_data:
            user_eng = gold_data['user_engagement']
            summary['metrics']['user_engagement'] = {
                'total_users': len(user_eng),
                'avg_engagement_score': float(user_eng['engagement_score'].mean()),
                'highly_engaged_users': int(len(user_eng[user_eng['engagement_score'] > 0.7])),
                'avg_watches_per_user': float(user_eng['total_watches'].mean() if 'total_watches' in user_eng else 0)
            }
            
        # Content Performance Summary
        if 'content_performance' in gold_data:
            content_perf = gold_data['content_performance']
            summary['metrics']['content_performance'] = {
                'total_movies': len(content_perf),
                'avg_watch_duration': float(content_perf['avg_duration'].mean() if 'avg_duration' in content_perf else 0),
                'top_performing_count': int(len(content_perf[content_perf['engagement_score'] > 0.7])),
                'avg_completion_rate': float(content_perf['completion_rate'].mean() if 'completion_rate' in content_perf else 0)
            }
            
        # Recommendation Effectiveness Summary
        if 'recommendation_kpis' in gold_data:
            rec_kpis = gold_data['recommendation_kpis']
            summary['metrics']['recommendations'] = {
                'total_recommendations': int(rec_kpis['times_recommended'].sum() if 'times_recommended' in rec_kpis else 0),
                'avg_click_through_rate': float(rec_kpis['click_through_rate'].mean() if 'click_through_rate' in rec_kpis else 0),
                'high_ctr_content_count': int(len(rec_kpis[rec_kpis['click_through_rate'] > 0.3]) if 'click_through_rate' in rec_kpis else 0)
            }
            
        # Retention Summary
        if 'retention_kpis' in gold_data:
            retention = gold_data['retention_kpis']
            summary['metrics']['retention'] = {
                'active_users': int(retention['is_active'].sum() if 'is_active' in retention else 0),
                'avg_retention_score': float(retention['retention_score'].mean() if 'retention_score' in retention else 0),
                'avg_days_between_watches': float(retention['days_between_watches'].mean() if 'days_between_watches' in retention else 0),
                'high_retention_users': int(len(retention[retention['retention_score'] > 0.7]) if 'retention_score' in retention else 0)
            }
        
        # Save summary
        summary_path = gold_path / "gold_layer_summary.json"
        with open(summary_path, 'w') as f:
            json.dumps(summary, indent=2, default=str)
        self.logger.info(f"Created gold layer summary at {summary_path}")
        
        return summary

if __name__ == "__main__":
    from config import DATA_CONFIG, SCHEMA_CONFIG
    
    # Initialize processor
    config = {
        'DATA_CONFIG': DATA_CONFIG,
        'SCHEMA_CONFIG': SCHEMA_CONFIG
    }
    processor = DataLakeProcessor(config)
    
    # Process through layers
    bronze_data = processor.create_bronze_layer()
    silver_data = processor.create_silver_layer(bronze_data)
    gold_data = processor.create_gold_layer(silver_data)