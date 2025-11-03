"""
Configuration settings for the Netflix User Engagement Analytics Pipeline
"""

# Data paths
DATA_CONFIG = {
    'raw_data': {
        'users': 'users.csv',
        'movies': 'movies.csv',
        'watch_history': 'watch_history.csv',
        'reviews': 'reviews.csv',
        'search_logs': 'search_logs.csv',
        'recommendation_logs': 'recommendation_logs.csv'
    },
    'bronze_layer': 'data/bronze/',
    'silver_layer': 'data/silver/',
    'gold_layer': 'data/gold/',
    'quality_reports_path': 'data/quality_reports/'
}

# Validation rules for different data sources
VALIDATION_RULES = {
    'users': {
        'required_fields': ['user_id', 'subscription_type', 'join_date', 'country', 'age', 'gender'],
        'numeric_fields': ['age'],
        'categorical_fields': ['subscription_type', 'country', 'gender'],
        'date_fields': ['join_date']
    },
    'movies': {
        'required_fields': ['movie_id', 'title', 'genre', 'release_year', 'duration'],
        'numeric_fields': ['release_year', 'duration'],
        'categorical_fields': ['genre'],
        'date_fields': []
    },
    'watch_history': {
        'required_fields': ['user_id', 'movie_id', 'watch_date', 'watch_duration'],
        'numeric_fields': ['watch_duration'],
        'categorical_fields': [],
        'date_fields': ['watch_date']
    },
    'reviews': {
        'required_fields': ['user_id', 'movie_id', 'rating', 'review_date'],
        'numeric_fields': ['rating'],
        'categorical_fields': [],
        'date_fields': ['review_date']
    },
    'search_logs': {
        'required_fields': ['user_id', 'search_query', 'search_date'],
        'numeric_fields': [],
        'categorical_fields': [],
        'date_fields': ['search_date']
    },
    'recommendation_logs': {
        'required_fields': ['user_id', 'movie_id', 'recommendation_date', 'clicked'],
        'numeric_fields': [],
        'categorical_fields': ['clicked'],
        'date_fields': ['recommendation_date']
    }
}

# Quality thresholds for data validation
QUALITY_THRESHOLDS = {
    'missing_threshold': 0.1,  # Maximum allowed percentage of missing values
    'duplicate_threshold': 0.05,  # Maximum allowed percentage of duplicates
    'outlier_std_threshold': 3,  # Number of standard deviations for outlier detection
    'min_completeness': 0.95,  # Minimum required data completeness
    'correlation_threshold': 0.8,  # Threshold for high correlation warning
    'category_threshold': 50,  # Maximum number of unique categories for categorical fields
}

# Schema version for data validation
SCHEMA_VERSION = "1.0"

# Data schemas with required and optional fields
SCHEMA_CONFIG = {
    'users': {
        'required': ['user_id'],  # Only user_id is truly required
        'enriched': ['subscription_type', 'join_date'],  # Fields that can be enriched if missing
        'optional': ['country', 'age', 'gender'],
        'numeric': ['age'],
        'categorical': ['subscription_type', 'country', 'gender'],
        'date': ['join_date'],
        'defaults': {
            'subscription_type': 'basic',  # Default subscription type
            'join_date': None,  # Will be enriched based on first activity
            'age': None,
            'country': 'unknown',
            'gender': 'unspecified'
        },
        'validation_rules': {
            'subscription_type': ['basic', 'standard', 'premium'],
            'age': {'min': 13, 'max': 100},
            'join_date': {'min': '2020-01-01', 'max': None}  # None means current date
        }
    },
    'movies': {
        'required': ['movie_id', 'title', 'release_year'],
        'optional': ['genre', 'duration', 'content_rating'],
        'numeric': ['release_year', 'duration'],
        'categorical': ['genre', 'content_rating'],
        'date': [],
        'defaults': {
            'genre': 'uncategorized',
            'duration': 0,
            'content_rating': 'unrated'
        }
    },
    'watch_history': {
        'required': ['user_id', 'movie_id', 'watch_date'],
        'optional': ['watch_duration', 'completed'],
        'numeric': ['watch_duration'],
        'categorical': ['completed'],
        'date': ['watch_date'],
        'defaults': {
            'watch_duration': 0,
            'completed': 0
        },
        'relationships': {
            'user_id': 'users.user_id',
            'movie_id': 'movies.movie_id'
        }
    },
    'reviews': {
        'required': ['user_id', 'movie_id', 'rating', 'review_date'],
        'optional': ['review_text'],
        'numeric': ['rating'],
        'categorical': [],
        'date': ['review_date'],
        'defaults': {
            'review_text': '',
            'rating': None
        },
        'relationships': {
            'user_id': 'users.user_id',
            'movie_id': 'movies.movie_id'
        }
    },
    'search_logs': {
        'required': ['user_id', 'search_query', 'search_date'],
        'optional': ['results_clicked'],
        'numeric': [],
        'categorical': ['results_clicked'],
        'date': ['search_date'],
        'defaults': {
            'results_clicked': None
        },
        'relationships': {
            'user_id': 'users.user_id'
        }
    },
    'recommendation_logs': {
        'required': ['user_id', 'movie_id', 'recommendation_date'],
        'optional': ['clicked'],
        'numeric': [],
        'categorical': ['clicked'],
        'date': ['recommendation_date'],
        'defaults': {
            'clicked': 0
        },
        'relationships': {
            'user_id': 'users.user_id',
            'movie_id': 'movies.movie_id'
        }
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    'temporal_features': [
        'subscription_length_days',
        'days_since_last_payment'
    ],
    'categorical_encoding': 'one_hot',  # Options: 'one_hot', 'label', 'target'
    'scaling_method': 'standard',  # Options: 'standard', 'minmax', 'robust'
}

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'database': 'netflix_analytics',
    'user': 'your_username',
    'password': 'your_password'
}
