"""
Main script to run the Netflix User Engagement Analytics Pipeline
"""
import logging
from pathlib import Path
from datetime import datetime

from data_lake_processor import DataLakeProcessor
from data_quality import DataQualityChecker
from config import DATA_CONFIG, SCHEMA_CONFIG, QUALITY_THRESHOLDS, VALIDATION_RULES

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    # Create main data layer directories
    data_dirs = [
        DATA_CONFIG['bronze_layer'],
        DATA_CONFIG['silver_layer'],
        DATA_CONFIG['gold_layer'],
        DATA_CONFIG['quality_reports_path']
    ]
    
    for path in data_dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

def main():
    """Run the complete ETL pipeline"""
    logger = setup_logging()
    logger.info("Starting Netflix User Engagement Analytics Pipeline")

    try:
        # Create directories
        create_directories()

        # Initialize components
        config = {
            'DATA_CONFIG': DATA_CONFIG,
            'SCHEMA_CONFIG': SCHEMA_CONFIG,
            'QUALITY_THRESHOLDS': QUALITY_THRESHOLDS,
            'VALIDATION_RULES': VALIDATION_RULES
        }

        processor = DataLakeProcessor(config)
        quality_checker = DataQualityChecker(config)

        # Step 1: Bronze Layer - Raw data ingestion
        logger.info("Step 1: Creating Bronze Layer")
        bronze_data = processor.create_bronze_layer()
        
        # Step 2: Data Quality Assessment
        logger.info("Step 2: Performing data quality assessment")
        quality_reports = {}
        for source, df in bronze_data.items():
            quality_report = quality_checker.generate_quality_report(df)
            quality_reports[source] = quality_report
            report_path = quality_checker.save_quality_report(quality_report)
            logger.info(f"Quality report for {source} saved to: {report_path}")

        # Step 3: Silver Layer - Cleaned and standardized data
        logger.info("Step 3: Creating Silver Layer")
        silver_data = processor.create_silver_layer(bronze_data)

        # Step 4: Gold Layer - Business KPIs and metrics
        logger.info("Step 4: Creating Gold Layer")
        gold_data = processor.create_gold_layer(silver_data)

        logger.info("Pipeline completed successfully!")
        logger.info("Generated KPIs:")
        for kpi_type, kpi_data in gold_data.items():
            logger.info(f"\n{kpi_type}:")
            if not kpi_data.empty:
                logger.info(f"Number of records: {len(kpi_data)}")
                logger.info(f"Columns: {list(kpi_data.columns)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()