from config.path_config import *

from utils.common_functions import read_yaml
from src.data_processing import DataProcessor
from src.model_training import ModelTraining

if __name__ == "__main__":
    # ===================================================================================================
    # 1) DATA INGESTION
    # ===================================================================================================
    """
    Skip the ingestion from GCP
    - Data artifacts will be stored in DVC
    """

    # ===================================================================================================
    # 2) DATA PROCESSING
    # ===================================================================================================
    data_processor = DataProcessor(input_file = ANIMELIST_CSV, output_dir= PROCESSED_DIR
    )
    data_processor.run()


    # ===================================================================================================
    # 3) MODEL TRAINING
    # ===================================================================================================
    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()