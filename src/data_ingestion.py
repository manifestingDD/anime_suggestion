import os
import pandas as pd
from google.cloud import storage

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]  
        self.bucket_name = self.config['bucket_name']
        self.file_names = self.config["bucket_file_names"] 

        os.makedirs(RAW_DIR, exist_ok= True)
        
        logger.info("Data ingestion started")

    # Download from GCP
    def download_csv_from_gcp(self):
        try:
            # Initilize a client
            client = storage.Client()
            bucket = client.bucket(bucket_name= self.bucket_name)
            
            # Looping download 3 files, loading only 5 million rows rather than full size of anime list
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                
                if file_name == "animelist.csv": 
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    
                    # Trim down the rows
                    data= pd.read_csv(file_path, nrows = 5000000)
                    data.to_csv(file_path, index = False)
                    logger.info("Large file detected >>> only downloading 5M rows") 
                
                else: 
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info("Downloading smaller files -- anime and anime_with_synopsis")


        except Exception as e:
            logger.error("Error while downloading data from GCP")
            raise CustomException("Failed to download data", e) 

            

    def run(self):
        try:
            logger.info("Stating data ingestion process ...")
            self.download_csv_from_gcp()
            logger.info("Data ingestion completed")

        except Exception as ce:
            logger.error(f"CustomException: {str(ce)}")

        finally:
            logger.info("Data ingestion done")



if __name__ == "__main__":  # Content in this block will be executed when we run this script
    data_ingestion = DataIngestion(config = read_yaml(CONFIG_PATH))
    data_ingestion.run()