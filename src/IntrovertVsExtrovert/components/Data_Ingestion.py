import os
import urllib.request as request
import zipfile
import pandas as pd
from IntrovertVsExtrovert import logger
from IntrovertVsExtrovert.utils.common import get_size
from src.IntrovertVsExtrovert.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """Downloads the file from source_URL to local_data_file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
        
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading data from {self.config.source_URL}...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,  # Use source_URL here
                filename=self.config.local_data_file  # Save to local_data_file
            )
            logger.info(f"Download completed to: {filename}")
            logger.debug(f"Download headers: {headers}")
        else:
            logger.info(f"File already exists at {self.config.local_data_file}, size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """Extracts the downloaded zip file to unzip_dir"""
        logger.info(f"Extracting zip file from {self.config.local_data_file} to {self.config.unzip_dir}")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        print("✅ Listing files after unzip:")
        for root, dirs, files in os.walk(self.config.unzip_dir):
            for file in files:
                print(os.path.join(root, file))

        
        logger.info(f"Successfully extracted to {self.config.unzip_dir}")

    def load_datasets(self):
        """Load train.csv and org CSVs from the ExtvsInt folder inside unzip_dir"""
        base_dir = os.path.join(self.config.unzip_dir, "ExtvsInt")   # <- key change

        train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
        org1     = pd.read_csv(os.path.join(base_dir, "personality_dataset.csv"))
        org2     = pd.read_csv(os.path.join(base_dir, "personality_dataset_1.csv"))
        org3     = pd.read_csv(os.path.join(base_dir, "personality_dataset_2.csv"))

        org_combined = pd.concat([org1, org2], ignore_index=True)
        # inside load_datasets() ─ after org_combined is created
        save_path = os.path.join(self.config.unzip_dir, "org_combined.csv")
        org_combined.to_csv(save_path, index=False)
        logger.info(f"org_combined saved to {save_path}")

        return train_df, org_combined

