from src.IntrovertVsExtrovert.config.configuration import ConfigurationManager
from src.IntrovertVsExtrovert.components.Data_Ingestion import DataIngestion
from src.IntrovertVsExtrovert import logger


STAGE_NAME="Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
           
            try:
                config = ConfigurationManager()
                data_ingestion_config = config.get_data_ingestion_config()
                data_ingestion = DataIngestion(config=data_ingestion_config)

    # Step 1: Download
                data_ingestion.download_file()

    # Step 2: Unzip
                data_ingestion.extract_zip_file()

    # Step 3: Load CSVs from extracted folder
                train_df, org_combined = data_ingestion.load_datasets()

                print("Train Shape:", train_df.shape)
                print("Org Combined Shape:", org_combined.shape)

            except Exception as e:
                logger.exception(e)
                raise e



if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataIngestionTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
