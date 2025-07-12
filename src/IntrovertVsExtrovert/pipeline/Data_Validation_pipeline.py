

from src.IntrovertVsExtrovert.config.configuration import ConfigurationManager
from src.IntrovertVsExtrovert.components.Data_Validation import DataValidation
from src.IntrovertVsExtrovert import logger


STAGE_NAME="Data Ingestion stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
           
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()

            validator = DataValidation(config=data_validation_config)
            cleaned_train_df, cleaned_org_df = validator.validate_and_clean()

            print("✅ Data Validation and Cleaning Successful.")
        except Exception as e:
            logger.exception(e)
            raise e




if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataValidationTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
