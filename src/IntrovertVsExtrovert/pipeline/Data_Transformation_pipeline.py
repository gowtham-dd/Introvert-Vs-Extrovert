from src.IntrovertVsExtrovert.config.configuration import ConfigurationManager
from src.IntrovertVsExtrovert.components.Data_Transformation import DataTransformation
from src.IntrovertVsExtrovert import logger


STAGE_NAME="Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
           
            try:
    # 1️⃣  Load config
                config = ConfigurationManager()
                data_trans_cfg = config.get_data_transformation_config()

    # 2️⃣  Run transformation
                transformer = DataTransformation(config=data_trans_cfg)
                X_train, X_val, y_train, y_val = transformer.transform()

                print("✅ Data Transformation Successful.")
            except Exception as e:
                print(f"❌ Exception during transformation: {e}")

if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataTransformationTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
