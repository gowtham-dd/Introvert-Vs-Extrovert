

from src.IntrovertVsExtrovert.config.configuration import ConfigurationManager
from src.IntrovertVsExtrovert.components.Model_Trainer import ModelTrainer
from src.IntrovertVsExtrovert import logger


STAGE_NAME="Data Ingestion stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
         
        try:
            config = ConfigurationManager()
            modeltrainer_config  = config.get_model_trainer_config()

            trainer = ModelTrainer(modeltrainer_config)
            trainer.train()

        except Exception as e:
            print(f"❌ Exception during model‑training stage: {e}")

           
       



if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=ModelTrainerPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
