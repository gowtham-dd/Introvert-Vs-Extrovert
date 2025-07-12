from src.IntrovertVsExtrovert import logger
from src.IntrovertVsExtrovert.pipeline.Data_Ingestion_pipeline import DataIngestionTrainingPipeline
from src.IntrovertVsExtrovert.pipeline.Data_Validation_pipeline import DataValidationTrainingPipeline
from src.IntrovertVsExtrovert.pipeline.Data_Transformation_pipeline import DataTransformationTrainingPipeline
from src.IntrovertVsExtrovert.pipeline.Model_Training_Pipeline import ModelTrainerPipeline
from src.IntrovertVsExtrovert.pipeline.Model_Evaluation_pipeline import ModelEvaluationPipeline
import dagshub
dagshub.init(repo_owner='gowtham-dd', repo_name='Introvert-Vs-Extrovert', mlflow=True)


STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Data Validation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Data Transformation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME="Model Evaluation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e