from src.IntrovertVsExtrovert import logger
from src.IntrovertVsExtrovert.pipeline.Data_Ingestion_pipeline import DataIngestionTrainingPipeline
from src.IntrovertVsExtrovert.pipeline.Data_Validation_pipeline import DataValidationTrainingPipeline
# from src.IntrovertVsExtrovert.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
# from src.IntrovertVsExtrovert.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
# from src.IntrovertVsExtrovert.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline
# import dagshub
# dagshub.init(repo_owner='gowtham-dd', repo_name='winepred-MLFLOW', mlflow=True)


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
