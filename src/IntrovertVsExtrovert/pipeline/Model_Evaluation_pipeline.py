from IntrovertVsExtrovert.config.configuration import ConfigurationManager
from IntrovertVsExtrovert.components.Model_Evaluation import ModelEvaluation
from IntrovertVsExtrovert import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()

        evaluator = ModelEvaluation(eval_config)
        evaluator.log_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
