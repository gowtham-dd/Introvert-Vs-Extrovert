from src.IntrovertVsExtrovert.constant import *
from src.IntrovertVsExtrovert.utils.common import read_yaml,create_directories 
from src.IntrovertVsExtrovert.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            train_data_path=config.train_data_path,
            original_data_path=config.original_data_path,
            clean_train_path=config.clean_train_path,     
            clean_org_path=config.clean_org_path,         
            all_schema=schema
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            org_combined_path=config.org_combined_path,
            x_train_path=config.x_train_path,
            x_val_path=config.x_val_path,
            y_train_path=config.y_train_path,
            y_val_path=config.y_val_path,
            ordinal_encoder_path=config.ordinal_encoder_path,
            label_encoder_path=config.label_encoder_path
        )



    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg    = self.config.model_trainer
        xgb_p  = self.params.XGBoost
        cat_p  = self.params.CatBoost
        train  = self.params.training

        create_directories([cfg.root_dir, cfg.model_dir])

        return ModelTrainerConfig(
            root_dir          = Path(cfg.root_dir),
            x_train_path      = Path(cfg.x_train_path),
            y_train_path      = Path(cfg.y_train_path),
            model_dir         = Path(cfg.model_dir),
            xgb_model_pattern = cfg.xgb_model_pattern,
            cat_model_pattern = cfg.cat_model_pattern,
            ensemble_path     = Path(cfg.ensemble_path),
            xgb_params        = dict(xgb_p),
            cat_params        = dict(cat_p),
            n_splits          = train.n_splits,
            n_repeats         = train.n_repeats,
            ensemble_weights  = dict(train.ensemble_weights)
        )
