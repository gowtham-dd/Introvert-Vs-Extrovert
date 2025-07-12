## ENTITY
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    train_data_path: Path
    original_data_path: Path
    clean_train_path: Path          # ➜ new
    clean_org_path: Path            # ➜ new
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    org_combined_path: Path
    x_train_path: Path
    x_val_path: Path
    y_train_path: Path
    y_val_path: Path
    ordinal_encoder_path: Path
    label_encoder_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    x_train_path: Path
    y_train_path: Path
    model_dir: Path
    xgb_model_pattern: str
    cat_model_pattern: str
    ensemble_path: Path
    # Hyper‑params & CV
    xgb_params: dict
    cat_params: dict
    n_splits: int
    n_repeats: int
    ensemble_weights: dict



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    x_val_path: Path
    y_val_path: Path
    model_dir: Path
    ensemble_path: Path
    ordinal_encoder_path: Path
    label_encoder_path: Path
    metric_file: Path
    mlflow_uri: str
    ensemble_weights: dict           # from params.yaml

