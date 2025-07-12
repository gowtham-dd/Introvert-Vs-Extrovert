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
