import pandas as pd
from src.IntrovertVsExtrovert.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _check_schema(self, df: pd.DataFrame, dataset_name: str) -> bool:
        actual_columns = set(df.columns)
        expected_columns = set(self.config.all_schema.keys())

        if not expected_columns.issubset(actual_columns):
            missing = expected_columns - actual_columns
            print(f"[{dataset_name}] ❌ Missing columns: {missing}")
            return False

        return True

    def _check_missing_and_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        print(f"\n[{dataset_name}] 🧹 Checking for missing values...")
        print(df.isnull().sum())

        print(f"\n[{dataset_name}] 🧹 Checking for duplicates...")
        dup_count = df.duplicated().sum()
        print(f"Found {dup_count} duplicated rows.")

        if dup_count > 0:
            df = df.drop_duplicates()
            print(f"[{dataset_name}] ✅ Duplicates dropped.")

        return df

    def validate_and_clean(self):
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            org_df = pd.read_csv(self.config.original_data_path)

            # Drop id column if present
            if 'id' in train_df.columns:
                train_df.drop(columns=['id'], inplace=True)

            # Schema check
            status_train = self._check_schema(train_df, "Train Data")
            status_org = self._check_schema(org_df, "Original Data")

            validation_status = status_train and status_org

            # Clean datasets
            if validation_status:
                train_df = self._check_missing_and_duplicates(train_df, "Train Data")
                org_df = self._check_missing_and_duplicates(org_df, "Original Data")
                train_df.to_csv(self.config.clean_train_path, index=False)
                org_df.to_csv(self.config.clean_org_path, index=False)

            # Save validation result
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return train_df, org_df

        except Exception as e:
            raise e
