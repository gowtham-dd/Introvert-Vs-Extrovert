# personality_prediction/components/data_transformation.py
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.IntrovertVsExtrovert.entity.config_entity import DataTransformationConfig


CAT_COLS = ["Stage_fear", "Drained_after_socializing", "P2"]

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _merge_datasets(self, train_df: pd.DataFrame, org_df: pd.DataFrame) -> pd.DataFrame:
        key_cols = [
            "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
            "Going_outside", "Drained_after_socializing",
            "Friends_circle_size", "Post_frequency"
        ]
        org_df = org_df.rename(columns={"Personality": "P2"})
        org_df = org_df.drop_duplicates(subset=key_cols)
        merged = train_df.merge(org_df, how="left", on=key_cols)
        return merged

    def transform(self):
        # 1 Load validated, cleaned CSVs from Stage 2
        train_df = pd.read_csv(self.config.train_data_path)
        org_df   = pd.read_csv(self.config.org_combined_path)

        # 2 Drop id if still present (safety)
        if "id" in train_df.columns:
            train_df = train_df.drop(columns=["id"])

        # 3 Merge & add P2
        train_df = self._merge_datasets(train_df, org_df)

        # 4 Target encode (Introvert=0, Extrovert=1)
        le = LabelEncoder()
        train_df["Personality"] = le.fit_transform(train_df["Personality"])

        # 5 Feature/target split
        X = train_df.drop(columns=["Personality"])
        y = train_df["Personality"]

        # 6 Ordinal‑encode categorical features (Stage_fear, Drained_after_socializing, P2)
        ord_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X[CAT_COLS] = ord_enc.fit_transform(X[CAT_COLS])

        # 7 Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 8 Persist everything
        Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)
        X_train.to_parquet(self.config.x_train_path, index=False)
        X_val.to_parquet(self.config.x_val_path, index=False)
        y_train.to_csv(self.config.y_train_path, index=False, header=False)
        y_val.to_csv(self.config.y_val_path, index=False, header=False)
        joblib.dump(ord_enc, self.config.ordinal_encoder_path)
        joblib.dump(le, self.config.label_encoder_path)

        print("✅ Data transformation complete.")
        return X_train, X_val, y_train, y_val
