import json, joblib, os, warnings, numpy as np, pandas as pd
from pathlib import Path
from src.IntrovertVsExtrovert.utils.common import read_yaml, create_directories,save_json

from typing import List, Dict
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (log_loss, accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
import mlflow
from urllib.parse import urlparse
from src.IntrovertVsExtrovert.entity.config_entity import ModelEvaluationConfig

warnings.filterwarnings("ignore")


class ModelEvaluation:
    def __init__(self, cfg: ModelEvaluationConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _load_xy(self):
        X = pd.read_parquet(self.cfg.x_val_path)
        y = pd.read_csv(self.cfg.y_val_path, header=None).squeeze("columns")
        return X, y

    # ------------------------------------------------------------------
    def _load_xgb_models(self) -> List[xgb.Booster]:
        models = []
        for p in sorted(self.cfg.model_dir.glob("xgb_fold*.bin")):
            model = xgb.Booster()
            model.load_model(str(p))
            models.append(model)
        if not models:
            raise FileNotFoundError("No XGB models found in model_dir")
        return models

    def _load_cat_models(self) -> List[CatBoostClassifier]:
        models = []
        for p in sorted(self.cfg.model_dir.glob("cat_fold*.cbm")):
            m = CatBoostClassifier()
            m.load_model(str(p))
            models.append(m)
        if not models:
            raise FileNotFoundError("No CatBoost models found in model_dir")
        return models

    # ------------------------------------------------------------------
    def _average_preds(self, preds: List[np.ndarray]) -> np.ndarray:
        return np.mean(np.vstack(preds), axis=0)

    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, float]:
        # 1 Load data
        X, y = self._load_xy()

        # 2 Load models & get probabilities
        xgb_probs = self._average_preds([
            m.predict(xgb.DMatrix(X)) for m in self._load_xgb_models()
        ])
        cat_probs = self._average_preds([
            m.predict_proba(X)[:, 1] for m in self._load_cat_models()
        ])

        # 3 Blend
        if Path(self.cfg.ensemble_path).exists():
            weights = joblib.load(self.cfg.ensemble_path)
        else:
            weights = self.cfg.ensemble_weights   # from params.yaml

        blended = weights["xgb"] * xgb_probs + weights["cat"] * cat_probs
        preds   = (blended >= 0.5).astype(int)

        # 4 Metrics
        metrics = {
            "accuracy":  accuracy_score(y, preds),
            "log_loss":  log_loss(y, blended),
            "f1_score":  f1_score(y, preds),
            "precision": precision_score(y, preds),
            "recall":    recall_score(y, preds),
            "roc_auc":   roc_auc_score(y, blended),
        }

        # 5 Persist metrics as JSON
        save_json(self.cfg.metric_file, metrics)
        print(f"Metrics saved ➜ {self.cfg.metric_file}")
        return metrics

    # ------------------------------------------------------------------
    def log_mlflow(self):
        metrics = self.evaluate()

        mlflow.set_tracking_uri(self.cfg.mlflow_uri)
        scheme = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name="model-eval"):
            # params: just ensemble weights here
            mlflow.log_params(self.cfg.ensemble_weights)

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # log artifacts
            mlflow.log_artifact(str(self.cfg.metric_file), artifact_path="metrics")
            mlflow.log_artifact(str(self.cfg.ordinal_encoder_path), artifact_path="encoders")
            mlflow.log_artifact(str(self.cfg.label_encoder_path),   artifact_path="encoders")

            # register first model of each type (optional)
            if scheme != "file":
                xgb0 = self._load_xgb_models()[0]
                mlflow.xgboost.log_model(
                    xgb_model=xgb0,
                    artifact_path="xgb_model",
                    registered_model_name="IntroExtro_XGB"
                )

                cat0 = self._load_cat_models()[0]
                mlflow.catboost.log_model(
                    cb_model=cat0,
                    artifact_path="cat_model",
                    registered_model_name="IntroExtro_Cat"
                )


            print("✅ Metrics & artifacts logged to MLflow.")

