import os, joblib, warnings, numpy as np, pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from typing import Tuple
from src.IntrovertVsExtrovert.entity.config_entity import ModelTrainerConfig

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.cfg = config

    # ----------------------------------------------------------------------------------
    def _load_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        X = pd.read_parquet(self.cfg.x_train_path)
        y = pd.read_csv(self.cfg.y_train_path, header=None,encoding='utf-8').squeeze("columns")
        return X, y

    # ----------------------------------------------------------------------------------
    def _train_xgb(self, X, y) -> Tuple[np.ndarray, list]:
        oof = np.zeros(len(X))
        folds_models = []
        rkf = RepeatedStratifiedKFold(
            n_splits=self.cfg.n_splits,
            n_repeats=self.cfg.n_repeats,
            random_state=self.cfg.xgb_params["random_state"],
        )

        for fold, (tr, va) in enumerate(rkf.split(X, y)):
            dtr = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dva = xgb.DMatrix(X.iloc[va], label=y.iloc[va])

            model = xgb.train(
                params=self.cfg.xgb_params,
                dtrain=dtr,
                num_boost_round=1000,
                evals=[(dva, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            oof[va] = model.predict(dva)
            mod_path = self.cfg.model_dir / self.cfg.xgb_model_pattern.format(fold=fold)
            model.save_model(mod_path)
            folds_models.append(mod_path)
            print(f"✓ XGB fold {fold} saved at {mod_path}")

        print(f"XGB log‑loss: {log_loss(y, oof):.4f}  acc: {accuracy_score(y, oof>0.5):.4f}")
        return oof, folds_models

    # ----------------------------------------------------------------------------------
    def _train_cat(self, X, y) -> Tuple[np.ndarray, list]:
        oof = np.zeros(len(X))
        folds_models = []
        rkf = RepeatedStratifiedKFold(
            n_splits=self.cfg.n_splits * 2,  # mimic notebook’s 10×2
            n_repeats=self.cfg.n_repeats,
            random_state=self.cfg.cat_params["random_seed"],
        )

        for fold, (tr, va) in enumerate(rkf.split(X, y)):
            model = CatBoostClassifier(**self.cfg.cat_params)
            model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            mod_path = self.cfg.model_dir / self.cfg.cat_model_pattern.format(fold=fold)
            model.save_model(mod_path)
            folds_models.append(mod_path)
            print(f"✓ CatBoost fold {fold} saved at {mod_path}")

        print(f"Cat log‑loss: {log_loss(y, oof):.4f}  acc: {accuracy_score(y, oof>0.5):.4f}")
        return oof, folds_models

    # ----------------------------------------------------------------------------------
    def train(self):
        try:
            Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
            X, y = self._load_xy()

            oof_xgb, _ = self._train_xgb(X, y)
            oof_cat, _ = self._train_cat(X, y)

            # Save ensemble weights for use in Stage 5 (evaluation) & Stage 6 (prediction)
            joblib.dump(self.cfg.ensemble_weights, self.cfg.ensemble_path)
            print(f"Ensemble weights saved ➜ {self.cfg.ensemble_path}")

            # Quick blended CV metric
            w = self.cfg.ensemble_weights
            blend = w["xgb"]*oof_xgb + w["cat"]*oof_cat
            print(f"Blended CV log‑loss: {log_loss(y, blend):.4f}  acc: {accuracy_score(y, blend>0.5):.4f}")

            print("✅ Model‑training stage completed.")
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}") from e
