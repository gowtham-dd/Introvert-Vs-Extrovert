# prediction_pipeline.py  – Introvert ↔ Extrovert
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path
import joblib
from typing import Dict, List


class PersonalityPredictor:
    """
    Lightweight inference wrapper for the Introvert‑vs‑Extrovert ensemble.

    • Loads every `xgb_fold*.bin`  (and optional `cat_fold*.cbm`) in
      artifacts/model_training/models
    • Applies the saved OrdinalEncoder & ensemble weights
    • Returns prediction label + class probabilities
    """

    # --- artifact paths ------------------------------------------------
    _MODEL_DIR  = Path("artifacts/model_training/models")
    _WEIGHT_PKL = Path("artifacts/model_training/ensemble_weights.pkl")
    _ORD_ENC    = Path("artifacts/data_transformation/ordinal_encoder.pkl")
    _LBL_ENC    = Path("artifacts/data_transformation/label_encoder.pkl")

    # --- feature schema ------------------------------------------------
    _NUM_COLS: List[str] = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    _CAT_COLS: List[str] = [
        "Stage_fear",
        "Drained_after_socializing",
        "P2",
    ]

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # Encoders & weights
        self.ordinal_enc = joblib.load(self._ORD_ENC)
        self.label_enc   = joblib.load(self._LBL_ENC)
        self.weights     = joblib.load(self._WEIGHT_PKL)  # {"xgb":0.6,"cat":0.4}

        # Determine which class index is Extrovert (0 or 1)
        self._extrovert_idx: int = int(
            np.where(self.label_enc.classes_ == "Extrovert")[0][0]
        )

        # XGBoost fold models
        self.xgb_models = [
            xgb.Booster(model_file=str(p))
            for p in sorted(self._MODEL_DIR.glob("xgb_fold*.bin"))
        ]
        if not self.xgb_models:
            raise FileNotFoundError("No XGB models found in model directory")
        self._col_order = self.xgb_models[0].feature_names  # preserve training order

        # CatBoost fold models (optional)
        self.cat_models = []
        for p in sorted(self._MODEL_DIR.glob("cat_fold*.cbm")):
            m = CatBoostClassifier()
            m.load_model(str(p))
            self.cat_models.append(m)

    # ------------------------------------------------------------------
    def _prepare_df(self, raw: Dict) -> pd.DataFrame:
        """Return DataFrame with ordinal‑encoded categorical columns."""
        df = pd.DataFrame([raw], columns=self._NUM_COLS + self._CAT_COLS)

        # numeric → float
        df[self._NUM_COLS] = df[self._NUM_COLS].astype(float)

        # fill missing categorical with 'Unknown'
        df[self._CAT_COLS] = df[self._CAT_COLS].fillna("Unknown")

        # ordinal encode (using fitted encoder)
        df[self._CAT_COLS] = self.ordinal_enc.transform(df[self._CAT_COLS])

        # align column order to what models expect
        df = df[self._col_order]

        return df

    # ------------------------------------------------------------------
    def _avg_xgb(self, dmat: xgb.DMatrix) -> float:
        """Mean probability across XGB folds (skip feature check)."""
        return float(
            np.mean([m.predict(dmat, validate_features=False)[0] for m in self.xgb_models])
        )

    def _avg_cat(self, df: pd.DataFrame) -> float:
        if not self.cat_models:
            return 0.0
        return float(
            np.mean([m.predict_proba(df)[:, 1][0] for m in self.cat_models])
        )

    # ------------------------------------------------------------------
    def predict(self, user_input: Dict) -> Dict:
        """
        Parameters
        ----------
        user_input : dict with the eight training features.
        Returns
        -------
        dict with keys:
            prediction               – \"Introvert\" or \"Extrovert\"
            probability_introvert    – float 0‑1
            probability_extrovert    – float 0‑1
        """
        df   = self._prepare_df(user_input)
        dmat = xgb.DMatrix(df)

        p_xgb = self._avg_xgb(dmat)
        p_cat = self._avg_cat(df)

        # Blended probability for class‑1 (XGBoost & CatBoost default)
        p_pos = self.weights["xgb"] * p_xgb + self.weights["cat"] * p_cat

        # Map to extrovert / introvert depending on encoder order
        if self._extrovert_idx == 1:
            p_ext = p_pos
            p_int = 1 - p_pos
        else:  # extrovert is class‑0
            p_ext = 1 - p_pos
            p_int = p_pos

        prediction_label = "Extrovert" if p_ext >= 0.5 else "Introvert"

        return {
            "prediction":            prediction_label,
            "probability_introvert": round(p_int, 4),
            "probability_extrovert": round(p_ext, 4),
        }

