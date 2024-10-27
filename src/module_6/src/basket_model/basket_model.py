import os
import joblib
import numpy as np

from src.module_6.src.exceptions import PredictionException

import logging

logger = logging.getLogger(__name__)

# Configure the logger
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "bin/model.joblib")
)


class BasketModel:
    def __init__(self):
        self.model = joblib.load(MODEL)

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            pred = self.model.predict(features)
        except Exception as exception:
            logger.error(f"Error during model inference: {exception}")
            raise PredictionException("Error during model inference") from exception
        return pred
