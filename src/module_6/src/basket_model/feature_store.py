import pandas as pd

from src.module_6.src.basket_model.utils import features
from src.module_6.src.basket_model.utils import loaders
from src.module_6.src.exceptions import UserNotFoundException

import logging

logger = logging.getLogger(__name__)

# Configure the logger
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FeatureStore:
    def __init__(self):
        orders = loaders.load_orders()
        regulars = loaders.load_regulars()
        mean_item_price = loaders.get_mean_item_price()
        self.feature_store = (
            features.build_feature_frame(orders, regulars, mean_item_price)
            .set_index("user_id")
            .loc[
                :,
                [
                    "prior_basket_value",
                    "prior_item_count",
                    "prior_regulars_count",
                    "regulars_count",
                ],
            ]
        )

    def get_features(self, user_id: str) -> pd.DataFrame:
        try:
            features = self.feature_store.loc[user_id]
        except Exception as exception:
            logger.error(f"User not found in feature store: {exception}")
            raise UserNotFoundException(
                "User not found in feature store"
            ) from exception
        return features
