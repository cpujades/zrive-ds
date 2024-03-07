import pandas as pd
import os
import joblib
import datetime
from typing import Tuple
import json

from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from .utils import build_feature_frame, STORAGE_PATH


HOLDOUT_SIZE = 0.2
LABEL_COL = "outcome"
OUTPUT_PATH = os.path.join(STORAGE_PATH, "module_4_models")



def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    '''Split feature frame into feature matrix and label vector
   
    Args:
        df: pd.DataFrame
            Feature frame with a column "outcome" containing the label
            
    Returns:
        X: pd.DataFrame
            Feature matrix
        y: pd.Series
            Label vector
    '''
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return X, y


def save_model(model: BaseEstimator, model_name: str) -> None:
    '''Save model to disk.
    
    Args:
        model: BaseEstimator
            Model to be saved
        model_name: str
            Name of the model to be saved
    '''
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))
    
    
def train_test_split(df: pd.DataFrame, train_size: float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    '''Split feature frame into train and test sets by date
    
    Args:
        df: pd.DataFrame
            Feature frame
        train_size: float
            Proportion of the data to be used as training set
            
    Returns:
        X_train: pd.DataFrame
            Feature matrix for training
        y_train: pd.Series
            Label vector for training
        X_val: pd.DataFrame
            Feature matrix for validation
        y_val: pd.Series
            Label vector for validation
    '''
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
    cumsum_daily_orders = cumsum_daily_orders.reset_index()
    cutoff = cumsum_daily_orders.loc[cumsum_daily_orders["order_id"] <= train_size, "order_date"].max()

    
    X_train, y_train = feature_label_split(df[df.order_date <= cutoff])
    X_val, y_val = feature_label_split(df[df.order_date > cutoff])
    
    return X_train, y_train, X_val, y_val


def name_model() -> str:
    '''Returns the name of the model to be saved.   '''
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    model_name = f"push_{today}.pkl"
    return model_name


def handler_fit(event: dict, _) -> dict:
    '''
    Receives the model parametrisation, loads the data, trains the model and saves it to disk.
    
    Args:
        event: dict
            Event example:
                    "model_parametrisation": {
                        "classifier_parametrisation": {
                            "n_estimators": 100,
                            "max_depth": 10,
                            "random_state": 42
                        }
                    }
    Returns: a dict with the path to the model.
    '''
    
    model_parametrisation = event["model_parametrisation"]
    
    df = build_feature_frame()
    X, y = feature_label_split(df)
    
    clf = RandomForestClassifier(**model_parametrisation)
    clf.fit(X, y)
    
    model_name = name_model()
    
    save_model(clf, model_name)
    
    return {
        "statusCode": "200",
        "body": json.dumps({"model_path": [OUTPUT_PATH]})
        }
    