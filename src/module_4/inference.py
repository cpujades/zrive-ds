import os
import pandas as pd
import json

from joblib import load
from .train import OUTPUT_PATH, feature_label_split
from .utils import build_feature_frame



def handler_predict(event: dict, _) -> dict:
    '''
    Receives a json with a field "users" containing a list of dictionaries with the same keys as the feature frame.
    
    Event example:
        {
            "user_id": {
                "feature 1": feature value,
                "feature 2": feature value, ...
                },
            "user_id2": {
                "feature 1": feature value,
                "feature 2": feature value, ...
                }.
        }
        
    Output is a json with a field "prediction" containing a dictionary with the user_id as key and the prediction as value.
        Output example:
        {
            "prediction": {
                "user_id": 1,
                "user_id2": 0
            }
        }
    '''
    data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"]))
    model = load(os.path.join(OUTPUT_PATH, event["model_name"]))
    
    df = build_feature_frame()
    X, y = feature_label_split(df)
    
    rf = load(os.path.join(OUTPUT_PATH, "push_2024_02_21.pkl"))
    
    preds = rf.predict(data_to_predict)
    
    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": preds.to_dict()}),
        }