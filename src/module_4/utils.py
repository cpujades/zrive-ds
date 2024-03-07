import os
import pandas as pd


STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "Module 4"))


def load_dataset() -> pd.DataFrame:
    '''Load the dataset from the storage path'''
    dataset_name = "feature_frame.csv"
    loading_file = os.path.join(STORAGE_PATH, dataset_name)
    df = pd.read_csv(loading_file)
    return df


def push_relevant_orders(df: pd.DataFrame, min_products: int=5) -> pd.DataFrame:
    '''Interested in orders with at least 5 products'''
    order_size = df.groupby("order_id").order_id.nunique()
    orders_of_min_size = order_size[order_size >= min_products].index
    big_orders = df.loc[lambda x: x["order_id"].isin(orders_of_min_size)]
    return big_orders


def build_feature_frame() -> pd.DataFrame:
    '''Build the feature frame for the model training'''
    df = load_dataset()
    df_orders = push_relevant_orders(df)
    df_orders["created_at"] = pd.to_datetime(df_orders["created_at"])
    df_orders["order_date"] = pd.to_datetime(df_orders["order_date"]).dt.date   
    return df_orders