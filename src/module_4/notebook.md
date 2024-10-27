```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.pipeline import Pipeline, make_pipeline

```


```python
df = pd.read_csv("../../../Module 2/feature_frame.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
info_cols = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
label_col = "outcome"
features_cols = [col for col in df.columns if col not in info_cols + [label_col]]

categorical_cols = ["product_type", "vendor"]
binary_cols = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]
numerical_cols = [col for col in features_cols if col not in categorical_cols + binary_cols]
```


```python
order_size = df.groupby("order_id").outcome.sum().reset_index()
big_orders = order_size.loc[order_size["outcome"] >= 5, "order_id"]
df_selected = df.loc[lambda x: x.order_id.isin(big_orders)]
df_selected
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2880541</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643241300100</td>
      <td>3864791220356</td>
      <td>2021-03-03 12:56:04</td>
      <td>2021-03-03 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880544</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643254800516</td>
      <td>3893722808452</td>
      <td>2021-03-03 13:19:28</td>
      <td>2021-03-03 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880545</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643274788996</td>
      <td>3883757174916</td>
      <td>2021-03-03 13:57:35</td>
      <td>2021-03-03 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880546</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643283734660</td>
      <td>3874925314180</td>
      <td>2021-03-03 14:14:24</td>
      <td>2021-03-03 00:00:00</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880547</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643294515332</td>
      <td>3906490826884</td>
      <td>2021-03-03 14:30:30</td>
      <td>2021-03-03 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
  </tbody>
</table>
<p>2163953 rows × 27 columns</p>
</div>




```python
df_selected['order_date'] = pd.to_datetime(df_selected['order_date'])
df_selected['created_at'] = pd.to_datetime(df_selected['created_at'])
df_selected.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2163953 entries, 0 to 2880547
    Data columns (total 27 columns):
     #   Column                            Dtype         
    ---  ------                            -----         
     0   variant_id                        int64         
     1   product_type                      object        
     2   order_id                          int64         
     3   user_id                           int64         
     4   created_at                        datetime64[ns]
     5   order_date                        datetime64[ns]
     6   user_order_seq                    int64         
     7   outcome                           float64       
     8   ordered_before                    float64       
     9   abandoned_before                  float64       
     10  active_snoozed                    float64       
     11  set_as_regular                    float64       
     12  normalised_price                  float64       
     13  discount_pct                      float64       
     14  vendor                            object        
     15  global_popularity                 float64       
     16  count_adults                      float64       
     17  count_children                    float64       
     18  count_babies                      float64       
     19  count_pets                        float64       
     20  people_ex_baby                    float64       
     21  days_since_purchase_variant_id    float64       
     22  avg_days_to_buy_variant_id        float64       
     23  std_days_to_buy_variant_id        float64       
     24  days_since_purchase_product_type  float64       
     25  avg_days_to_buy_product_type      float64       
     26  std_days_to_buy_product_type      float64       
    dtypes: datetime64[ns](2), float64(19), int64(4), object(2)
    memory usage: 462.3+ MB


    /var/folders/89/kh4sf6hs6xngc9mwq6wfcb_r0000gn/T/ipykernel_55548/2768600364.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_selected['order_date'] = pd.to_datetime(df_selected['order_date'])
    /var/folders/89/kh4sf6hs6xngc9mwq6wfcb_r0000gn/T/ipykernel_55548/2768600364.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_selected['created_at'] = pd.to_datetime(df_selected['created_at'])



```python
df.order_id.nunique() > df_selected.order_id.nunique()
```




    True




```python
daily_orders = df_selected.groupby('order_date').order_id.nunique()
daily_orders.head()
```




    order_date
    2020-10-05     3
    2020-10-06     7
    2020-10-07     6
    2020-10-08    12
    2020-10-09     4
    Name: order_id, dtype: int64




```python
plt.plot(daily_orders, label="daily orders")
plt.title("Daily Orders");
```


    
![png](notebook_files/notebook_9_0.png)
    


To avoid information leakage, like having the same orders in train/test, we make a temporal split. Also, due to the strong temporal component, it makes sense to split data in a chronological way and not randomly.


```python
cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
cumsum_daily_orders = cumsum_daily_orders.reset_index()

train_val_cutoff = cumsum_daily_orders.loc[cumsum_daily_orders["order_id"] <= 0.7, "order_date"].max()
val_test_cutoff = cumsum_daily_orders.loc[(cumsum_daily_orders["order_id"] <= 0.9), "order_date"].max()

print("Train since: ", cumsum_daily_orders.order_date.min())
print("Train until: ", train_val_cutoff)
print("Val until: ", val_test_cutoff)
print("Test until: ", cumsum_daily_orders.order_date.max())
```

    Train since:  2020-10-05 00:00:00
    Train until:  2021-02-04 00:00:00
    Val until:  2021-02-22 00:00:00
    Test until:  2021-03-03 00:00:00



```python
train_df = df_selected[df_selected.order_date <= train_val_cutoff]
val_df = df_selected[(df_selected.order_date > train_val_cutoff) & (df_selected.order_date <= val_test_cutoff)]
test_df = df_selected[df_selected.order_date > val_test_cutoff]
```

### Custom functions


```python
def plot_roc_and_pr_curves(model_name: str, y_test: pd.Series, y_pred: pd.Series, threshold_target: float=None, figure:Tuple[matplotlib.figure.Figure, np.array]=None):
    """
    Plots ROC and Precision-Recall curves side-by-side with AUC as legend.

    Args:
        model_name (str): The name of the classification model.
        y_test (array-like): True labels.
        y_preds (array-like): Predicted probabilities or scores.
        precision_target (float, optional): Desired precision level for highlighting.
        fig (matplotlib.figure.Figure, optional):  Figure object to plot on. 
                                                  Creates a new figure if not provided.
    """

    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax = figure

    # ROC Curve
    fpr, tpr, roc_th = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    ax[0].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend()

    # Precision-Recall Curve
    precision, recall, pr_th = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    ax[1].plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].legend()

    # Highlight precision target (if provided)
    if threshold_target is not None:
        idx_0 = np.argmin(np.abs(roc_th - threshold_target))
        ax[0].scatter(fpr[idx_0], tpr[idx_0])
        idx_1 = np.argmin(np.abs(pr_th - threshold_target))
        ax[1].scatter(recall[idx_1], precision[idx_1]) 
```


```python
def auc_score(y_test: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """
    Returns the ROC and PR curves AUC score for a given set of true labels and predicted probabilities.

    Args:
        y_test (array-like): True labels.
        y_preds (array-like): Predicted probabilities or scores.

    Returns:
        Tuple (float, float): ROC AUC score, PR curve AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    return roc_auc, pr_auc
```

### Baseline

We first try our baseline without any ML involved, just using a single column, `global_popularity` to predict the outcome.


```python
plot_roc_and_pr_curves("Baseline", y_pred=val_df["global_popularity"], y_test=val_df[label_col])
baseline_auc = auc_score(val_df[label_col], val_df["global_popularity"])
```


    
![png](notebook_files/notebook_18_0.png)
    


### Model training


```python
def feature_label_split(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_col], axis=1)
    y = df[label_col]
    return X, y

X_train, y_train = feature_label_split(train_df, label_col)
X_val, y_val = feature_label_split(val_df, label_col)
X_test, y_test = feature_label_split(test_df, label_col)
```


```python
train_cols = numerical_cols + binary_cols
```

### Linear Model

From the task of the previous module, we plot the best models for reference. We need to make sure our non-linear models (more complex models) improve on performance in order to justify the increase in model complexity.


```python
pr_auc_score_val = []
roc_auc_scores_val = []

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Validation metrics")

lrs = [
    make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1e-6)),
    make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=1e-4, solver='saga')),
]

names = ["Ridge 1e-6", "Lasso 1e-4"]
for name, lr in zip(names, lrs):
    lr.fit(X_train[train_cols], y_train)
    train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_roc_and_pr_curves(name, y_pred=train_proba, y_test=y_train, figure=(fig1, ax1))
    
    val_proba = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_roc_and_pr_curves(name, y_pred=val_proba, y_test=y_val, figure=(fig2, ax2))
    auc_scores_val = auc_score(y_val, val_proba)
    roc_auc_scores_val.append(auc_scores_val[0])
    pr_auc_score_val.append(auc_scores_val[1])
    
    coeffs = lr.named_steps["logisticregression"].coef_[0]
    feat_importance = pd.DataFrame({"feature": train_cols, "importance": np.abs(coeffs)})
    feat_importance = feat_importance.sort_values('importance', ascending=True)
    feat_importance.plot(kind='barh', x='feature', y='importance', title="Feature Importance", figsize=(9, 5))
    
    
plot_roc_and_pr_curves("Baseline", y_pred=val_df["global_popularity"], y_test=val_df[label_col], figure=(fig2, ax2))
```


    
![png](notebook_files/notebook_24_0.png)
    



    
![png](notebook_files/notebook_24_1.png)
    



    
![png](notebook_files/notebook_24_2.png)
    



    
![png](notebook_files/notebook_24_3.png)
    



```python
models = {
    "Model": ["Baseline", "Ridge 1e-6", "Lasso 1e-4"],
    "ROC AUC": [baseline_auc[0], roc_auc_scores_val[0], roc_auc_scores_val[1]],
    "PR AUC": [baseline_auc[1], pr_auc_score_val[0], pr_auc_score_val[1]]
    }
checkppoint_results = pd.DataFrame(models)
checkppoint_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>ROC AUC</th>
      <th>PR AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baseline</td>
      <td>0.787093</td>
      <td>0.066499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge 1e-6</td>
      <td>0.832559</td>
      <td>0.160177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso 1e-4</td>
      <td>0.834434</td>
      <td>0.148240</td>
    </tr>
  </tbody>
</table>
</div>



### Random Forest


```python
roc_scores_train = []
roc_scores_val = []
pr_scores_train = []
pr_scores_val = []

n_estimtors = [1, 10, 50, 100]
max_depths = [2, 5, 7, 9]

hyperparameters = []

for d in max_depths:
    for n in n_estimtors:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0)
        rf.fit(X_train[train_cols], y_train)
        train_proba = rf.predict_proba(X_train[train_cols])[:, 1]   
        auc_scores_train = auc_score(y_train, train_proba)
        roc_scores_train.append(auc_scores_train[0])
        pr_scores_train.append(auc_scores_train[1])
        val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
        auc_scores_val = auc_score(y_val, val_proba)
        roc_scores_val.append(auc_scores_val[0])
        pr_scores_val.append(auc_scores_val[1])
        
        hyperparameters.append((n, d))

results = pd.DataFrame({
    "Hyperparameters (estimators, depth)": hyperparameters,
    "ROC AUC Train": roc_scores_train,
    "PR AUC Train": pr_scores_train,
    "ROC AUC Val": roc_scores_val,
    "PR AUC Val": pr_scores_val})

results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hyperparameters (estimators, depth)</th>
      <th>ROC AUC Train</th>
      <th>PR AUC Train</th>
      <th>ROC AUC Val</th>
      <th>PR AUC Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(1, 2)</td>
      <td>0.536109</td>
      <td>0.263299</td>
      <td>0.594821</td>
      <td>0.206286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(10, 2)</td>
      <td>0.786365</td>
      <td>0.168654</td>
      <td>0.788705</td>
      <td>0.174788</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(50, 2)</td>
      <td>0.809042</td>
      <td>0.164210</td>
      <td>0.810103</td>
      <td>0.172845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(100, 2)</td>
      <td>0.808869</td>
      <td>0.162937</td>
      <td>0.811611</td>
      <td>0.171217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(1, 5)</td>
      <td>0.649386</td>
      <td>0.126528</td>
      <td>0.697842</td>
      <td>0.140273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(10, 5)</td>
      <td>0.829140</td>
      <td>0.187983</td>
      <td>0.829854</td>
      <td>0.188227</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(50, 5)</td>
      <td>0.832959</td>
      <td>0.193983</td>
      <td>0.834506</td>
      <td>0.192882</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(100, 5)</td>
      <td>0.833279</td>
      <td>0.193266</td>
      <td>0.834606</td>
      <td>0.193213</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(1, 7)</td>
      <td>0.804618</td>
      <td>0.164473</td>
      <td>0.806089</td>
      <td>0.157061</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(10, 7)</td>
      <td>0.833036</td>
      <td>0.203179</td>
      <td>0.834963</td>
      <td>0.192430</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(50, 7)</td>
      <td>0.836934</td>
      <td>0.208379</td>
      <td>0.838462</td>
      <td>0.197742</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(100, 7)</td>
      <td>0.837616</td>
      <td>0.208438</td>
      <td>0.838335</td>
      <td>0.198766</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(1, 9)</td>
      <td>0.798172</td>
      <td>0.186301</td>
      <td>0.780994</td>
      <td>0.152154</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(10, 9)</td>
      <td>0.834877</td>
      <td>0.220480</td>
      <td>0.832520</td>
      <td>0.189314</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(50, 9)</td>
      <td>0.842514</td>
      <td>0.228592</td>
      <td>0.841271</td>
      <td>0.198834</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(100, 9)</td>
      <td>0.843131</td>
      <td>0.228437</td>
      <td>0.841020</td>
      <td>0.200433</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.sort_values("ROC AUC Val", ascending=False).head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hyperparameters (estimators, depth)</th>
      <th>ROC AUC Train</th>
      <th>PR AUC Train</th>
      <th>ROC AUC Val</th>
      <th>PR AUC Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>(50, 9)</td>
      <td>0.842514</td>
      <td>0.228592</td>
      <td>0.841271</td>
      <td>0.198834</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(100, 9)</td>
      <td>0.843131</td>
      <td>0.228437</td>
      <td>0.841020</td>
      <td>0.200433</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(50, 7)</td>
      <td>0.836934</td>
      <td>0.208379</td>
      <td>0.838462</td>
      <td>0.197742</td>
    </tr>
  </tbody>
</table>
</div>



After trying different hyperparameters combinations we pick the one with `n_estimator=100` and `max_depth=9`. As seen from the table above (sorted by `ROC AUC Val`) those hyperparameters are't the best, however they yield the best PR AUC score, while only sacrificing 0.000251.

From the table with all the data we see that the improvement we get from 50 trees to 100 is very little, so trying with 150 is not worth it. However, the improvement leap in depth from 7 to 9 mayindicate that we can squeeze out a bit more performance by increasing the depth. Let's try with 100 and 150 trees for 9, 11 and 13 depths.


```python
n_est = [100, 150]
depths = [9, 11, 13]

for n in n_est:
    for d in depths:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0)
        rf.fit(X_train[train_cols], y_train)
        train_proba = rf.predict_proba(X_train[train_cols])[:, 1]
        auc_scores_train = auc_score(y_train, train_proba)
        print(f"Estimators {n}, depth {d}. Train AUC: {auc_scores_train}")

        val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
        auc_scores_val = auc_score(y_val, val_proba)
        print(f"Estimators {n}, depth {d}. Validation AUC: {auc_scores_val}")

```

    Estimators 100, depth 9. Train AUC: (0.8431308542157552, 0.22843724976492064)
    Estimators 100, depth 9. Validation AUC: (0.841019723103303, 0.2004329237782617)
    Estimators 100, depth 11. Train AUC: (0.8510575200728553, 0.25673821728420076)
    Estimators 100, depth 11. Validation AUC: (0.8443984795202771, 0.19989639803276882)
    Estimators 100, depth 13. Train AUC: (0.8631656258184507, 0.29411427996351847)
    Estimators 100, depth 13. Validation AUC: (0.8437744722414929, 0.19822915066240693)
    Estimators 150, depth 9. Train AUC: (0.8432346790365738, 0.22881234769151487)
    Estimators 150, depth 9. Validation AUC: (0.8410070722695171, 0.2005685047376905)
    Estimators 150, depth 11. Train AUC: (0.851501087061485, 0.25704765292295273)
    Estimators 150, depth 11. Validation AUC: (0.8444610396081079, 0.20021036219592903)
    Estimators 150, depth 13. Train AUC: (0.8638383159973727, 0.2951477760867607)
    Estimators 150, depth 13. Validation AUC: (0.8441732659744031, 0.19837703347620753)


The improvements we get by increasing the number of trees and depth are not very significative, thus not worth it the increase in execution time. We stick with `n_estimator=100` and `max_depth=9`.


```python
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Validation metrics")

rf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
rf.fit(X_train[train_cols], y_train)
train_proba = rf.predict_proba(X_train[train_cols])[:, 1]
plot_roc_and_pr_curves("RF", y_pred=train_proba, y_test=y_train, figure=(fig1, ax1))

val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
plot_roc_and_pr_curves("RF", y_pred=val_proba, y_test=y_val, figure=(fig2, ax2))

coeffs = rf.feature_importances_
feat_importance = pd.DataFrame({"feature": train_cols, "importance": coeffs})
feat_importance = feat_importance.sort_values('importance', ascending=True)
feat_importance.plot(kind='barh', x='feature', y='importance', title="Feature Importance", figsize=(9, 5))
        
plot_roc_and_pr_curves("Baseline", y_pred=val_df["global_popularity"], y_test=val_df[label_col], figure=(fig2, ax2))
```


    
![png](notebook_files/notebook_32_0.png)
    



    
![png](notebook_files/notebook_32_1.png)
    



    
![png](notebook_files/notebook_32_2.png)
    


### Gradient Boosting


```python
roc_scores_train = []
roc_scores_val = []
pr_scores_train = []
pr_scores_val = []

n_estimtors = [1, 10, 50, 100]
max_depths = [2, 5, 7, 9]
learnning_rate = [0.001, 0.01, 0.1]

hyper_estimators = []
hyper_depths = []
hyper_lr = []

for n in n_estimtors:
    for d in max_depths:
        for lr in learnning_rate:
            gbt = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
            gbt.fit(X_train[train_cols], y_train)
            train_proba = gbt.predict_proba(X_train[train_cols])[:, 1]   
            auc_scores_train = auc_score(y_train, train_proba)
            roc_scores_train.append(auc_scores_train[0])
            pr_scores_train.append(auc_scores_train[1])
            val_proba = gbt.predict_proba(X_val[train_cols])[:, 1]
            auc_scores_val = auc_score(y_val, val_proba)
            roc_scores_val.append(auc_scores_val[0])
            pr_scores_val.append(auc_scores_val[1])
            
            hyper_estimators.append(n)
            hyper_depths.append(d)
            hyper_lr.append(lr)

results = pd.DataFrame({
    "Estimators": hyper_estimators,
    "Depth": hyper_depths,
    "learnning_rate": hyper_lr,
    "ROC AUC Train": roc_scores_train,
    "PR AUC Train": pr_scores_train,
    "ROC AUC Val": roc_scores_val,
    "PR AUC Val": pr_scores_val})

results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estimators</th>
      <th>Depth</th>
      <th>learnning_rate</th>
      <th>ROC AUC Train</th>
      <th>PR AUC Train</th>
      <th>ROC AUC Val</th>
      <th>PR AUC Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>0.001</td>
      <td>0.646157</td>
      <td>0.236856</td>
      <td>0.654523</td>
      <td>0.227075</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.010</td>
      <td>0.646157</td>
      <td>0.236856</td>
      <td>0.654523</td>
      <td>0.227075</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>0.100</td>
      <td>0.646157</td>
      <td>0.236856</td>
      <td>0.654523</td>
      <td>0.227075</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>0.001</td>
      <td>0.817115</td>
      <td>0.198580</td>
      <td>0.818213</td>
      <td>0.170498</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.010</td>
      <td>0.817115</td>
      <td>0.198580</td>
      <td>0.818213</td>
      <td>0.170498</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>5</td>
      <td>0.100</td>
      <td>0.817115</td>
      <td>0.198580</td>
      <td>0.818213</td>
      <td>0.170498</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>7</td>
      <td>0.001</td>
      <td>0.836078</td>
      <td>0.200076</td>
      <td>0.835840</td>
      <td>0.170370</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>7</td>
      <td>0.010</td>
      <td>0.836078</td>
      <td>0.200076</td>
      <td>0.835842</td>
      <td>0.170508</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>7</td>
      <td>0.100</td>
      <td>0.836078</td>
      <td>0.200076</td>
      <td>0.835842</td>
      <td>0.170508</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>9</td>
      <td>0.001</td>
      <td>0.840869</td>
      <td>0.213039</td>
      <td>0.834571</td>
      <td>0.159354</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>9</td>
      <td>0.010</td>
      <td>0.840868</td>
      <td>0.213031</td>
      <td>0.834853</td>
      <td>0.159064</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>9</td>
      <td>0.100</td>
      <td>0.840868</td>
      <td>0.212987</td>
      <td>0.834571</td>
      <td>0.158488</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>2</td>
      <td>0.001</td>
      <td>0.646392</td>
      <td>0.207950</td>
      <td>0.654688</td>
      <td>0.198005</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10</td>
      <td>2</td>
      <td>0.010</td>
      <td>0.646392</td>
      <td>0.207950</td>
      <td>0.654688</td>
      <td>0.198005</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10</td>
      <td>2</td>
      <td>0.100</td>
      <td>0.815752</td>
      <td>0.192137</td>
      <td>0.810467</td>
      <td>0.174692</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10</td>
      <td>5</td>
      <td>0.001</td>
      <td>0.817115</td>
      <td>0.198580</td>
      <td>0.818213</td>
      <td>0.170498</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10</td>
      <td>5</td>
      <td>0.010</td>
      <td>0.832832</td>
      <td>0.191802</td>
      <td>0.836669</td>
      <td>0.170446</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10</td>
      <td>5</td>
      <td>0.100</td>
      <td>0.839141</td>
      <td>0.199944</td>
      <td>0.842513</td>
      <td>0.182876</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10</td>
      <td>7</td>
      <td>0.001</td>
      <td>0.836211</td>
      <td>0.200246</td>
      <td>0.836446</td>
      <td>0.171171</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10</td>
      <td>7</td>
      <td>0.010</td>
      <td>0.838486</td>
      <td>0.202912</td>
      <td>0.840530</td>
      <td>0.173908</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>7</td>
      <td>0.100</td>
      <td>0.843334</td>
      <td>0.220594</td>
      <td>0.844661</td>
      <td>0.176982</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10</td>
      <td>9</td>
      <td>0.001</td>
      <td>0.841099</td>
      <td>0.213809</td>
      <td>0.835589</td>
      <td>0.158253</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10</td>
      <td>9</td>
      <td>0.010</td>
      <td>0.842313</td>
      <td>0.219327</td>
      <td>0.839103</td>
      <td>0.163859</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10</td>
      <td>9</td>
      <td>0.100</td>
      <td>0.850337</td>
      <td>0.262643</td>
      <td>0.843740</td>
      <td>0.149277</td>
    </tr>
    <tr>
      <th>24</th>
      <td>50</td>
      <td>2</td>
      <td>0.001</td>
      <td>0.646392</td>
      <td>0.207950</td>
      <td>0.654688</td>
      <td>0.198005</td>
    </tr>
    <tr>
      <th>25</th>
      <td>50</td>
      <td>2</td>
      <td>0.010</td>
      <td>0.647053</td>
      <td>0.201688</td>
      <td>0.655241</td>
      <td>0.189741</td>
    </tr>
    <tr>
      <th>26</th>
      <td>50</td>
      <td>2</td>
      <td>0.100</td>
      <td>0.837213</td>
      <td>0.185069</td>
      <td>0.841849</td>
      <td>0.179232</td>
    </tr>
    <tr>
      <th>27</th>
      <td>50</td>
      <td>5</td>
      <td>0.001</td>
      <td>0.832144</td>
      <td>0.192190</td>
      <td>0.836013</td>
      <td>0.170808</td>
    </tr>
    <tr>
      <th>28</th>
      <td>50</td>
      <td>5</td>
      <td>0.010</td>
      <td>0.836110</td>
      <td>0.195955</td>
      <td>0.840121</td>
      <td>0.179777</td>
    </tr>
    <tr>
      <th>29</th>
      <td>50</td>
      <td>5</td>
      <td>0.100</td>
      <td>0.845048</td>
      <td>0.219997</td>
      <td>0.846809</td>
      <td>0.179914</td>
    </tr>
    <tr>
      <th>30</th>
      <td>50</td>
      <td>7</td>
      <td>0.001</td>
      <td>0.836718</td>
      <td>0.200965</td>
      <td>0.837204</td>
      <td>0.172959</td>
    </tr>
    <tr>
      <th>31</th>
      <td>50</td>
      <td>7</td>
      <td>0.010</td>
      <td>0.840124</td>
      <td>0.209118</td>
      <td>0.842499</td>
      <td>0.178410</td>
    </tr>
    <tr>
      <th>32</th>
      <td>50</td>
      <td>7</td>
      <td>0.100</td>
      <td>0.850913</td>
      <td>0.260989</td>
      <td>0.846062</td>
      <td>0.163007</td>
    </tr>
    <tr>
      <th>33</th>
      <td>50</td>
      <td>9</td>
      <td>0.001</td>
      <td>0.841501</td>
      <td>0.215622</td>
      <td>0.837210</td>
      <td>0.160210</td>
    </tr>
    <tr>
      <th>34</th>
      <td>50</td>
      <td>9</td>
      <td>0.010</td>
      <td>0.845091</td>
      <td>0.231142</td>
      <td>0.842841</td>
      <td>0.171484</td>
    </tr>
    <tr>
      <th>35</th>
      <td>50</td>
      <td>9</td>
      <td>0.100</td>
      <td>0.863701</td>
      <td>0.329992</td>
      <td>0.842299</td>
      <td>0.131601</td>
    </tr>
    <tr>
      <th>36</th>
      <td>100</td>
      <td>2</td>
      <td>0.001</td>
      <td>0.646392</td>
      <td>0.207950</td>
      <td>0.654688</td>
      <td>0.198005</td>
    </tr>
    <tr>
      <th>37</th>
      <td>100</td>
      <td>2</td>
      <td>0.010</td>
      <td>0.796307</td>
      <td>0.183393</td>
      <td>0.785086</td>
      <td>0.166477</td>
    </tr>
    <tr>
      <th>38</th>
      <td>100</td>
      <td>2</td>
      <td>0.100</td>
      <td>0.840020</td>
      <td>0.190637</td>
      <td>0.844664</td>
      <td>0.184307</td>
    </tr>
    <tr>
      <th>39</th>
      <td>100</td>
      <td>5</td>
      <td>0.001</td>
      <td>0.833352</td>
      <td>0.191831</td>
      <td>0.837169</td>
      <td>0.170209</td>
    </tr>
    <tr>
      <th>40</th>
      <td>100</td>
      <td>5</td>
      <td>0.010</td>
      <td>0.837331</td>
      <td>0.200160</td>
      <td>0.841696</td>
      <td>0.182182</td>
    </tr>
    <tr>
      <th>41</th>
      <td>100</td>
      <td>5</td>
      <td>0.100</td>
      <td>0.847104</td>
      <td>0.232739</td>
      <td>0.846499</td>
      <td>0.177450</td>
    </tr>
    <tr>
      <th>42</th>
      <td>100</td>
      <td>7</td>
      <td>0.001</td>
      <td>0.838738</td>
      <td>0.202689</td>
      <td>0.840921</td>
      <td>0.173460</td>
    </tr>
    <tr>
      <th>43</th>
      <td>100</td>
      <td>7</td>
      <td>0.010</td>
      <td>0.841505</td>
      <td>0.215009</td>
      <td>0.844058</td>
      <td>0.183511</td>
    </tr>
    <tr>
      <th>44</th>
      <td>100</td>
      <td>7</td>
      <td>0.100</td>
      <td>0.853768</td>
      <td>0.284612</td>
      <td>0.843815</td>
      <td>0.159704</td>
    </tr>
    <tr>
      <th>45</th>
      <td>100</td>
      <td>9</td>
      <td>0.001</td>
      <td>0.842641</td>
      <td>0.219288</td>
      <td>0.839550</td>
      <td>0.163182</td>
    </tr>
    <tr>
      <th>46</th>
      <td>100</td>
      <td>9</td>
      <td>0.010</td>
      <td>0.846645</td>
      <td>0.239594</td>
      <td>0.844414</td>
      <td>0.175713</td>
    </tr>
    <tr>
      <th>47</th>
      <td>100</td>
      <td>9</td>
      <td>0.100</td>
      <td>0.869113</td>
      <td>0.374962</td>
      <td>0.839551</td>
      <td>0.118623</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.sort_values("ROC AUC Val", ascending=False).head(9)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estimators</th>
      <th>Depth</th>
      <th>learnning_rate</th>
      <th>ROC AUC Train</th>
      <th>PR AUC Train</th>
      <th>ROC AUC Val</th>
      <th>PR AUC Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>50</td>
      <td>5</td>
      <td>0.10</td>
      <td>0.845048</td>
      <td>0.219997</td>
      <td>0.846809</td>
      <td>0.179914</td>
    </tr>
    <tr>
      <th>41</th>
      <td>100</td>
      <td>5</td>
      <td>0.10</td>
      <td>0.847104</td>
      <td>0.232739</td>
      <td>0.846499</td>
      <td>0.177450</td>
    </tr>
    <tr>
      <th>32</th>
      <td>50</td>
      <td>7</td>
      <td>0.10</td>
      <td>0.850913</td>
      <td>0.260989</td>
      <td>0.846062</td>
      <td>0.163007</td>
    </tr>
    <tr>
      <th>38</th>
      <td>100</td>
      <td>2</td>
      <td>0.10</td>
      <td>0.840020</td>
      <td>0.190637</td>
      <td>0.844664</td>
      <td>0.184307</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10</td>
      <td>7</td>
      <td>0.10</td>
      <td>0.843334</td>
      <td>0.220594</td>
      <td>0.844661</td>
      <td>0.176982</td>
    </tr>
    <tr>
      <th>46</th>
      <td>100</td>
      <td>9</td>
      <td>0.01</td>
      <td>0.846645</td>
      <td>0.239594</td>
      <td>0.844414</td>
      <td>0.175713</td>
    </tr>
    <tr>
      <th>43</th>
      <td>100</td>
      <td>7</td>
      <td>0.01</td>
      <td>0.841505</td>
      <td>0.215009</td>
      <td>0.844058</td>
      <td>0.183511</td>
    </tr>
    <tr>
      <th>44</th>
      <td>100</td>
      <td>7</td>
      <td>0.10</td>
      <td>0.853768</td>
      <td>0.284612</td>
      <td>0.843815</td>
      <td>0.159704</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10</td>
      <td>9</td>
      <td>0.10</td>
      <td>0.850337</td>
      <td>0.262643</td>
      <td>0.843740</td>
      <td>0.149277</td>
    </tr>
  </tbody>
</table>
</div>



From the previous table we see that the one with the best ROC AUC value isn't the best with the best PR AUC value. Therefore we choose the one with best PR AUC Val score (`n_estimators=100, max_depth=2, learning_rate=0.1`) that is only a bit worst than the first one in terms of ROC AUC score.

### Final conclusions


```python
pr_auc_score_val = []
roc_auc_scores_val = []

fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Validation metrics")

lrs = [
    make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1e-6)),
    make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', C=1e-4, solver='saga')),
    make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=9)),
    make_pipeline(GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1)),
]

names = ["Ridge 1e-6", "Lasso 1e-4", "Random Forest", "Gradient Boosting"]
for name, lr in zip(names, lrs):
    lr.fit(X_train[train_cols], y_train)
    train_proba = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_roc_and_pr_curves(name, y_pred=train_proba, y_test=y_train, figure=(fig1, ax1))
    
    val_proba = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_roc_and_pr_curves(name, y_pred=val_proba, y_test=y_val, threshold_target=0.05, figure=(fig2, ax2))
    auc_scores_val = auc_score(y_val, val_proba)
    roc_auc_scores_val.append(auc_scores_val[0])
    pr_auc_score_val.append(auc_scores_val[1])
    
    
plot_roc_and_pr_curves("Baseline", y_pred=val_df["global_popularity"], y_test=val_df[label_col], figure=(fig2, ax2))
```


    
![png](notebook_files/notebook_38_0.png)
    



    
![png](notebook_files/notebook_38_1.png)
    



```python
models = {
    "Model": ["Baseline"] + names,
    "ROC AUC": [baseline_auc[0]] + roc_auc_scores_val,
    "PR AUC": [baseline_auc[1]] + pr_auc_score_val
    }
final_results = pd.DataFrame(models)
final_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>ROC AUC</th>
      <th>PR AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Baseline</td>
      <td>0.787093</td>
      <td>0.066499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge 1e-6</td>
      <td>0.832559</td>
      <td>0.160177</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso 1e-4</td>
      <td>0.834434</td>
      <td>0.148240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>0.840997</td>
      <td>0.198718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gradient Boosting</td>
      <td>0.844664</td>
      <td>0.184307</td>
    </tr>
  </tbody>
</table>
</div>



 Focusing on the AUC scores, we can see that RF is ahead of GBT, although not by much. If we check in the chart above, we can see that both ROC Curves are very similar. However, the thing changes when we focus on the PR Curve. With the RF ahead until a recall of ~0.4, from there, the two lines are similar.
