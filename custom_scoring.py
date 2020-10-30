import numpy as np
import pandas as pd

def score(y_true, y_pred):
    S = 0.0

    if isinstance(y_true, np.ndarray):
      y_true = y_true.astype(int)
    elif isinstance(y_true, pd.Series):
      y_true = y_true.to_numpy().astype(int)
    
    if isinstance(y_pred, np.ndarray):
      y_pred = y_pred.astype(int)
    elif isinstance(y_pred, pd.Series):
      y_pred = y_pred.to_numpy().astype(int)

    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]
