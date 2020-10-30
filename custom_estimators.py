from sklearn.utils import resample, shuffle

  def get_resamples(X, y, n=3):
    mask = np.isin(y,[0,1,2])
    X_sub = X[mask]
    y_sub = y[mask]

    inv_mask = np.isin(y,[0,1,2],invert=True)
    X_keep = X[inv_mask]
    y_keep = y[inv_mask]

    Xs = []
    ys = []
    
    for it in range(n):
      X_resample, y_resample = resample(X_sub, y_sub, n_samples=len(y) - len(y_sub),stratify=y_sub,replace=False)
      cur_X = np.concatenate((X_resample, X_keep))
      cur_y = np.concatenate((y_resample, y_keep))

      Xs.append(cur_X)
      ys.append(cur_y)
    
    return Xs, ys

class CustomEstimator(BaseEstimator, TransformerMixin):
  def __init__(self, num_models = 5, model_type='xgb',**est_kwargs):
    if model_type == 'xgb':
      self.models = [XGBClassifier(**est_kwargs) for x in range(num_models)]
    if model_type == 'rf':
      self.models = [RandomForestClassifier(**est_kwargs) for x in range(num_models)]
    
    self.num_models = num_models
  
  def fit(self, X, y):
    mask = np.isin(y,[0,1,2])
    X_sub = X[mask]
    y_sub = y[mask]

    inv_mask = np.isin(y,[0,1,2],invert=True)
    X_keep = X[inv_mask]
    y_keep = y[inv_mask]

    for ind, model in enumerate(self.models):
      X_resample, y_resample = resample(X_sub, y_sub, n_samples=len(y) - len(y_sub),stratify=y_sub,replace=False)
      cur_X = np.concatenate((X_resample, X_keep))
      cur_y = np.concatenate((y_resample, y_keep))

      print('Training model ', ind+1)
      model.fit(cur_X, cur_y)
  
  def predict(self, X, y=None):
    preds_all = np.ndarray((len(X),self.num_models))
    preds = np.ndarray(len(X))
    for ind, model in enumerate(self.models):
      preds_all[:,ind] = model.predict(X)

    preds_all = preds_all.astype(int)

    for ind, pred_row in enumerate(preds_all):
      values, counts = np.unique(pred_row, return_counts=True)
      preds[ind] = values[np.argmax(counts)]
  
    return preds.astype(int)


class MultiModelCustomEstimator(BaseEstimator,TransformerMixin):
  def __init__(self, est_1, est_2, split_cols = ['PEF','RHOB','NPHI']):
    self.split_cols = split_cols
    self.est_1 = est_1
    self.est_2 = est_2
  
  def get_nonnull_indices(self, X, y=None):
    not_na = [True] * len(X)
    for col in self.split_cols:
      not_na = not_na & X[col].notna()
    not_na_indices = X.loc[not_na].index.tolist()

    return not_na_indices

  def fit(self, X, y):
    '''
    Assuming X, y are pandas objects, not numpy arrays - may change later
    '''
    not_na = [True] * len(X)
    for col in self.split_cols:
      not_na = not_na & X[col].notna()
    not_na_indices = X.loc[not_na].index.tolist()

    X_1 = X[X.index.isin(not_na_indices)].copy()
    y_1 = y[y.index.isin(not_na_indices)].copy()

    X_2 = X[~X.index.isin(not_na_indices)].copy()
    X_2 = X_2.drop(columns=self.split_cols)
    y_2 = y.loc[X_2.index].copy()

    self.est_1.fit(X_1, y_1)
    self.est_2.fit(X_2, y_2)


  def predict(self, X, y=None):
    not_na = [True] * len(X)
    for col in self.split_cols:
      not_na = not_na & X[col].notna()
    not_na_indices = X.loc[not_na].index.tolist()

    X_1 = X[X.index.isin(not_na_indices)].copy()
    X_2 = X[~X.index.isin(not_na_indices)].copy()
    X_2 = X_2.drop(columns=self.split_cols)

    pred_1 = self.est_1.predict(X_1)
    pred_2 = self.est_2.predict(X_2)

    # sklearn outputs numpy arrays only -
    # so those have to be re-assigned the original indices
    pred_1 = pd.Series(pred_1, index=X_1.index)
    pred_2 = pd.Series(pred_2, index=X_2.index)

    preds_concat = pd.concat([pred_1,pred_2])

    preds_concat = preds_concat.loc[X.index]

    return preds_concat


