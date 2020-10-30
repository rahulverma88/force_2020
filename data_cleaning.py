class SetRange(BaseEstimator, TransformerMixin):
  def __init__(self, range_dict):
    self.range_dict = range_dict

  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    X_ = X.copy()
    for col, range in self.range_dict.items():
      if len(range) > 0:
        col_min = range[0]
        X_.loc[X_[col] <= col_min, col] = col_min
      
      if len(range) > 1:
        col_max = range[1]
        X_.loc[X_[col] >= col_max, col] = col_max
    
    return X_


class DropCols(BaseEstimator, TransformerMixin):
  def __init__(self, drop_cols):
    self.drop_cols = drop_cols
  
  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    return X.drop(columns=self.drop_cols, errors='ignore')    

class FillNull(BaseEstimator, TransformerMixin):
  def __init__(self, cols, strategy='constant', grouped=False, fill_val = 'None'):
    self.cols = cols
    self.strategy = strategy
    self.grouped = grouped
    self.fill_val = fill_val

  def fit(self, X, y=None):
    return self
  
  def transform(self, X, y=None):
    X_ = X.copy()
    if self.strategy == 'constant':
      X_[self.cols] = X_[self.cols].fillna(value=self.fill_val)
      return X_
    
    if self.grouped:
      X_grouped = X_.groupby('WELL')
      if self.strategy == 'bffill':
        for col in self.cols:
          X_[col] = X_grouped[col].transform(lambda x: x.fillna(method='bfill').fillna(method='ffill'))
        return X_
      if self.strategy == 'rolling_mean':
        for col in self.cols:
          X_[col] = X_grouped[col].transform(lambda x: x.fillna(x.rolling(3).mean()).fillna(method='bfill').fillna(method='ffill'))
        return X_
      if self.strategy == 'mean':
        for col in self.cols:
          X_[col] = X_grouped[col].transform(lambda x: x.fillna(x.mean()))
        return X_
    else:
      if self.strategy == 'mean':
        for col in self.cols:
          X_[col] = X_[col].fillna(X_[col].mean())
        return X_


class CustomEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, cols, **enc_kwargs):
    self.cols = cols
    self.enc = OneHotEncoder(**enc_kwargs)

  def fit(self, X, y=None):
    self.enc.fit(X[self.cols])
    return self
  
  def transform(self, X, y=None):
    X_ = X.copy()
    enc_cols = self.enc.transform(X[self.cols])
    enc_col_names = self.enc.get_feature_names()
    enc_df = pd.DataFrame(data=enc_cols, columns=list(enc_col_names))
    X_ = X_.drop(columns=self.cols, errors='ignore')
    X_ = X_.join(enc_df)

    return X_


class CustomIterativeImputer(BaseEstimator, TransformerMixin):
  def __init__(self, target_cols = ['RHOB', 'RMED', 'DRHO', 'NPHI', 'RSHA'],
               exclude_cols = ['WELL','FORCE_2020_LITHOFACIES_LITHOLOGY','FORCE_2020_LITHOFACIES_CONFIDENCE'],
               **imputer_kwargs):
    self.target_cols = target_cols
    self.exclude_cols = exclude_cols
    self.imputer = IterativeImputer(**imputer_kwargs)

  def fit(self, X, y=None):
    X_ = X.copy()
    X_ = X_.drop(columns=self.exclude_cols, errors='ignore')
    self.imputer.fit(X_)
    
    return self
  
  def transform(self, X, y=None):
    X_ = X.copy()
    X_ = X_.drop(columns=self.exclude_cols, errors='ignore')
    X_trans = pd.DataFrame(self.imputer.transform(X_), columns=X_.columns)
    X_orig = X.copy()
    for col in X_trans:
      X_orig[col] = X_trans[col]
    
    return X_orig


class IsImputed(BaseEstimator, TransformerMixin):
  def __init__(self, ind_cols):
    self.ind_cols = ind_cols
  
  def fit(self, X, y=None):
    return self
  

class SpatialTargetEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, target_col):
    self.target_col = target_col
    self.form_models = dict()

  def fit(self, X, y):
    data = X.join(y)
    data_grpd_form = data.groupby(['WELL',self.target_col])
    data_grpd_well = data.groupby('WELL')

    lith_frac_df = data_grpd_form['lith_number'].agg(lambda x: (x == 0).sum()/len(x)).reset_index().rename(columns={'lith_number':'frac_0'})

    for num in range(1,12):
      col_name = 'frac_' + str(num)
      lith_frac_df = lith_frac_df.merge(data_grpd_form['lith_number'].agg(lambda x: (x == num).sum()/len(x)).reset_index().rename(columns={'lith_number':col_name}),
                on=['WELL',self.target_col])
    well_pos = pd.DataFrame()
    well_pos['X_LOC'] = data_grpd_well['X_LOC'].median()
    well_pos['Y_LOC'] = data_grpd_well['Y_LOC'].median()
    well_pos.reset_index(inplace=True)
    well_pos = well_pos.rename(columns={'index':'WELL'})
    lith_frac_df = lith_frac_df.merge(well_pos,on='WELL')
    lith_form_grpd = lith_frac_df.groupby(self.target_col)

    formations_list = data[self.target_col].unique().tolist()
    lith_list = list(range(12))

    for cur_form, cur_lith in itertools.product(formations_list,lith_list):
      cur_model = RandomForestRegressor(n_estimators=50, max_features=None,random_state=0)
      cur_data = lith_form_grpd.get_group(cur_form)
      cur_X = cur_data[['X_LOC','Y_LOC']]
      cur_y = cur_data['frac_' + str(cur_lith)]
      cur_model.fit(cur_X, cur_y)

      self.form_models[cur_form+'_' + str(cur_lith)] = cur_model
    
    return self
  
  def transform(self, X, y=None):
    X_ = X.copy()
    data_grpd_form = X_.groupby(['WELL',self.target_col])
    data_grpd_well = X_.groupby('WELL')

    for grp_name, grp in data_grpd_form:
      cur_well = grp_name[0]
      cur_form = grp_name[1]
      cur_well_df = data_grpd_well.get_group(cur_well)
      cur_x = cur_well_df['X_LOC'].median()
      cur_y = cur_well_df['Y_LOC'].median()

      cur_indices = grp.index

      for lith in range(12):
        cur_model = self.form_models[cur_form + '_' + str(lith)]
        cur_pred = cur_model.predict([[cur_x, cur_y]])[0]
        X_.loc[cur_indices, 'formation_lith_frac_' + str(lith)] = cur_pred
      
    return X_
      
class SegmentSplit(BaseEstimator, TransformerMixin):
  def __init__(self, width = 10, groupby_var = 'WELL', targ_var = 'lith_number'):
    self.width = width
    self.groupby_var = groupby_var
    self.targ_var = targ_var

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X_grouped = X.groupby(self.groupby_var)

    segs = []
    targs = []
    for _, grp in X_grouped:
      cur_targs = grp[self.targ_var].to_numpy()
      if self.width % 2 == 0:
        cent_ind = int(self.width/2 - 1)
      else:
        cent_ind = int(np.ceil(self.width/2))

      segs.extend(list(np.squeeze(view_as_windows(grp.to_numpy(),(self.width, grp.shape[1])))))
      targ_segs = view_as_windows
      targs.extend([seg[cent_ind] for seg in view_as_windows(cur_targs, self.width)])
    
    return segs, targs

def split_segments(data, return_targ = True, groupby_var = 'WELL', targ_var='lith_number',
                 width=5, drop_grp_var=True, scale_data=True):
  data_grouped = data.groupby(groupby_var)

  segs = []
  targs = []
  for _, grp in data_grouped:
    if return_targ:
      cur_targs = grp[targ_var].to_numpy()
      grp = grp.drop(columns=targ_var)

    if width % 2 == 0:
      cent_ind = int(width/2 - 1)
    else:
      cent_ind = int(np.ceil(width/2))

    if (drop_grp_var):
      grp = grp.drop(columns=groupby_var)
    
    if (scale_data):
      grp = scale(grp)

    if not isinstance(grp, np.ndarray):
      grp = grp.to_numpy()

    segs.extend(list(np.squeeze(view_as_windows(grp,(width, grp.shape[1])))))
    if return_targ:
      targ_segs = view_as_windows
      targs.extend([seg[cent_ind] for seg in view_as_windows(cur_targs, width)])
  
  return segs, targs

