## Model training functions

def crnn_model(width=5, n_vars=11, n_classes=12, conv_kernel_size=3,
               conv_filters=3, lstm_units=3):
    input_shape = (width, n_vars)
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu'))
    #model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    model.add(Flatten())
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def train_model(data, estimator, prep_pipeline, target_col = 'lith_number', est_type='xgb',do_tuning=False, param_grid=None,
                use_weighting=True, train_test_stratify=True, do_cross_val=False, test_pct=0.2, split_by_well=False,
                test_wells=False):
  
  t = time.time()
  print('Prepping data...')
  data_model = prep_pipeline.fit_transform(data.drop(columns=[target_col]), data[target_col])
  data_local = data.copy()
  print('done')

  if split_by_well:
    well_list = data_local.WELL.unique().tolist()
    np.random.seed(0)
    if (test_wells):
      test_indices = data_local[data_local.WELL.isin(test_wells)].index
      train_indices = data_local[~data_local.WELL.isin(test_wells)].index
    else:
      test_wells = np.random.choice(well_list, int(test_pct*len(well_list)), replace=False)
      train_wells = well_list
      for well in test_wells:
        train_wells.remove(well)
    
      test_indices = data_local[data_local.WELL.isin(test_wells)].index
      train_indices = data_local[data_local.WELL.isin(train_wells)].index
    train_cols = data_model.columns.tolist()

    X_train = data_model.loc[train_indices].copy()
    y_train = data_local[target_col].loc[train_indices]
    X_test = data_model.loc[test_indices].copy()
    y_test = data_local[target_col].loc[test_indices]
  else:
    train_cols = data_model.columns.tolist()
    X = data_model#.to_numpy()
    y = data_local[target_col]#.to_numpy()

    if train_test_stratify:
      X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_pct,random_state=0, stratify=y)
    else:
      X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_pct,random_state=0)
    

    train_indices = X_train.index
    test_indices = X_test.index


  data_local['train'] = np.nan
  data_local['test'] = np.nan

  data_local.loc[train_indices,'train'] = 'train'
  data_local.loc[test_indices,'test'] = 'test'

  extra_kwargs = dict()
  if est_type == 'xgb':
    sample_weights = class_weight.compute_sample_weight('balanced',y_train)
    extra_kwargs = dict(sample_weight=sample_weights)

  print('Training...')
  estimator.fit(X_train, y_train, **extra_kwargs)
  print('done')
  print('Score on:')
  print('Test set:', score_skl(estimator, X_test,y_test))
  print('Train set:', score_skl(estimator, X_train, y_train))

  y_pred = estimator.predict(X_test)
  y_pred_name = [lithology_keys[cat_lith_dict[x]] for x in y_pred.tolist()]
  y_test_name = [lithology_keys[cat_lith_dict[x]] for x in y_test.tolist()]

  data_local['lith_number_pred'] = estimator.predict(data_model)
  print(classification_report(y_test_name, y_pred_name))

  if do_cross_val:
    scores = cross_val_score(estimator, X_train, y_train, cv=3)
    print('Cross val scores:', scores)

  if do_tuning:
    rand_search = RandomizedSearchCV(estimator, search_grid, scoring=score_skl, n_jobs=1, verbose=10, cv=3)
    estimator = rand_search.best_estimator_
  print('Total training time (seconds):', time.time()-t)

  data_local.to_csv(data_folder + 'data/data_w_split.csv',index=False)

  return estimator, prep_pipeline, train_cols

def get_scoring_preds(estimator, prep_pipeline, train_cols, outfile_path, write_results = True):
  scoring_data = pd.read_csv(data_folder + 'test.csv',sep=';')

  scoring_df = prep_pipeline.transform(scoring_data)

  preds = estimator.predict(scoring_df[train_cols])
  preds_cat = [cat_lith_dict[x] for x in preds]

  scoring_data['lith_number_pred'] = preds
  scoring_data['lith_mapped_pred'] = [lithology_keys[x] for x in preds_cat]

  if write_results:
    np.savetxt(outfile_path, preds_cat, comments='',header='lithology', fmt='%i')
  
  scoring_data.to_csv(data_folder + 'data/scoring_data_w_preds.csv',index=False)
  
  return preds

