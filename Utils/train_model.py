class Training_df:
    def __init__(self, df, pred_col, models, plot_flag):
        self.df = df
        self.pred_col = pred_col
        self.models = models
        self.plot_flag = plot_flag
        
    def get_Xy(self, df, pred_col):
        X = df.drop(pred_col, axis=1)
        y = df[pred_col]
        return X, y
    
    def train_model(self, X, y, model):
        pre_X, _feat_names = self.processor(X)
        X_train, X_test, y_train, y_test = train_test_split(pre_X, y)
        _model = model
        print(f'Trainig {_model} ..........')
        _model.fit(X_train, y_train)
        try: _feat_imp = model.feature_importances_
        except: _feat_imp = 0
        return _model.score(X_test, y_test), _feat_imp, _feat_names
    
    def processor(self, X):
        cat_features = X.select_dtypes(include=object).columns
        num_features = X.select_dtypes(include=np.number).columns

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_transformer, cat_features),
                ('num', num_transformer, num_features)
            ]
        )
        return preprocessor.fit_transform(X), preprocessor.get_feature_names_out()
    
    def get_feat_importances_df(self, _feat_imp, name, _feat_names):
        feat_imp = pd.DataFrame({f'features {name}':_feat_names,
                                 'importance':_feat_imp})
        feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(10)
        return feat_imp
    
    def plot_results(self, feat_importances):
        item_del = []
        f_imp = [x for x in feat_importances if x['importance'].mean() > 0]
        if len(f_imp) > 5:
            print('\033[31m' + f'Max subplots are set in 5, and u are trying to subplot {len(f_imp)} (All_data count 1)' + '\033[0m')
        else:
            fig, ax = plt.subplots(ncols=1, nrows=len(f_imp), figsize=(20,10))
            if len(f_imp) == 1:
                _ax = sns.barplot(data=f_imp[0], x='importance', y=f_imp[0].columns[0])
                _ax.set(title=f'Importance {f_imp[0].columns[0]}', ylabel='', xlabel='')
            else:
                for x in range(len(f_imp)):
                    _ax = sns.barplot(ax=ax[x], data=f_imp[x], x='importance', y=f_imp[x].columns[0])
                    _ax.set(title=f'Importance {f_imp[x].columns[0]}', ylabel='', xlabel='')
            plt.show()
        
    def get_res_df(self, results):
        res_df = pd.DataFrame({'Score':results.values()},
                              index=results.keys())
        return res_df.sort_values(by='Score', ascending=False).head(20)
    
    def start_training(self):
        print(f'Starting training for a list of {len(self.models)} models.......')
        results = {}
        feat_importances = []
        for x in range(len(self.models)):
            X, y = self.get_Xy(self.df, self.pred_col)
            _score, _feat_imp, _feat_names = (self.train_model(X, y, self.models[x]))
            results[f'{self.models[x]}'] = _score
            feat_importances.append(self.get_feat_importances_df(_feat_imp, f'{self.models[x]} / {_score:.4f}', _feat_names))
        print(self.get_res_df(results))
        if self.plot_flag == True:
            self.plot_results(feat_importances)
        return feat_importances