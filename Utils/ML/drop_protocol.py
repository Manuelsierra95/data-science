class DropProtocol:
    def __init__(self, df, drop_col_list, pred_col, model, plot_flag):
        self.df = df
        self.drop_col_list = drop_col_list
        self.pred_col = pred_col
        self.model = model
        self.plot_flag = plot_flag
        
    def get_Xy(self, df, pred_col):
        X = df.drop(pred_col, axis=1)
        y = df[pred_col]
        return X, y
    
    def train_model(self, X, y, model):
        pre_X, _feat_names = self.processor(X)
        X_train, X_test, y_train, y_test = train_test_split(pre_X, y)
        _model = model
        _model.fit(X_train, y_train)
        _feat_imp = model.feature_importances_
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
        if len(feat_importances) > 5:
            print('\033[31m' + f'Max subplots are set in 5, and u are trying to subplot {len(feat_importances)} (All_data count 1)' + '\033[0m')
        else:
            fig, ax = plt.subplots(ncols=1, nrows=len(feat_importances), figsize=(20,10), sharex=True)
            for x in range(len(feat_importances)):
                _ax = sns.barplot(ax=ax[x], data=feat_importances[x], x='importance', y=feat_importances[x].columns[0])
                _ax.set(title=f'Importance {feat_importances[x].columns[0]}', ylabel='', xlabel='')
            plt.show()
        
    def get_res_df(self, results):
        res_df = pd.DataFrame({'Score':results.values()},
                              index=results.keys())
        return res_df.sort_values(by='Score', ascending=False).head(20)
    
    def drop_protocol(self):
        print(f'Starting Drop-Protocol for a list of {len(self.drop_col_list)} columns.......')
        results = {}
        feat_importances = []
        X, y = self.get_Xy(self.df, self.pred_col)
        _score, _feat_imp, _feat_names = (self.train_model(X, y, self.model))
        results['All_data'] = _score
        feat_importances.append(self.get_feat_importances_df(_feat_imp, f'All_data / {_score:.4f}', _feat_names))
        for x in range(len(self.drop_col_list)):
            temp_df = self.df.drop(self.drop_col_list[x], axis=1)
            X, y = self.get_Xy(temp_df, self.pred_col)
            _score, _feat_imp, _feat_names = (self.train_model(X, y, self.model))
            results[f'dropping {self.drop_col_list[x]}'] = _score
            if self.plot_flag == True:
                feat_importances.append(self.get_feat_importances_df(_feat_imp, f'dropping {self.drop_col_list[x]} / {_score:.4f}', _feat_names))
        top_results = self.get_res_df(results)
        print(top_results)
        if self.plot_flag == True:
            self.plot_results(feat_importances)
        return top_results

# Test
# np.random.seed(42)
# drop_list = ['1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea'] 
# test1 = DropProtocol(temp, drop_list, 'SalePrice', CatBoostRegressor(logging_level='Silent'), True)
# test1.drop_protocol()