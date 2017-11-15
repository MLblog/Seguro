import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from Predictor import Predictor
from LGBMPredictor import LGBMPredictor
from XGBoostPredictor import XGBoostPredictor
from FeatureEngineering import *


def preprocess(train, test):

    col = [c for c in train.columns if c not in ['id', 'target']]
    col = [c for c in col if not c.startswith('ps_calc_')]

    train = train.replace(-1, np.NaN)
    test = test.replace(-1, np.NaN)
    d_median = train.median(axis=0)
    d_mean = train.mean(axis=0)
    train = soft_impute(train)
    test = soft_impute(test)
    one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id', 'target']}

    def transform(df):
        df = pd.DataFrame(df)
        dcol = [c for c in df.columns if c not in ['id', 'target']]
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
        for c in dcol:
            if '_bin' not in c:
                df[c + str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
                df[c + str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

        for c in one_hot:
            if len(one_hot[c]) > 2 and len(one_hot[c]) < 7:
                for val in one_hot[c]:
                    df[c + '_oh_' + str(val)] = (df[c].values == val).astype(np.int)
        return df


    train = transform(train)
    test = transform(test)

    dups = train[train.duplicated(subset=col, keep=False)]

    train = train[~(train['id'].isin(dups['id'].values))]

    train = train[col + ['target']]
    test = test[col]
    return train, test

class EnsembleFromFiles(Predictor):
    def __init__(self, base_dir, target, stacker, name='Ensemble from base files'):
        self.base_dir = base_dir
        self.target = target
        self.model = stacker
        self.name = name
        self.train = pd.DataFrame()

    def read_train(self):
        all_files = os.listdir(self.base_dir)

        # Read and concatenate submissions
        outs = [pd.read_csv(os.path.join(self.base_dir, f), index_col=0) for f in all_files]
        train = pd.concat(outs, axis=1)
        cols = list(map(lambda x: "model_" + str(x), range(len(train.columns))))
        train.columns = cols
        return train

    def fit(self, params):
        if not self.train:
            self.train = self.read_train()

        self.model.fit(self.train, self.target)

        # Evaluate
        results = cross_val_score(self.model, features, target, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))
        persist_score(results.mean())



class Ensemble(Predictor):

    def __init__(self, train, test, n_splits, stacker, base_models, name='Ensemble'):
        self.n_splits = n_splits
        self.base_models = base_models
        self.model = stacker
        self.meta_features = None
        super().__init__(train, test, name=name)

    @staticmethod
    def read_dir(base_dir):
        """
        Reads every submission file found in a directory.

        Each file is expected to have the ID as the 1st column, followed by a base model's output in
        the 2nd column.
        :param base_dir: The root directory for submissions files
        :return: The concatenated pd.DataFrame
        """
        all_files = os.listdir(base_dir)
        outs = [pd.read_csv(os.path.join(base_dir, f), index_col=0) for f in all_files]
        concat = pd.concat(outs, axis=1)
        cols = list(map(lambda x: "model_" + str(x + 1), range(len(concat.columns))))
        concat.columns = cols
        return concat

    def fit_from_files(self, base_dir, target, evaluate=True):

        def persist_score(score, write_to='tuning.txt'):
            with open(write_to, "a") as f:
                f.write("------------------------------------------------\n")
                f.write("Model\t{}\n".format(self.name))
                f.write("Stacker score\t{}\nsubmissions: {}\nstacker: {}\n\n".format(score, base_dir, self.model))

        # Read the training set
        train = self.read_dir(base_dir)
        # Fit the stacker models using base predictions as meta-features
        self.model.fit(train, target)

        # Evaluate
        if evaluate:
            results = cross_val_score(self.model, train, target, cv=3, scoring='roc_auc')
            print("Stacker score: %.5f" % (results.mean()))
            persist_score(results.mean())

    def predict_from_files(self, base_dir):
        test = self.read_dir(base_dir)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(test)[:, 1]
        return self.model.predict(test)

    def fit(self):
        def persist_score(score, write_to='tuning.txt'):
            """
            Persists a set of parameters as well as their achieved score to a file.
            :param params: Parameters used
            :param score: Score achieved on the test set using params
            :param write_to: If passed, the optimal parameters found will be written to a file
            :return: Void
            """
            model_names = [clf.name for clf in self.base_models]
            with open(write_to, "a") as f:
                f.write("------------------------------------------------\n")
                f.write("Model\t{}\n".format(self.name))
                f.write("Stacker score\t{}\nmodels: {}\nstacker: {}\n\n".format(score, model_names, self.model))

        target = self.train['target']
        features = self.train.drop('target', axis=1)
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(features, target))

        def get_oof(clf, train, test):
            oof_train = np.zeros((train.shape[0],))
            oof_test_skf = np.empty((test.shape[0], self.n_splits))

            for i, (train_index, test_index) in enumerate(folds):
                print("{} fold {}".format(clf.name, i))
                tr = train.iloc[train_index]
                te = train.drop('target', axis=1).iloc[test_index]

                clf.fit(train=tr)

                oof_train[test_index] = clf.predict(te)
                oof_test_skf[:, i] = clf.predict(test)

            print("Shape is: {}".format(oof_test_skf.shape))
            oof_test = oof_test_skf.mean(axis=1)
            print("After averaging, shape is: {}".format(oof_test.shape))
            return oof_train, oof_test

        base_train = pd.DataFrame({'target': self.train['target']})
        base_test = pd.DataFrame({})
        for clf in self.base_models:
            print("About to train {}".format(clf.name))
            train, test = get_oof(clf, self.train, self.test)
            print(train.shape)
            print(test.shape)

            base_train[clf.name] = train
            base_test[clf.name] = test

        features = base_train.drop('target', axis=1)
        target = base_train['target']
        self.model.fit(features, target)
        self.meta_features = base_test

        # Evaluate
        results = cross_val_score(self.model, features, target, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))
        persist_score(results.mean())

    def predict(self, x_test=None):
        if x_test is None:
            x_test = self.meta_features

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x_test)[:, 1]

        return self.model.predict(x_test)

def from_file_test():
    """
    Check that fit and predict from submission files works as expected.
    """

    # Specify the path to our submission files and the actual target for the training set
    train_dir = 'training_submissions'
    test_dir = 'test_submissions'
    target = pd.read_csv('data/train.csv')['target']

    submission = pd.DataFrame()
    submission['id'] = pd.read_csv('data/test.csv')['id']

    # Meta learners. Can be either classifiers or regressors.
    log_model = LogisticRegression()
    linear_model = LinearRegression()

    # Create and train ensemble model. Cumbersome constructor since usual args are now meaningless.
    # Ensembling from files should have been a different class.
    ensemble = Ensemble(train=None, test=None, n_splits=3, stacker=log_model, base_models=None)
    ensemble.fit_from_files(train_dir, target)

    prediction = ensemble.predict_from_files(test_dir)

    # Create submission
    submission['target'] = prediction
    submission.to_csv('from_file_ensemble.csv', index=False)



if __name__ == '__main__':

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    sub = pd.DataFrame()
    sub['id'] = test['id']

    train, test = preprocess(train, test)

    xgb_params_1 = {
        'learning_rate': 0.02,
        'max_depth': 4,
        'subsample': 0.9,
        'n_estimators': 1500,
        'colsample_bytree': 0.9,
        'objective': 'binary:logistic',
        'min_child_weight': 10,
        'silent': 1
    }

    lgb_params_1 = {
        'learning_rate': 0.01,
        'n_estimators': 1250,
        'max_bin': 10,
        'subsample': 0.8,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_samples': 500
    }

    lgb_params_2 = {
        'learning_rate': 0.005,
        'n_estimators': 3700,
        'subsample': 0.7,
        'subsample_freq': 2,
        'colsample_bytree': 0.3,
        'num_leaves': 16
    }

    lgb_params_3 = {
        'learning_rate': 0.02,
        'n_estimators': 800,
        'max_depth': 4
    }



    def base_eval(base_model, train):
        train, val = train_test_split(train, test_size=0.3, random_state=42)
        base_model.fit(train=train)
        gini = base_model.evaluate(val=val)
        print("Model {} achieves: {} GINI".format(base_model.name, gini))

    # Base models
    lgb_model_1 = LGBMPredictor(train, test, lgb_params_1, name='LGB_1')
    lgb_model_2 = LGBMPredictor(train, test, lgb_params_2, name='LGB_2')
    lgb_model_3 = LGBMPredictor(train, test, lgb_params_3, name='LGB_3')
    xgb_model_1 = XGBoostPredictor(train, test, xgb_params_1, name='XGB_1')

    base_models = (xgb_model_1, lgb_model_1, lgb_model_2, lgb_model_3)

    # for base_model in base_models:
    #     base_eval(base_model, train)

    # Meta learners
    log_model = LogisticRegression()
    linear_model = LinearRegression()

    # Create and train ensemble model
    ensemble = Ensemble(train=train, test=test, n_splits=3, stacker=log_model, base_models=base_models)
    ensemble.fit()

    # Create submission
    pred = ensemble.predict()
    sub['target'] = pred
    sub.to_csv('stacked_2.csv', index=False)