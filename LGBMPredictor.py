from lightgbm import LGBMClassifier

from Predictor import Predictor
from FeatureEngineering import *


class LGBMPredictor(Predictor):

    def __init__(self, train, test, params={}, name='LightGBM'):
        self.model = LGBMClassifier(**params)
        super().__init__(train, test, params, name=name)

    def set_params(self, params):
        self.model = LGBMClassifier(**params)

    def fit(self, params=None, train=None):
        """
        A function that trains the predictor on the given dataset. Optionally accepts a set of parameters
        """

        # If parameters are supplieod, override constructor one's
        if params is not None:
            self.set_params(params)

        if train is None:
            self.preprocess()
            train, _ = self.split()

        y_train = train['target'].values
        x_train = train.drop('target', axis=1).values
        self.model.fit(x_train, y_train)

    def predict(self, x_val):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        prediction = self.model.predict_proba(x_val)[:, 1]
        return prediction

if __name__ == "__main__":

    # Test that the classifier works
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')


    ##### RUN XGBOOST
    print("\nSetting up data for LightGBM ...")

    params = {
        'learning_rate': 0.01,
        'n_estimators': 1250,
        'max_depth': 10,
        'max_bin': 10,
        'subsample': 0.8,
        'subsample_freq': 10,
        'colsample_bytree': 0.8,
        'min_child_samples': 500
    }


    model = LGBMPredictor(train, test, params)
    # model.create_submission(params)

    # # Tune Model
    # print("Tuning LightGBM...")
    # tuning_params = {
    #     'learning_rate': [0.01],
    #     'n_estimators': [1250],
    #     'max_depth': [10],
    #     'max_bin': [10],
    #     'subsample': [0.8],
    #     'subsample_freq': [10],
    #     'colsample_bytree': [0.8],
    #     'min_child_samples': [500]
    # }
    # # optimal_params, optimal_score = model.tune(tuning_params)



    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining LightGBM ...")
    model.fit(params)

    print("\nEvaluating model...")
    gini = model.evaluate()

    print("\n##########")
    print("GINI score is: ", gini)
