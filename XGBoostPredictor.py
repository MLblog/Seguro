from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
from Predictor import Predictor
from FeatureEngineering import *


class XGBoostPredictor(Predictor):

    def __init__(self, input, params={}, name='XGBoost'):
        self.model = xgb.XGBRegressor()
        super().__init__(input, params, name=name)

    def train(self, params=None, num_boost_rounds=242):
        """
        A function that trains the predictor on the given dataset. Optionally accepts a set of parameters
        """

        # If no parameters are supplied use the default ones
        if not params:
            params = self.params

        self.preprocess()
        train, _ = self.split()
        y_train = train['target'].values
        x_train = train.drop('target', axis=1)

        self.params['base_score'] = np.mean(y_train)

        dtrain = xgb.DMatrix(x_train, y_train)
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_rounds)

    def predict(self, x_val):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        dtest = xgb.DMatrix(x_val)
        prediction = self.model.predict(dtest)
        return prediction

if __name__ == "__main__":

    # Test that the classifier works
    input = pd.read_csv('data/train.csv', na_values=(-1))


    ##### RUN XGBOOST
    print("\nSetting up data for XGBoost ...")

    params = {
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 1,
        'n_estimators': 200,
        #'missing': -1,
        'reg_lambda': 0.8,
        'silent': 1
    }

    model = XGBoostPredictor(input, params)
    model.create_submission(params)

    # # Tune Model
    # print("Tuning XGBoost...")
    # tuning_params = {
    #     'learning_rate': [0.05, 0.06],
    #     'silent': [1],
    #     'max_depth': [6],
    #     'subsample': [1],
    #     'reg_lambda': [0.8],
    #     'n_jobs': [8],
    #     'n_estimators': [200],
    #     'missing': [-1]
    # }
    # optimal_params, optimal_score = model.tune(tuning_params)
    #
    #
    #
    # # Train the model using the best set of parameters found by the gridsearch
    # print("\nTraining XGBoost ...")
    # model.train(params)
    #
    # print("\nEvaluating model...")
    # gini = model.evaluate()
    #
    # print("\n##########")
    # print("GINI score is: ", gini)
    # print("##########")