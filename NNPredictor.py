from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from Predictor import Predictor
from FeatureEngineering import *


class NNPredictor(Predictor):

    def __init__(self, input, params={}, name='Neural Network'):
        super().__init__(input, params, name=name)
        self.model = MLPRegressor()

    def train(self, params=None):
        """
        A function that trains the predictor on the given dataset. Optionally accepts a set of parameters
        """
        if not params:
            params = self.params

        self.preprocess()
        train, _ = self.split()
        y_train = train['target'].values
        x_train = train.drop('target', axis=1)
        self.model = MLPRegressor(**params).fit(x_train, y_train)

    def predict(self, x_val):
        if not self.model:
            raise ValueError("The {} Predictor has not been trained yet".format(self.name))

        prediction = self.model.predict(x_val.values)
        return prediction

if __name__ == "__main__":

    # Test that the classifier works
    input = pd.read_csv('data/train.csv')

    print("\nSetting up data for Neural Network ...")

    # Play with the following params (manually or via a gridsearchCV tuner to optimize)
    # hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
    # learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
    # random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    # early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    params = {
        'hidden_layer_sizes': (100, 200, 100),
        'solver': 'sgd',
        'activation': 'logistic', # relu sucks. only god knows why
        'verbose': False,
        'warm_start': True,
        'alpha': 0.001,
        'learning_rate': 'invscaling',
        'power_t': 0.1,
        'learning_rate_init': 0.01,
        'max_iter': 125,
        'epsilon': 1e-08,
        'tol': 0.0001
    }

    model = NNPredictor(input, params)
    model.create_submission(params)

    # Tune Model
    tuning_params = {
        'hidden_layer_sizes': [(100, 200, 100)],
        'solver': ['sgd', 'adam'],
        'activation': ['logistic', 'relu'],
        'learning_rate': ['invscaling', 'constant'],
        'learning_rate_init': [0.01, 0.001],
        'power_t': [0.5],
        'tol': [0.0001]
    }

    print("Tuning neural network")
    optimal_params, optimal_score = model.tune(tuning_params)
    model.persist_tuning(score=optimal_score, params=optimal_params, write_to='tuning.txt')

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining NN ...")
    model.train()

    print("\nEvaluating NN...")
    gini = model.evaluate()
    print("\n##########")
    print("GINI score is: ", gini)
    print("##########")