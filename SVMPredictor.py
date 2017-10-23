from sklearn.neural_network import MLPRegressor
from Predictor import Predictor
from FeatureEngineering import *
from sklearn.svm import SVR
import numpy as np

class SVMPredictor(Predictor):

    def __init__(self, train, test, params={}, name='Support Vector Machine'):
        super().__init__(train, test, params, name=name)
        self.model = SVR()

    def preprocess(self, threshold=50):
        """
        A function that, given the raw dataset creates a feature vector.
        Feature Engineering, cleaning and imputation goes here
        """
        self.train = dummy_conversion(self.train, threshold)
        self.train = normalize(self.train)

    def fit(self, params=None):
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
        self.model = SVR(**params).fit(x_train, y_train)

    def predict(self, x_val):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        prediction = self.model.predict(x_val.values)
        return prediction

if __name__ == "__main__":

    # Test that the classifier works
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')


    ##### RUN XGBOOST
    print("\nSetting up data for SVM ...")

    params = {
        'kernel': 'rbf',
        'degree': 3,
        'gamma': 'auto',
        'coef0': 0.0,
        'C': 1.0,
        'epsilon': 0.1,
        'shrinking': True,
        'cache_size': 2000,
        'max_iter': 100
    }

    model = SVMPredictor(train, test, params)
    #model.create_submission(params)

    # Tune Model
    # kernel =’rbf’, degree = 3, gamma =’auto’, coef0 = 0.0, tol = 0.001, C = 1.0,
    # epsilon = 0.1, shrinking = True, cache_size = 200, verbose = False, max_iter = -1

    print("Tuning SVM...")
    tuning_params = {
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [3],
        'gamma': ['auto'],
        'coef0': [0.0],
        'C': [1.0],
        'epsilon': [0.1],
        'shrinking': [True, False],
        'cache_size': [2000],
        'max_iter': [100, 500]
    }

    optimal_params, optimal_score = model.tune(tuning_params)



    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining SVM ...")
    model.fit(params)

    print("\nEvaluating model...")
    gini = model.evaluate()

    print("\n##########")
    print("GINI score is: ", gini)
    print("##########")