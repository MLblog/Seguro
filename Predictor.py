from abc import abstractmethod, ABCMeta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from FeatureEngineering import *
from sklearn.model_selection import GridSearchCV

"""
An abstract class modeling our notion of a predictor.
Concrete implementations should follow the predictors
interface
"""
class Predictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, input, params={}, name=None):
        """
        Base constructor

        :param train: a pd.DataFrame of IDs, raw features and the target variable
        :param params: a dictionary of named model parameters
        :param name: Optional model name, used for logging
        """
        self.input = input
        self.params = params
        self.name = name

    def set_params(self, params):
        """Override parameters set in the constructor. Dictionary expected"""
        self.params = params

    def split(self, test_size=0.3):
        """
        Splits the merged input (features + labels) into a training and a validation set

        :return: a tuple of training and test sets with the given size ratio
        """
        train, test = train_test_split(self.input, test_size=test_size, random_state=42)
        return train, test

    @abstractmethod
    def train(self):
        """
        A function that trains the predictor on the given dataset.
        """

    @abstractmethod
    def predict(self, x_test):
        """
        Predicts the label for the given input
        :param x_test: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    def persist_tuning(self, score, params, write_to):
        """
        Persists a set of parameters as well as their achieved score to a file.
        :param params: Parameters used
        :param score: Score achieved on the test set using params
        :param write_to: If passed, the optimal parameters found will be written to a file
        :return: Void
        """
        with open(write_to, "a") as f:
            f.write("------------------------------------------------\n")
            f.write("Model\t{}\n".format(self.name))
            f.write("Best MAE\t{}\nparams: {}\n\n".format(score, params))

    @timing
    def tune(self, params, nfolds=3, verbose=3):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation.
        :param verbose: Verbosity level. 0 is silent, higher int prints more stuff
        :return: Dict of best parameters found.
        """
        self.preprocess()
        train, _ = self.split()
        y_train = train['target'].values
        x_train = train.drop('target', axis=1)

        grid = GridSearchCV(self.model, params, cv=nfolds, n_jobs=8, scoring='neg_mean_absolute_error', verbose=verbose)
        grid.fit(x_train, y_train)
        return grid.best_params_, grid.best_score_

    def evaluate(self, metric='gini'):
        _, test = self.split()
        y_val = test['target'].values
        x_val = test.drop('target', axis=1)
        prediction = self.predict(x_val)

        if metric == 'mae':
            return mean_absolute_error(y_val, prediction)

        elif metric == 'gini':

            def gini(actual, pred):
                assert len(actual) == len(pred)
                all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
                all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
                totalLosses = all[:, 0].sum()
                giniSum = all[:, 0].cumsum().sum() / totalLosses

                giniSum -= (len(actual) + 1) / 2.
                return giniSum / len(actual)

            def gini_normalized(a, p):
                return gini(a, p) / gini(a, a)

            return gini_normalized(y_val, prediction)

        raise NotImplementedError("Supported metrics: 'mae' and 'gini'")


class BasePredictor(Predictor):
    """
    A dummy predictor, always outputing the median. Used for benchmarking models.
    """
    def __init__(self, input, params={}, name=None):
        super().__init__(input, params, name='Naive')

    def train(self, params=None):
        """
        A dummy predictor does not require training. We only need the median
        """
        train, _ = self.split()
        y_train = train['target'].values
        self.params['median'] = np.median(y_train)

    def predict(self, x_val):
        return [self.params['median']] * len(x_val)


if __name__ == "__main__":

    print("Reading training data...")
    features = pd.read_csv('data/train.csv')

    print("\nSetting up data for Base Predictor ...")
    model = BasePredictor(features)

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining Base Predictor ...")
    model.train()

    print("\nEvaluating model...")
    gini = model.evaluate(metric='gini')

    print("\n##########")
    print("Gini score is: ", gini)
    print("##########")