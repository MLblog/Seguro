from abc import abstractmethod, ABCMeta
import numpy as np
from time import gmtime, strftime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from FeatureEngineering import *
from sklearn.model_selection import GridSearchCV

TUNING_OUTPUT_DEFAULT = 'tuning.txt'

"""
An abstract class modeling our notion of a predictor.
Concrete implementations should follow the predictors
interface
"""
class Predictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, train, test, params={}, name=None):
        """
        Base constructor

        :param train: a pd.DataFrame of IDs, raw features and the target variable
        :param params: a dictionary of named model parameters
        :param name: Optional model name, used for logging
        """
        self.train = train
        self.test = test
        self.params = params
        self.name = name

    def set_params(self, params):
        """Override parameters set in the constructor. Dictionary expected"""
        self.params = params

    def split(self, val_size=0.3):
        """
        Splits the merged input (features + labels) into a training and a validation set
        :return: a tuple of training and validation sets with the given size ratio
        """
        train, val = train_test_split(self.train, test_size=val_size, random_state=42)
        return train, val

    @abstractmethod
    def fit(self):
        """
        A function that fits the predictor to the given dataset.
        """

    @abstractmethod
    def predict(self, x_test):
        """
        Predicts the label for the given input
        :param x_test: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    def preprocess(self, threshold=50):
        def concat():
            self.train['is_train'] = True
            self.test['is_train'] = False
            self.test['target'] = None
            assert set(self.train) == set(self.test), 'train and test sets have different features'
            return pd.concat([self.train, self.test])

        def cleanup(full):
            self.train = full[full['is_train']]
            self.test = full[~full['is_train']]
            self.train = self.train.drop('is_train', axis=1)
            self.test = self.test.drop(['is_train', 'target'], axis=1)
            assert len(full) == (len(self.train) + len(self.test))
            del full

        full = concat()

        ######################################
        # ACTUAL PREPROCESSING CALLS GO HERE #
        ######################################
        full = dummy_conversion(full, threshold)

        cleanup(full)

    def create_submission(self, params):
        def submit():
            def truncate(p):
                if p < 0:
                    return 0
                elif p > 1:
                    return 1
                return p

            self.fit(params)
            ids = self.test['id']
            predictions = self.predict(self.test)
            truncated_predictions = list(map(truncate, predictions))
            submission = pd.DataFrame(
                {'id': ids,
                 'target': truncated_predictions
                 })
            timestamp = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            submission.to_csv('submissions/submission_{}_{}'.format(timestamp, self.name), index=False)

        print("\nCreating submission file...")
        submit()

    @timing
    def tune(self, params, nfolds=3, verbose=3, persist=True, write_to=TUNING_OUTPUT_DEFAULT):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation.
        :param verbose: Verbosity level. 0 is silent, higher int prints more stuff
        :param write_to: If persist is set to True, write_to defines the filepath to write to
        :return: Dict of best parameters found.
        """
        def persist_tuning(score, params, write_to=write_to):
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
                f.write("Best GINI\t{}\nparams: {}\n\n".format(score, params))

        self.preprocess()
        train, _ = self.split()
        y_train = train['target'].values
        x_train = train.drop('target', axis=1)

        # Create custom scorer to use GINI as optimization metric
        scoring = make_scorer(self._gini_normalized, greater_is_better=True)
        grid = GridSearchCV(self.model, params, cv=nfolds, n_jobs=8, scoring=scoring, verbose=verbose)
        grid.fit(x_train, y_train)

        if persist:
            persist_tuning(grid.best_score_, grid.best_params_)
        return grid.best_params_, grid.best_score_

    def _gini_normalized(self, a, p):

        def gini(actual, pred):
            assert len(actual) == len(pred)
            all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
            all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
            total_losses = all[:, 0].sum()
            gini_sum = all[:, 0].cumsum().sum() / total_losses

            gini_sum -= (len(actual) + 1) / 2.
            return gini_sum / len(actual)

        return gini(a, p) / gini(a, a)

    def evaluate(self, metric='gini'):
        _, val = self.split()
        y_val = val['target'].values
        x_val = val.drop('target', axis=1)
        prediction = self.predict(x_val)

        print("Predictions: {} \n".format(prediction))

        if metric == 'mae':
            return mean_absolute_error(y_val, prediction)

        elif metric == 'gini':
            return self._gini_normalized(y_val, prediction)

        raise NotImplementedError("Supported metrics: 'mae' and 'gini'")


class BasePredictor(Predictor):
    """
    A dummy predictor, always outputing the median. Used for benchmarking models.
    """
    def __init__(self, input, params={}, name=None):
        super().__init__(input, params, name='Naive')

    def fit(self, params=None):
        """
        A dummy predictor does not require training. We only need the median
        """
        train, _ = self.split()
        y_train = train['target'].values
        self.params['mean'] = np.mean(y_train)

    def predict(self, x_val):
        #return [self.params['mean']] * len(x_val)
        predictions = [0] * len(x_val)
        predictions[5] = 1


if __name__ == "__main__":

    print("Reading training data...")
    features = pd.read_csv('data/train.csv')

    print("\nSetting up data for Base Predictor ...")
    model = BasePredictor(features)

    # Fit the model using the best set of parameters found by the gridsearch
    print("\nTraining Base Predictor ...")
    model.fit()

    print("\nEvaluating model...")
    gini = model.evaluate(metric='gini')

    print("\n##########")
    print("Gini score is: ", gini)
    print("##########")