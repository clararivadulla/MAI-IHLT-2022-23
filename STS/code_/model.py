from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import numpy as np
import code_.utils as utils


# We create a custom grid search scorer, which uses the Pearson Correlation to predict choose which hyperparameters are better.
pearson = make_scorer(lambda y_true, y_predicted: pearsonr(y_true, y_predicted)[0], greater_is_better = True)

class Models:
    def __init__(self, x_train, x_test, y_train, y_test, train_origin, test_origin, seed = 1984, scaler = 'StandardScaler', verbose = 0):
        scaler = getattr(sklearn.preprocessing, scaler)() # creates an instance of a scaler object
        self.x_train = scaler.fit_transform(x_train)      # fits and transforms the training features
        self.x_test = scaler.transform(x_test)            # with the fitted scaler, we transform the test features
        self.train_origin = train_origin
        self.test_origin = test_origin
        self.verbose = verbose
        self.y_train = y_train
        self.y_test = y_test
        self.seed = seed

        self.grid_parameters = {
                'RandomForestRegressor': {
                    'n_estimators': [300, 500], 
                    'max_features': ['sqrt', 'log2'], 
                    'bootstrap': [True], 
                    'oob_score': [True], 
                    'criterion': ['squared_error', 'absolute_error'], 
                    'max_depth': [None], 
                    'min_samples_split': [2, 4], 
                    'min_samples_leaf': [2], 
                    'random_state': [self.seed]
                },
                'SVR': {
                    'C': [0.01, 0.1, 1, 100],
                    'kernel': ['rbf'], 
                    'gamma': ['auto', 'scale', 0.005, 0.05, 0.5, 2, 5]
                }
            }

    def build_unique(self, model, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters

        clf = GridSearchCV(globals()[model](), parameters[model], cv = 5, n_jobs = -1, scoring = pearson, verbose = self.verbose)
        clf.fit(self.x_train, self.y_train)
        self.regressors = {}
        self.regressors['global'] = clf

    def build_tuned(self, model_tuned, model_global, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters
        
        datasets = np.unique(self.train_origin)
        regressors = {}
        # For each of the different train sets, we train a model specific to it
        for dataset in datasets:
            clf = GridSearchCV(globals()[model_tuned](), parameters[model_tuned], cv = 5, n_jobs = -1, scoring = pearson, verbose = self.verbose)
            clf.fit(self.x_train[dataset == self.train_origin], self.y_train[dataset == self.train_origin])
            regressors[dataset] = clf

        clf = GridSearchCV(globals()[model_global](), parameters[model_global], cv = 5, n_jobs = -1, scoring = pearson, verbose = self.verbose)
        clf.fit(self.x_train, self.y_train)
        regressors['global'] = clf

        self.regressors = regressors       

    def build_ensemble(self, model, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters

        predictions = {}
        for dataset in np.unique(self.train_origin):
            regressor = self.regressors.get(dataset) 
            predictions[dataset] = regressor.predict(self.x_train)

        clf = GridSearchCV(globals()[model](), parameters[model], cv = 5, n_jobs = -1, scoring = pearson, verbose = self.verbose)
        clf.fit(pd.DataFrame(predictions), self.y_train)
        self.regressors['combine'] = clf

    def evaluate_train(self):
        results_dataframe = {'Dataset': [], 'Pearson CV Train': []}
        cv_score_train = 0
        # predict to get the mean Pearson score from the CV
        for dataset in np.unique(self.train_origin):
            regressor = self.regressors.get(dataset, self.regressors['global'])         # get the regressor from the specific train dataset
            results_dataframe['Pearson CV Train'].append(regressor.best_score_)         # get the CV score from the specific train dataset
            cv_score_train += regressor.best_score_*sum(dataset == self.train_origin)    # we get the weighted mean of the individual scores
            results_dataframe['Dataset'].append(dataset)
        
        # Create a dataframe and print the results
        specific = pd.DataFrame(results_dataframe)
        overall = pd.DataFrame({'Dataset': ['Overall'], 'Average Pearson CV Train': [cv_score_train/len(self.y_train)]})
        utils.print_evaluation(specific, overall)


    def evaluate_test(self, is_ensemble = False):
        results_dataframe = {'Dataset': [], 'Pearson Test': []}

        if is_ensemble:
            trained_datasets = np.unique(self.train_origin)
            predictions = {}                                            # To predict the new global model
            for dataset in trained_datasets:
                regressor = self.regressors.get(dataset)                # get the regressor from the specific train dataset
                predictions[dataset] = regressor.predict(self.x_test)   # predict ALL the test data with each model

            # we combine the overall predictions of each test into one regressor, that will be used by the combiner model to predict the surprise datasets
            predictions_combine = self.regressors['combine'].predict(pd.DataFrame(predictions))

        predictions = []
        results_dataframe['Pearson Test'] = []
        for dataset in np.unique(self.test_origin):
            regressor = self.regressors.get(dataset, self.regressors['global'])         # get the regressor from the train data, or else the global regressor

            y_predict = regressor.predict(self.x_test[dataset == self.test_origin])     # predict the specific test dataset
            if is_ensemble and dataset not in trained_datasets:
                y_predict = predictions_combine[dataset == self.test_origin]            # get the already evaluated predictions for the surprise, overwrite the previous
            y_predict = np.clip(y_predict, 0, 5)                                        # clamp it to [0, 5]
            predictions.extend(y_predict)                                               # concatenate predictions to evaluate global Pearson

            # Append the "local" results
            results_dataframe['Dataset'].append(dataset)
            results_dataframe['Pearson Test'].append(pearsonr(self.y_test[dataset == self.test_origin], y_predict)[0]) # evaluate specific test dataset Pearson

        # Create a dataframe and print the results
        specific = pd.DataFrame(results_dataframe)
        overall = pd.DataFrame({'Dataset': ['Overall'], 'Pearson Test': [pearsonr(self.y_test, predictions)[0]]})
        utils.print_evaluation(specific, overall, is_train = False)