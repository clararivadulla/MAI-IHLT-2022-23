from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import numpy as np
import code_.utils as utils


# We create a custom grid search scorer, which uses the Pearson Correlation to predict choose which hyperparameters are better.
pearson = make_scorer(lambda y_true, y_predicted: pearsonr(y_true, y_predicted)[0], greater_is_better = True)

class Models:
    def __init__(self, x_train, x_test, y_train, y_test, train_origin, test_origin, sentences, seed = 1984, scaler = 'StandardScaler', verbose = 0, cv = 5):
        scaler = getattr(sklearn.preprocessing, scaler)()   # creates an instance of a scaler object to normalize the features
        self.feature_names = x_train.columns
        self.sentences = sentences # original sentences form Train to see where it fails

        # we do a 3 way split of the data.
        #  - Train is used to find the best hyperparameters using CV
        #  - Validation is done to find the best model
        #  - Test is already given to us, and we will use it to give a generalization estimate of the Pearson measure
        # Train Full will be used once the best hyperparameters are found to retrain on all the data to estimate the Test data

        ################################################################################
        # This is a very important variable, it will mark if we are training with the full set 
        # or a partition as before. If set to True, it will set self.x_train and self.y_train to 
        # be the full partition. Once that is done, it will retrain the parameters. 
        self.final = False 
        # set that it is final by calling is_final() on the object instance
        ################################################################################

        self.x_train_full = scaler.fit_transform(x_train)   # fits and transforms the training features
        self.y_train_full = y_train
        self.train_origin_full = train_origin

        # we split the train data into train and validation for model selection
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train_full, y_train, test_size = 0.2, stratify = train_origin, random_state = seed)
        _, _, self.train_origin, self.test_origin = train_test_split(self.x_train_full, train_origin, test_size = 0.2, stratify = train_origin, random_state = seed)

        # test data for the final evaluation, these are stored until is_real = True
        self.x_test_real = scaler.transform(x_test)              # with the fitted scaler, we transform the test features
        self.y_test_real = y_test
        self.test_origin_real = test_origin
        
        # set parameters for the rest of models
        self.regressors = {}
        self.verbose = verbose
        self.seed = seed
        self.cv = cv

        # grid search 
        self.grid_parameters = {
                'RandomForestRegressor': {
                    'n_estimators': [100, 250, 500],  
                    'max_features': [None], 
                    'bootstrap': [True], 
                    'oob_score': [True], 
                    'criterion': ['squared_error'], 
                    'max_depth': [4], 
                    'min_samples_split': [2], 
                    'min_samples_leaf': [2], 
                    'random_state': [self.seed]
                },
                'SVR': {
                    'C': [0.01, 0.1, 1, 100],
                    'kernel': ['rbf'], 
                    'gamma': ['auto', 'scale', 0.005, 0.05, 0.5, 2, 5]
                }
            }

        # larger grid search for the final parameter tuning
        self.grid_parameters_big = {
                'RandomForestRegressor': {
                    'n_estimators': [100, 200, 300, 400, 500], 
                    'max_features': [None, 'log2'],
                    'bootstrap': [True], 
                    'oob_score': [True], 
                    'criterion': ['squared_error', 'absolute_error'],  
                    'max_depth': [None, 4], 
                    'min_samples_split': [2], 
                    'min_samples_leaf': [2], 
                    'random_state': [self.seed]
                },
                'SVR': {
                    'C': [0.01, 0.1, 1, 100],
                    'kernel': ['rbf'], 
                    'gamma': ['auto', 'scale', 0.005, 0.05, 0.5, 2, 5]
                }
            }

        self.features_selected = {}
        self.use_features_selection = False

    def is_final(self):
        self.final = True 
        self.x_train = self.x_train_full  
        self.y_train = self.y_train_full  
        self.train_origin = self.train_origin_full

        self.x_test = self.x_test_real    
        self.y_test = self.y_test_real
        self.test_origin = self.test_origin_real

        self.grid_parameters = self.grid_parameters_big

        self.regressors = {}


    def feature_selection_use(self, use_feature_selection = True):
        self.use_features_selection = use_feature_selection and len(self.features_selected) != 0

    def build_global(self, model, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters

        clf = GridSearchCV(globals()[model](), parameters[model], cv = self.cv, n_jobs = -1, scoring = pearson, verbose = self.verbose)

        if self.use_features_selection:
            clf.fit(self.x_train[:, self.features_selected['global']], self.y_train)
        else:
            clf.fit(self.x_train, self.y_train)
        self.regressors['global'] = clf

    def build_specific(self, model, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters
        
        datasets = np.unique(self.train_origin)
        regressors = {}
        # For each of the different train sets, we train a model specific to it
        for dataset in datasets:
            clf = GridSearchCV(globals()[model](), parameters[model], cv = self.cv, n_jobs = -1, scoring = pearson, verbose = self.verbose)
            if self.use_features_selection:
                clf.fit(self.x_train[dataset == self.train_origin, :][:, self.features_selected[dataset]], self.y_train[dataset == self.train_origin])
            else:
                clf.fit(self.x_train[dataset == self.train_origin], self.y_train[dataset == self.train_origin])
            self.regressors[dataset] = clf

    def predict_for_ensemble(self, train):
        # To predict the new ensemble model
        predictions = {}
        if train:
            x = self.x_train
        else:
            x = self.x_test

        for dataset in np.unique(self.train_origin): 
            regressor = self.regressors.get(dataset)                # get the regressor from the specific train dataset
            if self.use_features_selection:                         # predict ALL the test data with each model
                predictions[dataset] = regressor.predict(x[:, self.features_selected[dataset]])
            else:
                predictions[dataset] = regressor.predict(x)
        
        # we combine the overall predictions of each test into one regressor, that will be used by the combiner model to predict the surprise datasets
        if self.use_features_selection:
            predictions['global'] = self.regressors['global'].predict(x[:, self.features_selected['global']])
        else:
            predictions['global'] = self.regressors['global'].predict(x)

        return pd.DataFrame(predictions)

    def build_ensemble(self, model, parameters = None):
        if parameters is None:
            parameters = self.grid_parameters

        clf = GridSearchCV(globals()[model](), parameters[model], cv = self.cv, n_jobs = -1, scoring = pearson, verbose = self.verbose)
        clf.fit(self.predict_for_ensemble(train = True), self.y_train)
        self.regressors['combine'] = clf

    def evaluate_train(self):
        results_dataframe = {'Dataset': [], 'Pearson CV Train': []}
        cv_score_train = 0
        # predict to get the mean Pearson score from the CV
        for dataset in np.unique(self.train_origin):
            regressor = self.regressors.get(dataset, self.regressors['global'])         # get the regressor from the specific train dataset
            results_dataframe['Pearson CV Train'].append(regressor.best_score_)         # get the CV score from the specific train dataset
            cv_score_train += regressor.best_score_*sum(dataset == self.train_origin)   # we get the weighted mean of the individual scores
            results_dataframe['Dataset'].append(dataset)
        
        # Create a dataframe and print the results
        specific = pd.DataFrame(results_dataframe)
        overall = pd.DataFrame({'Dataset': ['Overall'], 'Average Pearson CV Train': [cv_score_train/len(self.y_train)]})
        utils.print_evaluation(specific, overall)


    def evaluate_test(self, validation = False, is_ensemble = False, print_failures = False, plot_interpretations = False):
        name_col = 'Pearson Validation' if validation else 'Pearson Test'
        results_dataframe = {'Dataset': [], name_col: []}

        if is_ensemble:
            trained_datasets = np.unique(self.train_origin)
            predictions_combine = self.regressors['combine'].predict(self.predict_for_ensemble(train = False))

        predictions = []
        results_dataframe[name_col] = []
        cv_score = 0
        for dataset in np.unique(self.test_origin):
            regressor = self.regressors.get(dataset, self.regressors['global'])         # get the regressor from the train data, or else the global regressor

            if is_ensemble and dataset not in trained_datasets:
                y_predict = predictions_combine[dataset == self.test_origin]            # get the already evaluated predictions for the surprise, overwrite the previous
            elif self.use_features_selection:
                y_predict = regressor.predict(self.x_test[dataset == self.test_origin, :][:, self.features_selected[dataset]])     # predict the specific test dataset
            else:
                y_predict = regressor.predict(self.x_test[dataset == self.test_origin])     # predict the specific test dataset

            y_predict = np.clip(y_predict, 0, 5)                                        # clamp it to [0, 5]
            predictions.extend(y_predict)                                               # concatenate predictions to evaluate global Pearson

            pearson = pearsonr(self.y_test[dataset == self.test_origin], y_predict)[0]
            cv_score += pearson*sum(dataset == self.test_origin)
            # Append the "local" results
            results_dataframe['Dataset'].append(dataset)
            results_dataframe[name_col].append(pearson) # evaluate specific test dataset Pearson

        # Create a dataframe and print the results
        specific = pd.DataFrame(results_dataframe)
        if validation:
            overall = pd.DataFrame({'Dataset': ['Overall'], name_col: [cv_score/len(self.y_test)]})
        else:
            overall = pd.DataFrame({'Dataset': ['Overall'], name_col: [pearsonr(self.y_test, predictions)[0]]})
        utils.print_evaluation(specific, overall, validation = validation, is_train = False)

        if not self.final and print_failures:
            # print where the sentences are failing and working well
            utils.sentence_analysis(predictions, self.y_test, self.sentences, self.regressors, self.x_test, self.feature_names, self.features_selected, plot_interpretations)
            


    def feature_importance(self):
        # Feature importance from RandomForest
        importances_impurity = {}
        importances_permutat = {}
        std_impurity = {}
        std_permutat = {}
        for model, regressor in self.regressors.items():
            model = model.title()
            regressor = regressor.best_estimator_
            if 'Forest' not in regressor.__class__.__name__:
                continue
            importances_impurity[model] = regressor.feature_importances_
            std_impurity[model] = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis = 0)
            
            # We are doing this on the validation dataset, not the actual test set
            result = permutation_importance(regressor, self.x_test, self.y_test, n_repeats = 5, random_state = self.seed, n_jobs = -1, scoring = pearson)
            importances_permutat[model] = result.importances_mean
            std_permutat[model] = result.importances_std

        utils.plot_importances(importances_impurity, std_impurity, self.feature_names, 'impurity', print_error = False)
        utils.plot_importances(importances_permutat, std_permutat, self.feature_names, 'permutation')

        # Permutation importance on SVM
        predictions = {}                                            # To predict the new ensemble model
        for dataset in np.unique(self.train_origin):
            regressor = self.regressors.get(dataset)                # get the regressor from the specific train dataset
            predictions[dataset] = regressor.predict(self.x_test)   # predict ALL the test data with each model

        predictions['global'] = self.regressors['global'].predict(self.x_test)
        # we combine the overall predictions of each test into one regressor, that will be used by the combiner model to predict the surprise datasets

        result = permutation_importance(self.regressors['combine'], pd.DataFrame(predictions), self.y_test, n_repeats = 5, random_state = self.seed, n_jobs = -1, scoring = pearson)
        importances = result.importances_mean
        std = result.importances_std
        
        utils.plot_importances(importances, std, self.regressors['combine'].feature_names_in_, 'SVR')


    def feature_selection(self):
        subset_features = set()
        scores = {}
        features = {}
       
        for model, regressor in self.regressors.items():
            if model == 'combine':
                continue
            # Recursive Cross-Validated Feature selection, done on each model, except the SVM layer
            rfecv = RFECV(estimator = regressor.best_estimator_, step = 1, cv = self.cv, scoring = pearson, min_features_to_select = 1, n_jobs = -1)
            rfecv.fit(self.x_test, self.y_test) # THIS IS DONE BEFORE is_final! THIS IS USING VALIDATION
            
            self.features_selected[model] = rfecv.support_
            features[model] = self.feature_names[rfecv.support_]

            scores[model] = (rfecv.cv_results_["mean_test_score"], rfecv.cv_results_["std_test_score"])
            subset_features = subset_features.union(self.feature_names[rfecv.support_])
        
        self.use_features_selection = True
        utils.plot_feature_selection(scores)
        print('Original number of features:', len(self.feature_names))
        print('Reduced number of features:', len(subset_features))

        print()
        print('We keep for each model the following:')
        for model, feats in features.items():
            print(model.title(), 'using now:', len(feats), 'features')
            #print(feats)
            #print()


            
