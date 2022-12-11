from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr

# We create a custom grid search scorer, which uses the Pearson Correlation to predict choose which hyperparameters are better.
pearson = make_scorer(lambda y_true, y_predicted: pearsonr(y_true, y_predicted)[0], greater_is_better = True)

class Models:
    def __init__(self, x_train, x_test, y_train, y_test, seed = 1984, scaler = 'StandardScaler'):
        scaler = getattr(sklearn.preprocessing, scaler)() # creates an instance of a scaler object
        self.x_train = scaler.fit_transform(x_train)      # fits and transforms the training features
        self.x_test = scaler.transform(x_test)            # with the fitted scaler, we transform the test features
        self.y_train = y_train
        self.y_test = y_test
        self.seed = seed
    
    def RF(self):
        parameters = {
            'n_estimators': [300, 400, 500], 
            'max_features': [None], 
            'bootstrap': [True], 
            'oob_score': [True], 
            'criterion': ['squared_error'], 
            'max_depth': [None], 
            'min_samples_split': [2, 4], 
            'min_samples_leaf': [2], 
            'random_state': [self.seed]
        }

        clf = GridSearchCV(RandomForestRegressor(), parameters, cv = 5, n_jobs = -1, scoring = pearson, verbose = 10)
        clf.fit(self.x_train, self.y_train)
        y_predict = clf.predict(self.x_test)
        print('Finished training')
        print('Best parameters', clf.best_params_)
        print('Pearson with the test data', pearsonr(self.y_test, y_predict)[0])

    def SVR(self):
        parameters = {
            'C': [0.1, 1],
            'kernel': ['rbf'], 
            #'degree': [3, 4], 
            'gamma': ['auto', 'scale']
        }

        clf = GridSearchCV(SVR(), parameters, cv = 5, n_jobs = -1, scoring = pearson, verbose = 10)
        clf.fit(self.x_train, self.y_train)
        y_predict = clf.predict(self.x_test)
        print('Finished training')
        print('Best parameters', clf.best_params_)
        print('Pearson with the test data', pearsonr(self.y_test, y_predict)[0])

    def MLP(self):
        parameters = {
            'hidden_layer_sizes': [(100,), (50,)],
            'activation': ['logistic', 'tanh', 'relu'], 
            'max_iter': [1000],
        }

        clf = GridSearchCV(MLPRegressor(), parameters, cv = 5, n_jobs = -1, scoring = pearson, verbose = 10)
        clf.fit(self.x_train, self.y_train)
        y_predict = clf.predict(self.x_test)
        print('Finished training')
        print('Best parameters', clf.best_params_)
        print('Pearson with the test data', pearsonr(self.y_test, y_predict)[0])


