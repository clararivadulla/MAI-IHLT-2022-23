from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr

def pearson_score(y_true, y_predicted):
    return pearsonr(y_true, y_predicted)[0]

pearson = make_scorer(pearson_score, greater_is_better = True)

class Models:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.seed = 1984
    
    def RF(self):
        parameters = {
            'n_estimators': [50, 75, 100, 150], 
            'max_features': [None], 
            'bootstrap': [True], 
            'oob_score': [True], 
            'criterion': ['squared_error', 'absolute_error'], 
            'max_depth': [3, 5, 7, None], 
            'min_samples_split': [4, 6, 8], 
            'min_samples_leaf': [1, 2, 4, 6], 
            'random_state': [self.seed]
        }

        print(self.x_train)
        clf = GridSearchCV(RandomForestRegressor(), parameters, cv = 5, n_jobs = -1, scoring = pearson, verbose = 10)
        clf.fit(self.x_train, self.y_train)
        y_predict = clf.predict(self.x_test)
        print(pearsonr(self.y_test, y_predict)[0])
        print(clf.best_params_)

    def SVR(self):
        parameters = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf'], 
            'degree': [3, 4], 
            'gamma': ['auto', 'scale'], 
        }

        print(self.x_train)
        clf = GridSearchCV(SVR(), parameters, cv = 5, n_jobs = -1, scoring = pearson, verbose = 10)
        clf.fit(self.x_train, self.y_train)
        y_predict = clf.predict(self.x_test)
        print(pearsonr(self.y_test, y_predict)[0])
        print(clf.best_params_)



