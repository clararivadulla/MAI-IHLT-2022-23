from code_.data_reader import data_reader
from code_.feature_extractor import Features
from code_.model import Models
import code_.utils as utils
import pickle

# Read the data
x_train, y_train = data_reader(True, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt'])
x_test, y_test = data_reader(False, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt', 'surprise.OnWN.txt', 'surprise.SMTnews.txt'])

# Extract the features
train_features = Features(x_train).extract_all()
test_features = Features(x_test).extract_all()

#with open('data.pickle', 'rb') as f:
#    train_features = pickle.load(f)
#    test_features = pickle.load(f)

with open("data.pickle", "wb") as f:
    pickle.dump(train_features, f)
    pickle.dump(test_features, f)

# Build an object, will split the train into train and validation sets. Will build the models and do Feature Selection
models = Models(train_features, test_features, y_train, y_test, x_train['Origin'], x_test['Origin'])

# Data Visualization
#utils.covariance_matrix_plot(train_features, y_train)
#utils.PCA(train_features, y_train)

# STEP 1: MODEL SELECTION


# we build a global with SVM
#models.build_global('SVR')
#models.evaluate_train()
#models.evaluate_test(validation = True)

# we test with Random Forest
#models.build_unique('RandomForestRegressor')
#models.evaluate_train()
#models.evaluate_test(validation = True)


# After seeing that the validation partition is better on Random Forest, we will use it instead of SVM
# Feature Selection

# we evaluate for each specific dataset and a global RandomForest
#models.build_global('RandomForestRegressor')
#models.build_specific('RandomForestRegressor')
#models.evaluate_train()
#models.evaluate_test(validation = True)

# we evaluate the ensemble method
models.build_specific('RandomForestRegressor')
models.build_global('RandomForestRegressor')
models.build_ensemble('SVR')
models.evaluate_train()
models.evaluate_test(validation = True, is_ensemble = True)
models.feature_importance()
models.feature_selection()

# Building Final Estimator
# We rebuild the predictor with the entire train dataset, without the validation split
# we evaluate the result on the test datasets
models.is_final()
models.build_global('RandomForestRegressor')
models.build_specific('RandomForestRegressor')
models.build_ensemble('SVR')
models.evaluate_train()
models.evaluate_test(validation = False, is_ensemble = True)

