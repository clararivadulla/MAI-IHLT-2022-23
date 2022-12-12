from code_.data_reader import data_reader
from code_.feature_extractor import Features
from code_.model import Models
import code_.utils as utils

x_train, y_train = data_reader(True, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt'])
x_test, y_test = data_reader(False, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt', 'surprise.OnWN.txt', 'surprise.SMTnews.txt'])
train_features = Features(x_train).extract_all()
test_features = Features(x_test).extract_all()

models = Models(train_features, test_features, y_train, y_test, x_train['Origin'], x_test['Origin'])
#utils.covariance_matrix_plot(train_features, y_train)
#utils.PCA(train_features, y_train)

models.build_unique('SVR')
models.evaluate_train()
models.evaluate_test()


models.build_tuned('RandomForestRegressor', 'RandomForestRegressor')
models.evaluate_train()

models.build_ensemble('SVR')
models.evaluate_train()
models.evaluate_test(True)

models.evaluate_ensemble()