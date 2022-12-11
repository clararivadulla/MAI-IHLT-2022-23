from code_.data_reader import data_reader
from code_.feature_extractor import Features
from code_.model import Models

#path = '/home/beny/Documents/MAI-projects/IHLT-project/'

x_train, y_train = data_reader(True, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt'])
x_test, y_test = data_reader(False, ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt', 'surprise.OnWN.txt', 'surprise.SMTnews.txt'])
train_features = Features(x_train).extract_all()
test_features = Features(x_test).extract_all()

models = Models(train_features, test_features, y_train, y_test)
models.RF()
models.SVR()
models.MLP()

