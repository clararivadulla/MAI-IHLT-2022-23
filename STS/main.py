#from google.colab import drive
#drive.mount('/content/drive')
#path = '/content/drive/MyDrive/IHLT/sts/' # path where the data is contained
path = '/home/beny/Documents/MAI-projects/IHLT-project/'

from code_.data_reader import data_reader
from code_.feature_extractor import Features
from code_.model import Models

x_train, x_test, y_train, y_test = data_reader('')
train_features = Features(x_train).extract_all()
test_features = Features(x_test).extract_all()

models = Models(train_features, test_features, y_train, y_test)
models.RF()
