import pandas as pd
import csv

def read_input_file(filename):
    with open(filename, 'r') as f:
        sentences = f.read().splitlines()
        sentences = [s.split('\t') for s in sentences]
    return pd.DataFrame(sentences, columns = ['Sentence 1','Sentence 2']) 

def data_reader(path):
    # Read the train input files
    input_files = ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt'] 
    train = pd.DataFrame(columns=['Sentence 1', 'Sentence 2', 'GS', 'Origin'])

    for input in input_files:
        input_data = read_input_file(path + 'train/STS.input.' + input)
        input_gold = pd.read_table(path + 'train/STS.gs.' + input, names=['GS']) 
        input_df = pd.concat([input_data, input_gold], axis=1)
        input_df['Origin'] = input[:-4]
        train = pd.concat([train, input_df], ignore_index=True)
    train.reset_index(inplace=True)
    train.drop_duplicates(subset = ['Sentence 1', 'Sentence 2', 'GS'], keep = False, inplace = True) # 12 duplicates
    y_train = train['GS']
    train.drop(columns=['index', 'GS'], inplace=True)

    # Read the test input files
    input_files = ['MSRpar.txt', 'MSRvid.txt', 'SMTeuroparl.txt', 'surprise.OnWN.txt', 'surprise.SMTnews.txt'] 
    test = pd.DataFrame(columns=['Sentence 1', 'Sentence 2', 'GS'])

    for input in input_files:
        input_data = read_input_file(path + 'test-gold/STS.input.' + input) 
        input_gold = pd.read_table(path + 'test-gold/STS.gs.' + input, names=['GS']) 
        input_df = pd.concat([input_data, input_gold], axis=1)
        test = pd.concat([test, input_df], ignore_index=True)
    test.reset_index(inplace=True)
    y_test = test['GS']
    test.drop(columns=['index', 'GS'], inplace=True)

    return train, test, y_train, y_test

