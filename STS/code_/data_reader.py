import pandas as pd
import csv

def read_input_file(filename):
    """File to read the sentences file from the .txt files given. This has to be done 
    since some of the sentences have a " symbol that does not close, and breaks the pandas reader.

    Args:
        filename (str): path to the input file

    Returns:
        pandas.Dataframe: returns a pandas.Dataframe where each sentence pair belongs to a row, 
        with column 0 being the first pair of the two.
    """
    with open(filename, 'r') as f:
        sentences = f.read().splitlines()
        sentences = [s.split('\t') for s in sentences]
    return pd.DataFrame(sentences, columns = ['Sentence 1', 'Sentence 2']) 

def data_reader(train, input_files, path = ''):
    """Function that reads the data given a path and a list of input files. Additionally, it will
     remove the possible duplicates that exist in the data.

    Args:
        train (bool): is the dataset we are reading the train or test. If it is train, we remove the duplicates. 
        input_files (list): a list containing the different names of the datasets.
        path (str): the path to the where the STS is contained, either

    Returns:
        (data, target): it returns two pandas Dataframes, the first, `data` contains the sentence pairs and 
        the origin of the data, i.e., what dataset the sentence pair belongs to. The second `target`, contains the
        gold standard, matching the rows.
    """
    if train:
        path += 'train'
        print('Reading the training data')
    else:
        path += 'test-gold'
        print('Reading the test data')

    data = pd.DataFrame(columns = ['Sentence 1', 'Sentence 2', 'GS', 'Origin']) # empty dataframe to allocate data
    
    for input in input_files:
        input_data = read_input_file(path + '/STS.input.' + input)              # read the sentence pairs
        input_gold = pd.read_table(path + '/STS.gs.' + input, names=['GS'])     # read the gold standard
        input_df = pd.concat([input_data, input_gold], axis=1)                  # add the gold standard to the sentence pairs
        input_df['Origin'] = input[:-4]                                         # add the origin of the sentences pairs, which dataset comes from
        data = pd.concat([data, input_df], ignore_index=True)                   # concatenate datasets into a big one
    data.reset_index(inplace=True)
    if train:
        data.drop_duplicates(subset = ['Sentence 1', 'Sentence 2', 'GS'], keep = False, inplace = True) # 12 duplicates in training set

    # get the target feature and remove it from the testing data
    target = data['GS'] 
    data.drop(columns=['index', 'GS'], inplace=True)
    
    print(f'Data contains {data.shape[0]} sentences')
    return data, target

