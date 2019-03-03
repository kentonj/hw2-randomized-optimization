import pandas as pd 
import datetime
import pickle
import os
import glob
import numpy as np
import copy

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, auc, accuracy_score, roc_curve, balanced_accuracy_score

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle, resample


class Dataset(object):
    '''
    this class is used to more easily structure classification datasets between the data (matrix of values) and the target (class)
    instead of doing any indexing when training an algorithm, simply use Dataset.data and Dataset.target
    '''
    def __init__(self, df, target_col_name):
        self.target_col_name = target_col_name
        self.target = df.loc[:, target_col_name].values.astype('int32')
        self.data = df.loc[:, df.columns != target_col_name].values.astype('float32')
        self.df = df
    def __str__(self):
        return ('first 10 data points\n' + str(self.data.head(10)) + '\nfirst 10 labels\n' + str(self.target.head(10)) + '\n')

class MachineLearningModel(object):
    '''
    this is to help separate models by attributes, i.e. IDs to keep track of many models being trained in one batch
    '''
    def __init__(self, model, model_family, model_type, framework, nn=False, id=None):
        self.model = model
        self.framework = framework
        self.model_family = model_family
        self.model_type = model_type
        self.nn = nn
        self.train_sizes = None
        self.iter_scores = None
        self.train_scores = None
        self.val_scores = None
        self.cm = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.roc_auc = None
        self.accuracy = None
        self.balanced_accuracy = None

        if id:
            self.id = id #set as id if provided
        else:
            self.id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) #otherwise set id to time right now
    def set_training_time(self, training_time):
        self.training_time = round(training_time,4) #training time rounded to 4 decimals
    def set_evaluation_time(self, evaluation_time):
        self.evaluation_time = round(evaluation_time,4) #training time rounded to 4 decimals
    def get_train_sizes(self):
        return self.train_sizes
    def get_train_scores(self):
        try:
            return np.mean(self.train_scores, axis=1)
        except:
            return self.train_scores
    def get_validation_scores(self):
        try:
            return np.mean(self.val_scores, axis=1)
        except:
            return self.val_scores
    def get_iter_scores(self):
        return self.iter_scores
    def get_cm(self):
        return self.cm
    def get_normalized_cm(self):
        return self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
    def __str__(self):
        return 'MODEL DETAILS: ' + self.model_type + ' model from ' + self.framework + ' with ID: ' + str(self.id)


def clean_and_scale_dataset(dirty_df_dict, na_action='mean', scaler=None, class_col='class'):
    
    clean_df_list = []

    if scaler:
        scaler.fit(dirty_df_dict['train'].loc[:,dirty_df_dict['train'].columns!=class_col])
    for df_name, dirty_df in dirty_df_dict.items():
        if scaler:
            dirty_df.loc[:,dirty_df.columns!=class_col] = scaler.transform(dirty_df.loc[:,dirty_df.columns!=class_col])

        #how to handle na values in dataset:
        if na_action == 'drop':
            dirty_df = dirty_df.dropna()
        if na_action == 'mean':
            dirty_df = dirty_df.fillna(dirty_df.mean())
        if na_action == 'mode':
            dirty_df = dirty_df.fillna(dirty_df.mode())
        if na_action == 'zeros':
            dirty_df = dirty_df.fillna(0)
        else:
            try:
                dirty_df = dirty_df.fillna(int(na_action))
            except:
                dirty_df = dirty_df.fillna(0) 
                print('filled with zeros as a failover')

        cleaned_df = dirty_df
        clean_df_list.append(cleaned_df)
    
    return clean_df_list

def balance(df, class_col='class', balance_method='downsample'):

    if type(balance_method) == int:
        n_samples = balance_method
    elif type(balance_method) == str:
        if balance_method == 'downsample':
            n_samples = min(df[class_col].value_counts())
        elif balance_method == 'upsample':
            n_samples = max(df[class_col].value_counts())
        else:
            raise ValueError('no viable sampling method provided, please enter (upsample, downsample, or an integer)')

    df_list = []
    for label in np.unique(df[class_col]):
        subset_df = df[df[class_col]==label]
        resampled_subset_df = resample(subset_df, 
                                        replace=(subset_df.shape[0]<n_samples),    # sample with replacement if less than number of samples, otherwise without replacement
                                        n_samples=n_samples)    # to match minority class
        df_list.append(resampled_subset_df)
    balanced_df = pd.concat(df_list)
    
    return balanced_df


def prep_data(df_dict, shuffle_data=True, balance_method='downsample', class_col='class'):
    '''
    always pass training set as first df in list
    '''
    #encode dataset to binary variables
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df_dict['train'][class_col])

    prepped_df_list = []
    
    for df_key, df in df_dict.items():

        #encode training dataset
        df[class_col] = encoder.transform(df[class_col])

        if balance_method:
            if df_key=='train': #only balance training data
                df = balance(df, class_col=class_col, balance_method=balance_method)

        dataset_df = Dataset(df, class_col)
        if shuffle_data:
            dataset_df.data, dataset_df.target = shuffle(dataset_df.data, dataset_df.target)
        
        prepped_df_list.append(dataset_df)

    return prepped_df_list, encoder

def pickle_save_model(algo, model_folder='models'):
    '''
    save the model with datetime if with_datetime=True, which will save model_20190202122930
    '''
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    filename = model_folder+'/'+str(algo.model_type)+'.model'
    print(f'saving as file: {filename}')
    pickle.dump(algo, open(filename, 'wb'))
    return None

def pickle_load_model(model_path):
    '''
    if no model_full_path is provided, then assume that we are looking for the most recent model
    model type can also be specified to retrieve the most recent model of a current type
    '''
    try:
        model = pickle.load(open(model_path, 'rb'))
        print('model successfully loaded from: {}'.format(model_path))
        return model
    except:
        print('did not successfully load model from: {}'.format(model_path))
        FileNotFoundError('model file not found')

def generate_learning_curves(algo, training_data, num_chunks=3, num_k=3):
    learning_chunks = [int(x) for x in np.linspace(0, training_data.data.shape[0], num_chunks+1)]

    train_scores_list = [] #average scores
    val_scores_list = []
    train_sizes_list = []
    print('')
    for i in range(num_chunks):
        number_lc_points = learning_chunks[i+1]
        print('{} - working on learning curve: {} data points'.format(algo.model_type, number_lc_points))

        lc_allotted_data = training_data.data[:number_lc_points]
        
        lc_allotted_target = training_data.target[:number_lc_points]

        if num_k == 1:
            train_data, test_data, train_target, test_target = train_test_split(lc_allotted_data, lc_allotted_target, test_size=0.25)
            #fit the model
            algo.model.fit(train_data, train_target)
            
            train_predictions = algo.model.predict(train_data)
            train_score = roc_auc_score(train_target, train_predictions)

            val_predictions = algo.model.predict(test_data)
            val_score = roc_auc_score(test_target, val_predictions)
            
            train_scores_list.append(train_score)
            val_scores_list.append(val_score)
        elif num_k < 1:
            train_data, test_data, train_target, test_target = train_test_split(lc_allotted_data, lc_allotted_target, test_size=num_k)
            algo.model.fit(train_data, train_target)
            
            train_predictions = algo.model.predict(train_data)
            train_score = roc_auc_score(train_target, train_predictions)

            val_predictions = algo.model.predict(test_data)
            val_score = roc_auc_score(test_target, val_predictions)
            
            train_scores_list.append(train_score)
            val_scores_list.append(val_score)
        else:
            train_chunk_score_list = []
            val_chunk_score_list = []
            cv_data_chunks_nums = [int(x) for x in np.linspace(0, lc_allotted_data.shape[0], num_k+1)]
            # print('shape of lc_allotted_data', lc_allotted_data.shape)
            cv_data_chunk_list = [(lc_allotted_data[cv_data_chunks_nums[i]:cv_data_chunks_nums[i+1],:], lc_allotted_target[cv_data_chunks_nums[i]:cv_data_chunks_nums[i+1]]) for i in range(len(cv_data_chunks_nums)-1)]
            # print('cv_data_chunks_nums',cv_data_chunks_nums)
            
            for i in range(len(cv_data_chunk_list)):
                cv_set = cv_data_chunk_list[i]
                cv_data = cv_set[0]
                cv_target = cv_set[1]
                training_set = [x for j,x in enumerate(cv_data_chunk_list) if j!=i] #select all chunks that aren't the current cv chunk
                train_data = np.concatenate([x[0] for x in training_set], axis=0)
                train_target = np.concatenate([x[1] for x in training_set], axis=0)
                # print('shape of training data', train_data.shape)
                # print('shape of trianing target', train_target.shape)
                # print('shape of cv data', cv_data.shape)
                # print('shape of cv target', cv_target.shape)
                
                #fit the model
                algo.model.fit(train_data, train_target)
                
                train_predictions = algo.model.predict(train_data)
                train_score = roc_auc_score(train_target, train_predictions)

                val_predictions = algo.model.predict(cv_data)
                val_score = roc_auc_score(cv_target, val_predictions)
                
                train_chunk_score_list.append(train_score)
                val_chunk_score_list.append(val_score)

            train_scores_list.append(train_chunk_score_list)
            val_scores_list.append(val_chunk_score_list)

        train_sizes_list.append(train_target.shape[0]) #length of most recent training example

    train_score_array = np.array(train_scores_list)
    val_scores_array = np.array(val_scores_list)
    train_sizes = np.array(train_sizes_list)

    algo.train_sizes = train_sizes
    algo.train_scores = train_score_array
    algo.val_scores = val_scores_array
    print('')
    return train_sizes, train_score_array, val_scores_array
