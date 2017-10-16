# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:23:12 2017

@author: s157084
"""

import pandas as pd
import time


def timing(f):
    """
    Decorator to time a function call and print results.
    :param f: Callable to be timed
    :return: Void. Prints to std:out as a side effect
    """
    def wrap(*args):
        start = time.time()
        ret = f(*args)
        stop = time.time()
        print('{} function took {:.1f} s'.format(f.__name__, (stop - start)))
        return ret
    return wrap


def drop_columns(df, threshold):
    """
    Drop the columns with more than threshold % of missing values. Based on Manos algorithm.
    """
    nan_count = df.isnull().mean()
    nan_count = nan_count[nan_count <= threshold]
    df = df[nan_count.index.tolist()]
    return df


def identify_categories(df, match_word='cat'):
    """
    Identify the categorical functions
    """
    col_name=list(df)
    categories=[]
    for i in col_name:
        if match_word in i:
            categories.append(i)
    
    return categories

def dummy_conversion(df, threshold, categories=[]):
    """
    Transform the columns with strings to features only if the number of dummy variables 
    created are smaller than the the threshold
    """
    list_names = []
    for c in df.columns:
        if (df[c].dtype == 'object') | (c in categories):
            if c in categories:
                df[c] = df[c].astype('category')
                n = len(df[c].cat.categories)
            else:
                n = len(set(df[c]))
                
            if n <= threshold:
                list_names.append(c)
            else:
                print("Dropping variable " + str(c))
                del df[c]

    print('The features that will be transformed are:')
    print(list_names)
    df = pd.get_dummies(df, columns=list_names)
    return df


def create_submission(train, test, threshold, col_ignore=['target']):
    """
    Transform the train and test stes in equivalent versions.
    """
    print('Creating dummies')
    col_ignore=list(col_ignore)
    train=dummy_conversion(train,threshold,identify_categories(train))
    test=dummy_conversion(test,threshold,identify_categories(test))
    train_names= list(train)
    for i in train_names:
        if i in col_ignore:
            train_names.remove(i)

    test_names = list(test)

    print('Creating features in train')
    for i in train_names:
        if (i not in test_names):
            print('The feature '+i+' is not included in the test set. Accion: create the feature with value=0.')
            test[i]=0
    
    print('Deleting extra varibles in train')
    for i in test_names:
        if i not in train_names:
            print('The feature '+i+' is not included in the training set. Accion: delete the feature')
            del test[i]
            
    test = test[train_names]
    return train, test


if __name__ == '__main__':
    df_test=pd.read_csv('data/test.csv')
    df_train=pd.read_csv('data/train.csv')
    df_train,df_test=create_submission(df_train, df_test, 50, ['target'])