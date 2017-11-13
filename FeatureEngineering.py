# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:23:12 2017

@author: s157084
"""

import pandas as pd
import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from SoftImpute import *



def PCA_feature_reduction(X,threshold=0.95,plot_import=False):

    """ 
    This function redues the number of dimensions of the matrix X using Principal component analysis. This code follows
    the guidelines from:
    
    https://github.com/llSourcell/Dimensionality_Reduction/blob/master/principal_component_analysis.ipynb
    
    Parameters
    ----------
    X = data frame with idependent variables.
    Y = series with dependent variable.
    threshold = min % of information that has to be preserved
    ----------
    X_red = data frame with reduced dimentions.
    """
    names_x = list(X)
    X_std = StandardScaler().fit_transform(X)
    
    #u,s,v = np.linalg.svd(X_std.T)
    n_components = X_std.shape[1]
    pca = sklearnPCA(n_components=n_components)
    pca.fit_transform(X_std)
    X_pca = pd.DataFrame(pca.transform(X_std))

    U = pca.components_ #(n_components, n_features)
    #Select the components that provides mode information 
    cum_sum = np.cumsum(pca.explained_variance_ratio_)
    cum_sum = cum_sum[cum_sum<threshold]
    
    number_factors = len(cum_sum)
    X_pca = X_pca.T
    X_pca = X_pca[0:number_factors]
    X_pca = X_pca.T
    
    U = U.T
    U = U[0:number_factors]
    U = U.T
    
    #calculate importnace 
    U = U**2
    importance = np.sum(U,axis=1)

    importance = importance/np.sum(importance)
    importance = pd.DataFrame(importance,columns=['Importnace (%)'])
    names_x=pd.DataFrame(names_x,columns=['Var Name'])
    importance=pd.concat([names_x,importance],axis=1)
    importance = importance.sort_values(by=['Importnace (%)'],ascending=[0])
    importance.reset_index(drop=True)
    
    if plot_import == True:
        sns.set(font_scale=2)
        f, ax = plt.subplots(figsize=(18, 15))
        plt.title('Feature Importance: ')
        sns.barplot(y="Var Name", x="Importnace (%)", data=importance, color="b",label= " n_categories= " + str(X_pca.shape[1]))
        ax.set(xlabel='Importnace (%)', ylabel='')
        plt.show()
        
    return X_pca


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

def add_na_count(df):
    df['na_count'] = df.isnull().sum(axis=1)
    return df

# Untested, use at own risk
def normalize(df, method='minmax'):
    """
    Normalizes the features of the training set, leaving the target variable intact
    :param df: The pd.DataFrame to be normalized.
    :param method: a String defining the normalization method to be used. Minmax and standard are supported.
    :return: The normalized pd.DataFrame.
    """

    # We should only normalize the features, not the label
    if 'target' in list(df):
        target = df['target']
        df.drop('target', axis=1)

    if method == 'minmax':
        normalized = (df - df.min()) / (df.max() - df.min())
        normalized['target'] = target
        return normalized

    if method == 'standard':
        normalized = (df - df.mean()) / df.std()
        normalized['target'] = target
        return normalized

    raise NotImplementedError("Supported normalization methods: 'minmax', 'standard'")
    
    
    
def identify_categories(df,match_word='_cat'):
        """
        Identify the categorical functions
        """
        cols = list(df)
        return list(filter(lambda col: col.endswith(match_word), cols))


def dummy_conversion(df, threshold):
    """
    Transform the columns with strings to features only if the number of dummy variables 
    created are smaller than the the threshold
    
    """
    categories = identify_categories(df)
    list_names = []
    for c in df.columns:
        if c in categories:
            df[c] = df[c].astype('category')
            n = len(df[c].cat.categories)
                
            if n <= threshold:
                if n<3:
                    print("Variable "+str(c)+" has "+ str(n) + " categories. Action: No dummy transformation.")
                    df[c] = df[c].astype('f')
                else:
                    list_names.append(c)
            else:
                print("Dropping variable " + str(c))
                del df[c]

    print('The features that will be transformed are:')
    print(list_names)
    df = pd.get_dummies(df, columns=list_names)
    return df


def soft_impute(df):
    """
    Fill NaN values of a data frame based on the research:
        Spectral Regularization Algorithms for Learning Large Incomplete Matrices
    """
    id = df['id']
    df = df.drop('id', axis=1)

    categories = identify_categories(df)
    names = list(df)
    df = pd.DataFrame.as_matrix(df)
    clf = SoftImpute().fit(df)
    X_imp = clf.predict(df.copy())
    df = pd.DataFrame(df,columns=names)
    X_imp = pd.DataFrame(X_imp,columns=names)
    
    for i in names:
        #Ensures Categorical transformation
        if i in categories:
            imput=X_imp[i][df[i].isnull()].astype(int)
        else:
            imput=X_imp[i][df[i].isnull()]
                
        df[i][df[i].isnull()] = imput

    df['id'] = id
    return df

if __name__ == '__main__':
    df_test = pd.read_csv('data/test.csv',na_values=[-1])
    aux = dummy_conversion(df_test, 40)
    aux2 = soft_impute(aux)
    aux3 = PCA_feature_reduction(aux2,threshold=0.95,plot_import=True)
    
    