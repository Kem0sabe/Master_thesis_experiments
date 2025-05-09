import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


from gower_mix import gower_distance
def load_gower_matrix(df1, df2=None,alpha=1,remove_self=False):
    df1 = df1.copy()
    if df2 is None:
        df2 = df1.copy()
    
    dist_matrix = gower_distance(df1,df2,alpha=alpha)
    if remove_self:
        dist_matrix = dist_matrix[~np.eye(dist_matrix.shape[0],dtype=bool)].reshape(dist_matrix.shape[0],-1)
    return dist_matrix



def euclidean_distance(df1, df2=None, categorical_columns=[], remove_self=False):
    if df2 is None:
        df2 = df1

    # Identify numerical columns
    numerical_columns = df1.columns.difference(categorical_columns)

    # One-hot encode categorical columns
    if categorical_columns:

        df1[categorical_columns] = df1[categorical_columns].astype(str)
        df2[categorical_columns] = df2[categorical_columns].astype(str)
        
        df1_cat = pd.get_dummies(df1[categorical_columns])
        df2_cat = pd.get_dummies(df2[categorical_columns])

        # Align encoded categorical features
        df1_cat, df2_cat = df1_cat.align(df2_cat, join='outer', axis=1, fill_value=0)
    else:
        df1_cat = df2_cat = pd.DataFrame(index=df1.index), pd.DataFrame(index=df2.index)

    # Standardize numerical columns
    scaler = StandardScaler()
    df1_num = scaler.fit_transform(df1[numerical_columns])
    df2_num = scaler.transform(df2[numerical_columns])

    # Combine numerical and categorical features
    df1_all = np.hstack([df1_num, df1_cat.values])
    df2_all = np.hstack([df2_num, df2_cat.values])

    # Compute distances
    distances = pairwise_distances(df1_all, df2_all, metric='euclidean')

    if remove_self:
        distances = distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1)

    return distances
