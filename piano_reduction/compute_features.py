from piano_reduction import features as ft
from piano_reduction import tools as pr
import pandas as pd
import numpy as np

def feature_count():
    """return the number of usable features
    """
    all_features = []
    for i in dir(ft):
        if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
            all_features.append((i, getattr(ft, i)))
    return len(all_features)

def get_features():
    """return the list of usable features
    """
    all_features = []
    for i in dir(ft):
        if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
            all_features.append(i)
    return all_features

def transform(df):
    """(I think it is useless now)
        I can't find it anywhere now
    """
    import pandas as pd
    all_features = []
    for i in dir(ft):
        if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
            all_features.append((i, getattr(ft, i)))
    df2 = pd.DataFrame(index = df.index)
    for feature, call in all_features:
        df2[feature] = df['x_train'] * call(df)
    return df2

def load_features(num, use_cache=True, cosi=False):
    """(I think it is useless now)
        I can't find it anywhere now too
    """
    from os import listdir, path, mkdir
    if not path.exists('df'):
        mkdir('df')
    if not path.exists('df/features'):
        mkdir('df/features')
    if cosi:
        df = pr.load_cosi(num, use_cache=use_cache)[0]
        if not path.exists('df/features/cosi'):
            mkdir('df/features/cosi')
        filename = 'df/features/cosi/%d.pkl' % num
    else:
        df = pr.load_merged_df(num, use_cache=use_cache)[0]
        if not path.exists('df/features/merged'):
            mkdir('df/features/merged')
        filename = 'df/features/merged/%d.pkl' % num
    if use_cache and path.isfile(filename):
        df2 = pd.read_pickle(filename)
        return df2[df['x_train'] == 1], df[df['x_train'] == 1][['y_train']]
    else:
        print('Creating %s' % filename)
        df2 = transform(df)
        df2.to_pickle(filename)
        return df2[df['x_train'] == 1], df[df['x_train'] == 1][['y_train']]

"""Turns out everything is useless now"""

def generate_data(sample, length=20):
    x_ = None
    y_ = None
    for ind, s in enumerate(sample):
        x, y = s
        count = len(x.index) - length + 1
        if ind > 0:
            x_ = np.append(x_, np.array([x[i: i + length].values for i in range(count)]), axis=0)
            y_ = np.append(y_, np.array([y[i: i + length].values for i in range(count)]), axis=0)
        else:
            x_ = np.array([x[i: i + length].values for i in range(count)])
            y_ = np.array([y[i: i + length].values for i in range(count)])
    return x_, y_

def generate_data_2(sample, length=20):
    x_ = None
    y_ = None
    for ind, s in enumerate(sample):
        x, y = s
        count = x.shape[0] - length + 1
        if ind > 0:
            x_ = np.append(x_, np.array([x[i: i + length, :] for i in range(count)]), axis=0)
            y_ = np.append(y_, np.array([y[i: i + length, :] for i in range(count)]), axis=0)
        else:
            x_ = np.array([x[i: i + length, :] for i in range(count)])
            y_ = np.array([y[i: i + length, :] for i in range(count)])
    return x_, y_

def df_to_onehot(df, col='x_train'):
    tmp = 0
    last = None
    x = np.zeros((1, 128))
    for i, row in df.iterrows():
        if last  and last != (row['measure'], row['offset']):
            if tmp == 0:
                x_ = x
                tmp = 1
            else:
                x_ = np.append(x_, x, axis=0)
            x = np.zeros((1, 128))
        if row[col] == 1:
            x[0, int(row['ps'])] = 1
        last = (row['measure'], row['offset'])
    if tmp == 0:
        x_ = x
    else:
        x_ = np.append(x_, x, axis=0)
    return x_

import copy
def onehot_to_df(df, v, name='y_pred'):
    df[name] = pd.Series(np.zeros((len(df.index))))
    p = 0
    df2 = df.copy()
    v = np.copy(v)
    last = None

    for i, row in df.iterrows():
        if i != df.index[0] and (last['measure'], last['offset']) != (row['measure'], row['offset']):
            for j in range(128):
                if v[p, j] == 1:
                    tmp = last.copy()
                    tmp['x_train'] = 0
                    tmp['y_train'] = 0
                    tmp['ps'] = j * 1.0
                    tmp['pitch'] = copy.copy(tmp['pitch'])
                    tmp['pitch'].ps = j
                    tmp['part'] = -2
                    tmp['y_pred'] = 1
                    df2 = df2.append(tmp)
            p += 1
            if p == v.shape[0]:
                break
        df2.at[i, name] = v[p, int(row['ps'])]
        v[p, int(row['ps'])] = 0
        last = row
    if p < v.shape[0]:
        for j in range(128):
            if v[p, j] == 1:
                tmp = last.copy(deep=True)
                tmp['x_train'] = 0
                tmp['y_train'] = 0
                tmp['ps'] = j * 1.0
                tmp['pitch'] = copy.copy(tmp['pitch'])
                tmp['pitch'].ps = j
                tmp['part'] = -2
                tmp['y_pred'] = 1
                df2 = df2.append(tmp)
    df2 = df2.sort_values(['measure', 'offset'])
    df2[name] = df2[name].astype(int)
    return df2
