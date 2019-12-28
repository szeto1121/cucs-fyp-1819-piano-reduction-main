from piano_reduction import tools as pr, features as ft, compute_features as cf
import numpy as np
import pickle
class ScoreData():
    def __init__(self, dfs, skip_features=False):
        self.df = dfs[0]
        self.other = dfs[1:]
        if not skip_features:
            import pandas as pd
            all_features = []
            for i in dir(ft):
                if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
                    all_features.append((i, getattr(ft, i)))
            for feature, call in all_features:
                self.df[feature] = self.df['x_train'] * call(self.df)
            
    def load_features(self, x_train=False):
        all_features = []
        for i in dir(ft):
            if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
                all_features.append(i)
        df = self.df[all_features]
        if not x_train:
            return df, self.df['y_train']
        else:
            return df[self.df['x_train'] == 1], self.df['y_train'][self.df['x_train'] == 1]
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            temp = pickle.load(f)
        return temp
    
    def generate_data(self, length=20):
        x, y = self.load_features(x_train=True)
        x_values = x.values
        y_values = y.values
        temp = []
        if length > 0:
            que = [np.zeros(x.values[0].shape) for _ in range(length)]
            for i in range(len(x_values)):
                que.append(x_values[i])
                que.pop(0)
                temp.append(np.array(que[:]))
            return np.array(temp), np.array(y_values)
        else:
            return np.array(x_values), np.array(y_values)
        
    
    def split(self, training_ratio=0.5):
        half = self.other[1].index[int(len(self.other[1].index) * training_ratio)]
        df1 = [self.df[self.df['measure'] < half].copy(), 
               self.other[0][self.other[0]['measure'] < half].copy(),
               self.other[1][self.other[1].index < half].copy()
              ]
        df2 = [self.df[self.df['measure'] >= half].copy(), 
               self.other[0][self.other[0]['measure'] >= half].copy(),
               self.other[1][self.other[1].index >= half].copy()
              ]
        return ScoreData(df1, skip_features=True), ScoreData(df2, skip_features=True)
    
    def kfold(self, k=2):
        cuts = [self.other[1].index[int(len(self.other[1].index) * i / k)] for i in range(k)]
        cuts.append(self.other[1].index[-1] + 1)
        df1 = [(ScoreData([self.df[~((cuts[i] <= self.df['measure']) & (self.df['measure'] < cuts[i + 1]))].copy(), 
                self.other[0][~((cuts[i] <= self.other[0]['measure']) & (self.other[0]['measure'] < cuts[i + 1]))].copy(),
                self.other[1][~((cuts[i] <= self.other[1].index) & (self.other[1].index < cuts[i + 1]))].copy()
               ], skip_features=True),
                ScoreData([self.df[(cuts[i] <= self.df['measure']) & (self.df['measure'] < cuts[i + 1])].copy(), 
                self.other[0][(cuts[i] <= self.other[0]['measure']) & (self.other[0]['measure'] < cuts[i + 1])].copy(),
                self.other[1][(cuts[i] <= self.other[1].index) & (self.other[1].index < cuts[i + 1])].copy()
               ], skip_features=True)) for i in range(k)]
        return df1
    
    def get_y_pred(self, model, length=20, threshold = 0.5):
        x_pred, y_test = self.generate_data(length=length)
        tmp = np.hstack(model.predict(x_pred))
        pred = (tmp > threshold) * 1
        tmp = []
        count = 0
        for i in self.df.index:
            if self.df.loc[i, 'x_train'] == 0:
                tmp.append(0)
            else:
                tmp.append(pred[count])
                count += 1
        new_data = self.copy()
        new_data.df['y_pred'] = tmp
        return new_data
        
    def show_score(self, col=['x_train', 'y_pred', 'y_train'], colors=[]):
        col = [i for i in col if i in self.df]
        colors = [i for i in colors if i in self.df]
        pr.df_to_scores(self.df, self.other[0], self.other[1], col=col, colors=colors).show('musicxml')
        
    def show_score_with_colors(self, col=['x_train'], colors=[]):
        col = [i for i in col if i in self.df]
        
        
        
    def copy(self):
        return ScoreData([self.df] + self.other, skip_features=True)
    
    def to_binary(self, col):
        tmp = 0
        last = None
        x = np.zeros((1, 128))
        for i, row in self.df.iterrows():
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
    
    def merge_binary(self, v, name='y_pred', skip_features=False):
        import copy
        p = 0
        df2 = self.df.copy()
        v = np.copy(v)
        last = None

        for i, row in self.df.iterrows():
            if i != self.df.index[0] and (last['measure'], last['offset']) != (row['measure'], row['offset']):
                for j in range(128):
                    if v[p, j] == 1:
                        tmp = last.copy()
                        tmp['x_train'] = 0
                        tmp['y_train'] = 0
                        tmp['ps'] = j * 1.0
                        tmp['pitch'] = copy.copy(tmp['pitch'])
                        tmp['pitch'].ps = j
                        tmp['part'] = -2
                        tmp[name] = 1
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
                    tmp[name] = 1
                    df2 = df2.append(tmp)
        df2 = df2.sort_values(['measure', 'offset', 'ps'])
        df2[name] = df2[name].astype(int)
        return ScoreData([df2] + self.other, skip_features=skip_features)
    
    def generate_data_2(self, length=20):
        x = self.to_binary(col='x_train')
        y = self.to_binary(col='y_train')
        if length > 0:
            temp = []
            que = [np.zeros(x[0].shape) for _ in range(length)]
            for i in range(x.shape[0]):
                que.append(x[i])
                que.pop(0)
                temp.append(np.array(que[:]))
            return np.array(temp), np.array(y)
        else:
            return np.array(x), np.array(y)
        
    def to_binary_2(self, col):
        end_time = self.other[1].index[-1] + self.other[1].iloc[-1]
        min_interval = end_time
        last = 0
        for _, row in self.df.iterrows():
            if row['measure'] + row['offset'] != last:
                min_interval = min(min_interval, row['measure'] + row['offset'] - last)
                last = row['measure'] + row['offset']



        offsets = [0]
        while end_time - offsets[-1] > 1e-7:
            offsets.append(offsets[-1] + min_interval)

        v = np.zeros((len(offsets), 128))

        for _, row in self.df.iterrows():
            if row[col] != 1:
                continue
            l = 0
            r = v.shape[0] - 1
            while l <= r:
                mid = (l + r) // 2
                if abs(offsets[mid] - row['measure'] - row['offset']) < 1e-7:
                    break
                elif offsets[mid] < row['measure'] + row['offset']:
                    l = mid + 1
                else:
                    r = mid - 1
            v[mid, int(row['ps'])] = 2
            while mid < v.shape[0] - 1 and row['offset'] + row['measure'] + row['duration'].quarterLength > offsets[mid + 1] + 1e-7:
                mid += 1
                v[mid, int(row['ps'])] = 1
        
        return v
    
    def generate_data_3(self, length=20):
        x = self.to_binary_2(col='x_train')
        y = self.to_binary_2(col='y_train')
        temp = []
        que = [np.zeros(x[0].shape) for _ in range(length)]
        for i in range(x.shape[0]):
            que.append(x[i])
            que.pop(0)
            temp.append(np.array(que[:]))
        return np.array(temp), np.array(y)
    
    def to_binary_3(self, col, features=cf.get_features()):
        tmp = 0
        last = None
        feature_count = len(features)
        x = np.zeros((1, 128 + 128 * feature_count))
        for i, row in self.df.iterrows():
            if last  and last != (row['measure'], row['offset']):
                if tmp == 0:
                    x_ = x
                    tmp = 1
                else:
                    x_ = np.append(x_, x, axis=0)
                x = np.zeros((1, 128 + 128 * feature_count))
            if row[col] == 1:
                x[0, int(row['ps'])] = 1
                x[0, 128 + int(row['ps']) * feature_count: 128 + (int(row['ps']) + 1) * feature_count] = np.array(row[features]).astype(float)
            last = (row['measure'], row['offset'])
        if tmp == 0:
            x_ = x
        else:
            x_ = np.append(x_, x, axis=0)
        return x_ 
    
    def generate_data_4(self, length=20, features=cf.get_features()):
        x = self.to_binary_3(col='x_train', features=features)
        y = self.to_binary(col='y_train')
        if length > 0:
            temp = []
            que = [np.zeros(x[0].shape) for _ in range(length)]
            for i in range(x.shape[0]):
                que.append(x[i])
                que.pop(0)
                temp.append(np.array(que[:]))
            return np.array(temp), np.array(y)
        else:
            return np.array(x), np.array(y)
        
    def get_color(self, old, new):
        tmp = self.df[old] * 2 + self.df[new]
        tmp = tmp.apply(lambda x: 'black' if x == 3 else ('red' if x == 2 else 'green'))
        return tmp