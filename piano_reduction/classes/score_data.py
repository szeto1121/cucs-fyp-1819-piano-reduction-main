from piano_reduction import tools as pr, features as ft, compute_features as cf
import numpy as np
import pickle
class ScoreData():
    def __init__(self, dfs, skip_features=False):
        """Instantiate the ScoreData using the lists of Dataframes

        Keyword arguments:
        dfs -- the list of Dataframes that should be storing the notes,
            miscellaneous elements and duration of each measures respectively
        skip_features -- whether the features should be computed or not
        """
        self.df = dfs[0]
        self.other = dfs[1:]

        if 'y_train' not in self.df:
            import pandas as pd
            self.df['y_train'] = pd.Series(index=self.df.index).fillna(0)
        if 'x_train' not in self.df:
            import pandas as pd
            self.df['x_train'] = pd.Series(index=self.df.index).fillna(1)
        if not skip_features:
            import pandas as pd
            all_features = []
            for i in dir(ft):
                if callable(getattr(ft, i)) and (i not in ['pd', 'np', 'defaultdict']) and '__' not in i:
                    all_features.append((i, getattr(ft, i)))
            for feature, call in all_features:
                self.df[feature] = self.df['x_train'] * call(self.df)

    def load_features(self, x_train=False):
        """(Seems useless now) return numpy arrays of features of its notes

        Keyword arguments:
        x_train -- whether the first array only return the values of notes that is in the original score (x_train)
        """
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
        """Save the ScoreData

        Keyword arguments:
        filename -- the desired filename for the ScoreData
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load the saved ScoreData

        Keyword arguments:
        filename -- the filename of the saved ScoreData
        """
        with open(filename, 'rb') as f:
            temp = pickle.load(f)
        return temp

    @classmethod
    def read_xml(cls, filename):
        """Directly convert a musicXML file into a ScoreData

        Keyword arguments:
        filename -- the filename of the musicXML file
        """
        import music21
        score = music21.converter.parse(filename)
        dfs = pr.score_to_df(score)
        data = ScoreData(dfs)
        return data

    def generate_data(self, length=20):
        """(I think it is useless now)
            Return the 3-dimensional numpy array for some old model
        """
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
        """Split itself into two ScoreData, one with former part of the piece and one with latter

        Keyword arguments:
        training_ratio -- the percentage of score put into the first output ScoreData
        """
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

    def split_in(self, testing_from=0.8, testing_to=0.9):
        """Split itself into two ScoreData by specifying when the second ScoreData starts and ends
            The first ScoreData will be the remaining parts

        Keyword arguments:
        testing_from -- the percentage when the second ScoreData starts from
        testing_to -- the percentage when the second ScoreData ends from
        """
        h1 = self.other[1].index[int(len(self.other[1].index) * testing_from)]
        try:
            h2 = self.other[1].index[int(len(self.other[1].index) * testing_to)]
        except:
            h2 = 1000000000
        df1 = [self.df[~((h1 <= self.df['measure']) & (self.df['measure'] < h2))].copy(),
               self.other[0][~((h1 <= self.other[0]['measure']) & (self.other[0]['measure'] < h2))].copy(),
               self.other[1][~((h1 <= self.other[1].index) & (self.other[1].index < h2))].copy()
              ]
        df2 = [self.df[(h1 <= self.df['measure']) & (self.df['measure'] < h2)].copy(),
               self.other[0][(h1 <= self.other[0]['measure']) & (self.other[0]['measure'] < h2)].copy(),
               self.other[1][(h1 <= self.other[1].index) & (self.other[1].index < h2)].copy()
              ]
        return ScoreData(df1, skip_features=True), ScoreData(df2, skip_features=True)

    def kfold(self, k=2):
        """(I think it is useless now)
            Returns a k-fold of two splits of ScoreData
        """

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
        """(I think it is useless now)
        """
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

    def show_score(self, col=['x_train', 'y_pred', 'y_train'], staff_name=[]):
        """Display the notes of the ScoreData using MuseScore

        Keyword arguments:
        col -- column names to be shown
        staff_name -- the display names of the columns
        """
        col = [i for i in col if i in self.df]
        tmp_data = self.copy()
        for i in range(min(len(col), len(staff_name))):
            tmp_data.df[staff_name[i]] = tmp_data.df[col[i]]
            col[i] = staff_name[i]

        pr.df_to_scores(tmp_data.df, tmp_data.other[0], tmp_data.other[1], col=col).show('musicxml')

    def show_score_with_colors(self, col=['x_train'], colors=[]):
        """Display the notes of the ScoreData using MuseScore, some with colors

        Keyword arguments:
        col -- column names to be shown
        colors -- column names to be shown with colors
        """
        col = [i for i in col if i in self.df]
        colors = [i for i in colors if i in self.df]

        pr.df_to_scores(self.df, self.other[0], self.other[1], col=col, colors=colors).show('musicxml')



    def copy(self):
        """Return a copy of itself so that you won't ruin the original one
        """
        return ScoreData([self.df.copy()] + [i.copy() for i in self.other], skip_features=True)

    def to_binary(self, col):
        """Return a binary array representing the notes of every moment.

        Keyword arguments:
        col -- the column to be represented
        """
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
        """Return a new ScoreData with the output numpy array merged as the column

        Keyword arguments:
        v -- the numpy array to be merged
        name -- the column name marking the newly merged notes
        skip_features -- whether the features should be computed again or not
        """
        import copy
        p = 0
        df2 = self.df.copy()
        v = np.copy(v)
        last = None

        ids = set()
        for i, row in self.df.iterrows():
            ids.add(i)
        for i, row in self.df.iterrows():
            if i != self.df.index[0] and (last['measure'], last['offset']) != (row['measure'], row['offset']):
                count = 0
                for j in range(128):
                    if v[p, j] == 1:
                        tmp = last.copy()
                        tmp.name += 100000 + count * 10000
                        while tmp.name in ids:
                            tmp.name += 100000
                        ids.add(tmp.name)
                        count += 1
                        tmp['x_train'] = 0
                        tmp['y_train'] = 0
                        tmp['ps'] = j * 1.0
                        tmp['pitch'] = copy.copy(tmp['pitch'])
                        tmp['pitch'].ps = j
                        tmp['part'] = -2
                        for ind, val in tmp.iteritems():
                            if 'y_pred' in ind:
                                tmp[ind] = 0
                        tmp[name] = 1
                        df2 = df2.append(tmp)
                p += 1
                if p == v.shape[0]:
                    break
            df2.at[i, name] = v[p, int(row['ps'])]
            v[p, int(row['ps'])] = 0
            last = row
        if p < v.shape[0]:
            count = 0
            for j in range(128):
                if v[p, j] == 1:
                    tmp = last.copy(deep=True)
                    tmp.name += 100000 + count * 10000
                    while tmp.name in ids:
                        tmp.name += 100000
                    ids.add(tmp.name)
                    count += 1
                    tmp['x_train'] = 0
                    tmp['y_train'] = 0
                    tmp['ps'] = j * 1.0
                    tmp['pitch'] = copy.copy(tmp['pitch'])
                    tmp['pitch'].ps = j
                    tmp['part'] = -2
                    for ind, val in tmp.iteritems():
                        if 'y_pred' in ind:
                            tmp[ind] = 0
                    tmp[name] = 1
                    df2 = df2.append(tmp)
        df2 = df2.sort_values(['measure', 'offset', 'ps'])
        df2[name] = df2[name].astype(int)
        return ScoreData([df2] + self.other, skip_features=skip_features)

    def merge_binary_threshold(self, v, threshold=0.5, name='y_pred', skip_features=False):
        """(I think it is useless now)
            Same as the previous one but keep the notes if the value is greater than the threshold
        """
        import copy
        p = 0
        df2 = self.df.copy()
        v = np.copy(v)
        last = None

        for i, row in self.df.iterrows():
            if i != self.df.index[0] and (last['measure'], last['offset']) != (row['measure'], row['offset']):
                for j in range(128):
                    if v[p, j] > threshold:
                        tmp = last.copy()
                        tmp['x_train'] = 0
                        tmp['y_train'] = 0
                        tmp['ps'] = j * 1.0
                        tmp['pitch'] = copy.copy(tmp['pitch'])
                        tmp['pitch'].ps = j
                        tmp['part'] = -2
                        tmp['keep_score'] = v[p, j]
                        tmp[name] = 1
                        df2 = df2.append(tmp)
                p += 1
                if p == v.shape[0]:
                    break
            df2.at[i, name] = (v[p, int(row['ps'])] > threshold) * 1
            df2.at[i, 'keep_score'] = v[p, int(row['ps'])] * df2.at[i, name]
            v[p, int(row['ps'])] = 0
            last = row
        if p < v.shape[0]:
            for j in range(128):
                if v[p, j] > threshold:
                    tmp = last.copy(deep=True)
                    tmp['x_train'] = 0
                    tmp['y_train'] = 0
                    tmp['ps'] = j * 1.0
                    tmp['pitch'] = copy.copy(tmp['pitch'])
                    tmp['pitch'].ps = j
                    tmp['part'] = -2
                    tmp['keep_score'] = v[p, j]
                    tmp[name] = 1
                    df2 = df2.append(tmp)
        df2 = df2.sort_values(['measure', 'offset', 'ps'])
        df2[name] = df2[name].astype(int)
        return ScoreData([df2] + self.other, skip_features=skip_features)

    def generate_data_2(self, length=20):
        """Return the 3-dimensional numpy arrays for keras_models.dense,
            keras_models.lstm and keras_models.gru
           The first array is obtained using the original score (x_train)
           The second array is obtained using the answer score (y_train)

        Keyword arguments:
        length -- if it is greater than 0, it specifies the length of lstm and gru
                otherwise it is for keras_models.dense
        """
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
        """(I think it is useless now)
            Seems like a fail experiment
        """
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
        """(I think it is useless now)
            The continuation of the fail experiment
        """
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
        """Return a binary array representing the notes of every moment.
            Different from to_binary, it includes the musical features of each note as well.

        Keyword arguments:
        col -- the column to be represented
        features -- the list of feature names to be put into the array
        """
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
            if row[col] == 1 and 0 <= row['ps'] < 128:
                x[0, int(row['ps'])] = 1
                x[0, 128 + int(row['ps']) * feature_count: 128 + (int(row['ps']) + 1) * feature_count] = np.array(row[features]).astype(float)
            last = (row['measure'], row['offset'])
        if tmp == 0:
            x_ = x
        else:
            x_ = np.append(x_, x, axis=0)
        return x_

    def generate_data_4(self, length=20, features=cf.get_features()):
        """Return the 3-dimensional numpy arrays for keras_models.dense_with_features,
            keras_models.lstm_with_features and keras_models.gru_with_features
           The first array is obtained using the original score (x_train)
           The second array is obtained using the answer score (y_train)
           Similar to generate_data_2, but with musical features added

        Keyword arguments:
        length -- if it is greater than 0, it specifies the length of lstm and gru
                otherwise it is for keras_models.dense
        features -- the list of feature names to be put into the array
        """
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

    def generate_data_5(self, length=21, features=cf.get_features()):
        """Return the 3-dimensional numpy arrays for keras_models.bidirectional_gru
           The first array is obtained using the original score (x_train)
           The second array is obtained using the answer score (y_train)
           Similar to generate_data_4, but with both sides of adjacent notes

        Keyword arguments:
        length -- if it is greater than 0, it specifies the length of lstm and gru
                otherwise it is for keras_models.dense
        features -- the list of feature names to be put into the array
        """
        x = self.to_binary_3(col='x_train', features=features)
        y = self.to_binary(col='y_train')
        half = length // 2
        if length > 0:
            temp = []
            que = [np.zeros(x[0].shape) for _ in range(length)]
            for i in range(x.shape[0]):
                que.append(x[i])
                que.pop(0)
                if i >= half:
                    temp.append(np.array(que[:]))
            for i in range(half):
                que.append(np.zeros(x[0].shape))
                que.pop(0)
                temp.append(np.array(que[:]))
            return np.array(temp), np.array(y)
        else:
            return np.array(x), np.array(y)

    def get_color(self, old, new):
        """(I think it is useless now)
            Maybe some function for show_score_with_colors
        """
        tmp = self.df[old] * 2 + self.df[new]
        tmp = tmp.apply(lambda x: 'green' if x == 3 else ('black' if x == 2 else ('black' if x == 1 else 0)))
        return tmp

    def get_moments(self):
        """(I think it is useless now)
            Something for the fail one
        """
        t = set()
        last = 0
        for i, row in self.df.iterrows():
            t.add(row['measure'] + row['offset'])
            if row['measure'] > last:
                last = row['measure']
        t.add(last + self.other[1].loc[last])
        t = sorted(list(t))
        s = []
        for i in range(len(t) - 1):
            s.append(t[i + 1] - t[i])
        return s

    def transform(self):
        """(I think it is useless now)
            This performs poorly too
        """
        import music21
        from collections import defaultdict
        keys = defaultdict(lambda : defaultdict(lambda : 0))
        all_keys = []
        for ind, row in self.other[0].iterrows():
            if type(row['element']) is music21.key.KeySignature:
                keys[row['measure'] + row['offset']][int(music21.key.sharpsToPitch(row['element'].sharps).ps) - 60] += 1


        for time in keys:
            best_key_signature = None
            for ind in keys[time]:
                if (not best_key_signature) or keys[time][ind] > keys[time][best_key_signature]:
                    best_key_signature = ind
            all_keys.append([time, best_key_signature])

        all_keys.append([1e9, 0])
        new_data = self.copy()
        for i in range(len(all_keys) - 1):
            time = (all_keys[i][0], all_keys[i + 1][0])
            key = all_keys[i][1]
            for ind in new_data.df.index:
                if time[0] <= new_data.df.at[ind,'measure'] + new_data.df.at[ind,'offset'] < time[1]:
                    new_data.df.at[ind, 'ps'] = new_data.df.at[ind, 'ps'] - key
        return new_data

    def revert_transform(self):
        """(I think it is useless now)
            Same as above
        """
        import music21
        from collections import defaultdict
        keys = defaultdict(lambda : defaultdict(lambda : 0))
        all_keys = []
        for ind, row in self.other[0].iterrows():
            if type(row['element']) is music21.key.KeySignature:
                keys[row['measure'] + row['offset']][int(music21.key.sharpsToPitch(row['element'].sharps).ps) - 60] += 1


        for time in keys:
            best_key_signature = None
            for ind in keys[time]:
                if (not best_key_signature) or keys[time][ind] > keys[time][best_key_signature]:
                    best_key_signature = ind
            all_keys.append([time, best_key_signature])

        all_keys.append([1e9, 0])
        new_data = self.copy()
        for i in range(len(all_keys) - 1):
            time = (all_keys[i][0], all_keys[i + 1][0])
            key = all_keys[i][1]
            for ind in new_data.df.index:
                if time[0] <= new_data.df.at[ind,'measure'] + new_data.df.at[ind,'offset'] < time[1]:
                    new_data.df.at[ind, 'ps'] = new_data.df.at[ind, 'ps'] + key
        return new_data
