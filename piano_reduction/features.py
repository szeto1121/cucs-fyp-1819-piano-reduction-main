import pandas as pd
import numpy as np
from collections import defaultdict

"""These functions take the dataframe as input and return a dataframe column containing the feature values.
    Each musical feature is implemented (hopefully) based on the definition shown in the reports
"""

def in_chord(df):
    return df['color'].apply(lambda x: (not pd.isnull(x) and x != '#000000')*1)

def lowest(df):
    s = pd.Series()
    ps = defaultdict(lambda: 999)
    for ind, row in df.iterrows():
        if row['x_train'] == 0:
            continue
        if row['ps'] < ps[(row['measure'], row['offset'])]:
            ps[(row['measure'], row['offset'])] = row['ps']
    for ind, row in df.iterrows():
        s.loc[ind] = (row['ps'] == ps[(row['measure'], row['offset'])]) * 1
    return s

def active_rhythm(df):
    counts = defaultdict(lambda : 0)
    for ind, row in df.iterrows():
        counts[row['measure'], row['part']] += 1
    most = defaultdict(lambda : 0)
    for measure, part in counts:
        if counts[measure, part] > most[measure] and part >= 0:
            most[measure] = counts[measure, part]
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = (counts[row['measure'], row['part']] == most[row['measure']]) * 1
    return s

def bass_line(df):
    notes = defaultdict(list)
    for ind, row in df.iterrows():
        notes[row['measure'], row['part']].append(row['ps'])
    lowest = defaultdict(lambda : 10000)
    for measure, part in notes:
        notes[measure, part] = np.median(notes[measure, part])
        if np.median(notes[measure, part]) < lowest[measure] and part >= 0:
            lowest[measure] = np.median(notes[measure, part])
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = (notes[row['measure'], row['part']] == lowest[row['measure']]) * 1
    return s

def entrance_effect(df):
    s = pd.Series()
    last_time = defaultdict(lambda : 0)
    current_time = defaultdict(lambda : 0)
    for ind, row in df.iterrows():
        if row['measure'] + row['offset'] > current_time[row['part']]:
            last_time[row['part']] = row['measure'] + row['offset']
        s.loc[ind] = row['measure'] + row['offset'] - last_time[row['part']]
        current_time[row['part']] = max(current_time[row['part']], row['measure'] + row['offset'] + row['duration'].quarterLength)
    return s

def harmony(df):
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = 1
    return s

def highest(df):
    s = pd.Series()
    ps = defaultdict(lambda: 0)
    for ind, row in df.iterrows():
        if row['x_train'] == 0:
            continue
        if row['ps'] > ps[(row['measure'], row['offset'])]:
            ps[(row['measure'], row['offset'])] = row['ps']
    for ind, row in df.iterrows():
        s.loc[ind] = (row['ps'] == ps[(row['measure'], row['offset'])]) * 1
    return s

def occurrence(df):
    s = pd.Series()
    counts = defaultdict(lambda: 0)
    for ind, row in df.iterrows():
        if row['x_train'] == 0:
            continue
        counts[row['measure'], row['part'], row['pitch'].name] += 1
    highest = defaultdict(lambda: 2)
    for measure, part, name in counts:
        highest[measure, part] = max(highest[measure, part], counts[measure, part, name])
    for ind, row in df.iterrows():
        s.loc[ind] = (counts[row['measure'], row['part'], row['pitch'].name] == highest[row['measure'], row['part']]) * 1
    return s

def offset_value(df):
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = row['measure'] + row['offset']
    return s

def onset_after_rest(df):
    s = pd.Series()
    current_time = defaultdict(lambda : 0)
    for ind, row in df.iterrows():
        s.loc[ind] = (row['measure'] + row['offset'] > current_time[row['part']]) * 1
        current_time[row['part']] = max(current_time[row['part']], row['measure'] + row['offset'] + row['duration'].quarterLength)
    return s

def pitch_distance(df):
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = abs(row['ps'] - 60)
    return s

def rhythm_variety(df):
    s = pd.Series()
    current_time = defaultdict(lambda : 0)
    last_length = defaultdict(lambda : 0)
    last_index = defaultdict(lambda : 0)
    for ind, row in df.iterrows():
        s.loc[ind] = (row['measure'] + row['offset'] > current_time[row['part']]) * 1
        if row['duration'].quarterLength != last_length[row['part']]:
            s.loc[ind] = 1
            s.loc[last_index[row['part']]] = 1
        current_time[row['part']] = max(current_time[row['part']], row['measure'] + row['offset'] + row['duration'].quarterLength)
        last_length[row['part']] = row['duration'].quarterLength
        last_index[row['part']] = ind
    return s

def strong_beats(df):
    return df['offset'].apply(lambda x: (float(x)).is_integer())

def sustained_rhythm(df):
    counts = defaultdict(lambda : 0)
    for ind, row in df.iterrows():
        counts[row['measure'], row['part']] += 1
    least = defaultdict(lambda : 10000)
    for measure, part in counts:
        if counts[measure, part] < least[measure] and part >= 0:
            least[measure] = counts[measure, part]
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = (counts[row['measure'], row['part']] == least[row['measure']]) * 1
    return s

def vertical_doubling(df):
    stats = defaultdict(lambda: 0)
    for ind, row in df.iterrows():
        if row['x_train'] == 0:
            continue
        stats[(row['measure'], row['pitch'].name)] += 1
    s = pd.Series()
    for ind, row in df.iterrows():
        s.loc[ind] = (stats[(row['measure'], row['pitch'].name)] > 1) * 1
    return s

def duration_length(df):
    return df['duration'].apply(lambda x: x.quarterLength)
