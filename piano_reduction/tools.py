import music21
import copy
from os import listdir, path, mkdir
from collections import OrderedDict
import pandas as pd
import numpy as np

def load_score(folder, num):
    """Return an music21 object of the specified musicXML file

    Keyword arguments:
    folder, num -- the num-th file of the folder is returned
    """
    files = sorted([f for f in listdir(folder) if f[0] != '.'])
    try:
    	filename = folder + '/' + files[num]
    except:
    	raise Exception(':O(')
    print(filename)
    return music21.converter.parse(filename)

def load_df(folder, num, use_cache = True):
    """Return the dataframes of the specified musicXML file

    Keyword arguments:
    folder, num -- the num-th file of the folder is returned
    use_cache -- whether the cached version should be used.
    			If not, a new one will be generated
    """
    files = sorted([f for f in listdir(folder) if f[0] != '.'])
    try:
        filename = folder + '/' + files[num]
    except:
        raise Exception(':O(')
    if not path.exists('df'):
        mkdir('df')
    if not path.exists('df/' + folder):
        mkdir('df/' + folder)
    if path.isfile('df/' + filename + '.0.pkl') and use_cache:
        res = []
        for i in range(3):
            res.append(pd.read_pickle('df/' + filename + ('.%d.pkl' % i)))
    else:
        print('Creating df/' + filename)
        res = score_to_df(music21.converter.parse(filename))
        for i in range(len(res)):
            res[i].to_pickle('df/' + filename + ('.%d.pkl' % i))
    return res

def load_merged_df(num, in_folder='input_with_chords', out_folder='output', use_cache = True):
    """Return the dataframes of the two specified musicXML files, one as original score and one as answer score

    Keyword arguments:
    in_folder, out_folder, num -- the num-th file of the two folders are used
    use_cache -- whether the cached version should be used.
    			If not, a new one will be generated
    """
    x_res = load_df(in_folder, num, use_cache)
    y_res = load_df(out_folder, num, use_cache)
    if not path.exists('df'):
        mkdir('df')
    if not path.exists('df/merged'):
        mkdir('df/merged')
    if path.isfile('df/merged/%d.pkl' % num) and use_cache:
        return [pd.read_pickle('df/merged/%d.pkl' % num)] + x_res[1:]
    else:
        print('Creating df/merged/%d.pkl' % num)
        res = merge_df(x_res[0], y_res[0])
        res.to_pickle('df/merged/%d.pkl' % num)
        return [res] + x_res[1:]

def load_cosi(num, use_cache = True):
    """Return the dataframes of the specified cosi musicXML file, combined with the original score of it

    Keyword arguments:
    num -- the num-th file of the Cosi fan tutte in the folder "cosi" are used
    use_cache -- whether the cached version should be used.
    			If not, a new one will be generated
    """
    x_res = load_df('cosi', 0, use_cache)
    y_res = load_df('cosi', num, use_cache)
    if not path.exists('df'):
        mkdir('df')
    if not path.exists('df/merged_cosi'):
        mkdir('df/merged_cosi')
    if path.isfile('df/merged_cosi/%d.pkl' % num) and use_cache:
        return [pd.read_pickle('df/merged_cosi/%d.pkl' % num)] + x_res[1:]
    else:
        print('Creating df/merged_cosi/%d.pkl' % num)
        res = merge_df(x_res[0], y_res[0])
        res.to_pickle('df/merged_cosi/%d.pkl' % num)
        return [res] + x_res[1:]
def score_to_df(original_score_0):
    """Convert the music21 object into dataframes
        *** The ties are broken now but they are not used currently ***
    """
    original_score_1 = copy.deepcopy(original_score_0)
    for part in original_score_1.parts:
        transpose = 0
        try:
            transpose = part.getInstrument().transposition.semitones
        except:
            transpose = 0
        #print(part.getInstrument(), transpose)
        try:
            for measure in part.getElementsByClass(music21.stream.Measure):
                for element in measure.notesAndRests:
                    #print(element)
                    if type(element) is music21.note.Note:
                        element.pitch.ps += transpose
                    elif type(element) is music21.chord.Chord:
                        for note in element._notes:
                            note.pitch.ps += transpose
                    #print('NOW ', element)
                for voice in measure.getElementsByClass(music21.stream.Voice):
                    #print(voice)
                    for element in voice.notesAndRests:
                        #print(element)
                        if type(element) is music21.note.Note:
                            element.pitch.ps += transpose
                        elif type(element) is music21.chord.Chord:
                            for note in element._notes:
                                note.pitch.ps += transpose
                        #print('NOW ', element)
        except:
            print("WHY")
    original_score = original_score_1.voicesToParts()
    score = music21.stream.Score()
    for element in original_score._elements:
        if type(element) is music21.text.TextBox or type(element) is music21.metadata.Metadata:
            score.insert(0, element)
    parts = {'L':music21.stream.Part(), 'R':music21.stream.Part()}
    parts['L'].insert(0, music21.instrument.fromString('Piano'))
    parts['R'].insert(0, music21.instrument.fromString('Piano'))
    hands = [0, 1]
    notes = pd.DataFrame(columns = ['measure', 'offset', 'part', 'pitch', 'ps', 'duration', 'tie', 'chosen', 'color'])
    others = pd.DataFrame(columns = ['measure', 'offset', 'element'])
    bar_lengths = pd.Series()
    nc = 0
    oc = 0
    original_measures = original_score.measureOffsetMap()
    xx = []
    measure_elements = {}
    for offset, measures in original_measures.items():
        measure = music21.stream.Measure()
        for original_measure in measures:
            if original_measure.barDuration.quarterLength > 0:
                bar_length = original_measure.barDuration.quarterLength
        bar_lengths.loc[offset] = bar_length
        pc = 0
        for original_measure in measures:
            for element in original_measure.notesAndRests:
                if type(element) is music21.note.Note:
                    notes.loc[nc] = [offset, element.offset, pc, element.pitch, element.pitch.ps, element.duration, element.tie, 1, element.style.color]
                    nc += 1
                elif type(element) is music21.chord.Chord:
                    for note in element._notes:
                        notes.loc[nc] = [offset, element.offset, pc, note.pitch, note.pitch.ps, element.duration, note.tie, 1, note.style.color]
                        nc += 1
            for element in original_measure.getElementsByClass(music21.key.KeySignature):
                others.loc[oc] = [offset, element.offset, element]
                oc += 1
            for element in original_measure.getElementsByClass(music21.tempo.MetronomeMark):
                others.loc[oc] = [offset, element.offset, element]
                oc += 1
            for element in original_measure.getElementsByClass(music21.dynamics.Dynamic):
                others.loc[oc] = [offset, element.offset, element]
                oc += 1
            for element in original_measure.getElementsByClass(music21.expressions.TextExpression):
                others.loc[oc] = [offset, element.offset, element]
                oc += 1
            for element in original_measure.getElementsByClass(music21.meter.TimeSignature):
                others.loc[oc] = [offset, element.offset, element]
                oc += 1
            pc += 1
    notes = notes.sort_values(['measure', 'offset', 'ps']).drop_duplicates(['measure', 'offset', 'ps'])
    return [notes, others, bar_lengths]

def insert_note(measure, offset, pitch=None, duration=None, tie=None, color=None):
    """Add a note into the measure
        *** There are no PARTS in the measure now, the durations might change
            when they cannot be fitted using one part ***
    """
    try:
        merge = False
        if pitch:
            note = music21.note.Note(pitch, quarterLength=duration.quarterLength)
            if color:
                note.style.color = color
            note.tie = tie
        else:
            note = None
        if len(measure.notesAndRests) > 0:
            if offset - measure.notesAndRests[-1].offset - measure.notesAndRests[-1].duration.quarterLength > 1e-9:
                r = music21.note.Rest()
                r.duration.quarterLength = offset - measure.notesAndRests[-1].offset - measure.notesAndRests[-1].duration.quarterLength
                measure.insert(measure.notesAndRests[-1].offset + measure.notesAndRests[-1].duration.quarterLength, r)
            elif abs(offset - measure.notesAndRests[-1].offset) < 1e-9:
                if note:
                    merge = True
            elif measure.notesAndRests[-1].offset + measure.notesAndRests[-1].duration.quarterLength - offset > 1e-9:
                # Force changing duration of previous note
                measure.notesAndRests[-1].duration.quarterLength = offset - measure.notesAndRests[-1].offset
        if len(measure.notesAndRests) == 0 and offset != 0:
            measure.insert(music21.note.Rest(quarterLength = offset))
        if pitch:
            if merge:
                if type(measure.notesAndRests[-1]) is music21.note.Note:
                    if measure.notesAndRests[-1].pitch != note.pitch:
                        chord = music21.chord.Chord([measure.notesAndRests[-1], note], offset = offset, duration = note.duration)
                        measure.remove(measure.notesAndRests[-1])
                        measure.insert(offset, chord)
                else:
                    if note.pitch not in map(lambda x:x.pitch,measure.notesAndRests[-1]._notes):
                        measure.notesAndRests[-1].add(note)
            else:
                chord = music21.chord.Chord([note], offset = offset, duration = note.duration)
                measure.insert(offset, chord)
    except:
        print(offset, pitch, duration, tie, note)
        measure.show('text')
        raise ':O)'

def df_to_score(notes, others, bar_lengths, col='chosen', color=False):
    """Convert the notes marked in a specific column of the dataframes into music21 object
    """
    score = music21.stream.Score()
    parts = {'L':music21.stream.Part(), 'R':music21.stream.Part()}
    parts['L'].insert(0, music21.instrument.fromString('Piano'))
    parts['R'].insert(0, music21.instrument.fromString('Piano'))
    hands = [0, 1]
    measure_elements = {}
    for offset in bar_lengths.index:
        measure = music21.stream.Measure()
        bar_length = bar_lengths.loc[offset]
        notes_and_rests = notes[notes['measure'] == offset]
        right_measure = music21.stream.Measure()
        left_measure = music21.stream.Measure()
        if offset == bar_lengths.index[0]:
            right_measure.clef = music21.clef.TrebleClef()
            left_measure.clef = music21.clef.BassClef()
        key_signatures = {}
        for ind, row in others[others['measure'] == offset].iterrows():
            if type(row['element']) is music21.key.KeySignature or type(row['element']) is music21.key.Key:
                #print('I AM A KEY', row['element'])
                if (row['offset'], row['element'].sharps) not in key_signatures:
                    key_signatures[(row['offset'], row['element'].sharps)] = 0
                key_signatures[(row['offset'], row['element'].sharps)] += 1
            elif type(row['element']) is music21.tempo.MetronomeMark:
                if not right_measure.getElementsByClass(music21.tempo.MetronomeMark):
                    right_measure.insert(row['offset'], row['element'])
            elif type(row['element']) is music21.dynamics.Dynamic:
                pass
            elif type(row['element']) is music21.meter.TimeSignature:
                right_measure.insert(row['offset'], row['element'])
                left_measure.insert(row['offset'], row['element'])
            else:
                right_measure.insert(row['offset'], row['element'])
        best_key_signature = None
        for ind in key_signatures:
            if (not best_key_signature) or key_signatures[ind] > key_signatures[best_key_signature]:
                best_key_signature = ind
        if best_key_signature:
            left_measure.insert(best_key_signature[0], music21.key.KeySignature(best_key_signature[1]))
            right_measure.insert(best_key_signature[0], music21.key.KeySignature(best_key_signature[1]))
        for ind, row in notes_and_rests.iterrows():
            if row[col] == 0:
                continue
            if row['pitch'].octave >= 4:
                insert_note(right_measure, row['offset'], row['pitch'], row['duration'], row['tie'], row[col] if color else None)
            else:
                insert_note(left_measure, row['offset'], row['pitch'], row['duration'], row['tie'], row[col] if color else None)
        left_measure.duration.quarterLength = bar_length
        right_measure.duration.quarterLength = bar_length
        insert_note(right_measure, bar_length)
        insert_note(left_measure, bar_length)
        parts['R'].insert(offset, right_measure)
        parts['L'].insert(offset, left_measure)
    score.insert(0, parts['R'])
    score.insert(0, parts['L'])
    return score

def df_to_scores(notes, others, bar_lengths, col='y_train', colors=[]):
    """Convert the notes marked in several columns of the dataframes into music21 object

    Keyword arguments:
    notes, others, bar_lengths -- the dataframes storing the information of the musical piece
    col -- list of column name to be output
    colors -- list of column name to be output with colors
    """
    if not type(col) is list:
        col = [col]
    score = music21.stream.Score()
    for i, c in enumerate(col):
        score_x = df_to_score(notes, others, bar_lengths, c)
        for part in score_x.parts:
            part.getInstrument().partName = c
            score.insert(0, part)
    for i, c in enumerate(colors):
        score_x = df_to_score(notes, others, bar_lengths, c, color=True)
        for part in score_x.parts:
            part.getInstrument().partName = c
            score.insert(0, part)
    return score

def merge_df(x, y):
    """Combine two dataframes into one
    """
    x = x.rename(columns={'chosen':'x_train'})
    x['offset'] = x['offset'].astype('object')
    x['measure'] = x['measure'].astype('object')
    y = y.rename(columns={'chosen':'y_train'})
    y['offset'] = y['offset'].astype('object')
    y['measure'] = y['measure'].astype('object')
    del y['tie'], y['color'], y['part']
    new_df = x.merge(y, how='outer', on=['measure', 'offset', 'ps'])
    new_df['x_train'] = new_df['x_train'].fillna(0)
    new_df['y_train'] = new_df['y_train'].fillna(0)
    new_df['part'] = new_df['part'].fillna(-1)
    new_df['duration'] = pd.Series()
    new_df['pitch'] = pd.Series()
    for ind in new_df.index:
        if pd.isnull(new_df.at[ind, 'duration_x']):
            new_df.at[ind, 'duration'] = new_df.at[ind, 'duration_y']
        else:
            new_df.at[ind, 'duration'] = new_df.at[ind, 'duration_x']
    for ind in new_df.index:
        if pd.isnull(new_df.at[ind, 'pitch_x']):
            new_df.at[ind, 'pitch'] = new_df.at[ind, 'pitch_y']
        else:
            new_df.at[ind, 'pitch'] = new_df.at[ind, 'pitch_x']
        if pd.isnull(new_df.at[ind, 'tie']):
            new_df.at[ind, 'tie'] = None
    del new_df['duration_x'], new_df['duration_y'], new_df['pitch_x'], new_df['pitch_y']
    new_df = new_df.sort_values(['measure', 'offset', 'ps'])
    return new_df

def f1(y_true, y_pred):
    """F1 score implementation to be used by keras models
    """
    from keras import backend as K
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

import keras
class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_epoch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['epochs'],
                                          metrics_log))
            self.metric_cache.clear()

def dense(hidden=200):
    """I don't know why I put it here again"""
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from piano_reduction import compute_features as cf
    nn = Sequential()
    nn.add(Dense(hidden, input_shape=[cf.feature_count()]))
    nn.add(Activation('sigmoid'))
    nn.add(Dense(1))
    nn.add(Activation('sigmoid'))
    nn.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    return nn

def lstm(hidden=200):
    """I don't know why I put it here again"""
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    from piano_reduction import compute_features as cf
    lstm = Sequential()
    lstm.add(LSTM(hidden, input_shape=(None, cf.feature_count())))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(1))
    lstm.add(Activation('sigmoid'))
    lstm.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    return lstm

def gru(hidden=200):
    """I don't know why I put it here again"""
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import GRU
    from piano_reduction import compute_features as cf
    gru = Sequential()
    gru.add(GRU(hidden, input_shape=(None, cf.feature_count())))
    gru.add(Dropout(0.2))
    gru.add(Dense(1))
    gru.add(Activation('sigmoid'))
    gru.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    return gru

def early_stop(patience=10):
    """The early stopping callbacks used by keras training
    """
    from keras.callbacks import EarlyStopping
    return EarlyStopping(monitor='val_f1', mode='max', min_delta=0, patience=patience, verbose=0, restore_best_weights=True)

def jaccard_similarity(y_train, y_pred):
    """Jaccard Similarity used as metrics
    """
    s = set()
    t = set()
    for i in range(0, len(y_train)):
        for j in range(0, 128):
            if y_train[i, j] == 1:
                t.add((i, j))
    for i in range(0, len(y_pred)):
        for j in range(0, 128):
            if y_pred[i, j] == 1:
                s.add((i, j))
    return len(s&t) * 1.0 / (len(s) + len(t) - len(s&t))

def pitch_class_jaccard_similarity(y_train, y_pred):
    """Jaccard Similarity considering only the pitch class
    """
    s = set()
    t = set()
    for i in range(0, len(y_train)):
        for j in range(0, 128):
            if y_train[i, j] == 1:
                t.add((i, j % 12))
    for i in range(0, len(y_pred)):
        for j in range(0, 128):
            if y_pred[i, j] == 1:
                s.add((i, j % 12))
    return len(s&t) * 1.0 / (len(s) + len(t) - len(s&t))
