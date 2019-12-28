LENGTH = 39
DATE = '20190306b'
train_list = [2, 6, 8]
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"
os.makedirs('models/%s' % DATE, exist_ok=True)

import piano_reduction.tools as pr
import piano_reduction.compute_features as cf
import piano_reduction.keras_models as km
from piano_reduction.classes import ScoreData, Reduction
import random
from sklearn.model_selection import train_test_split

from sklearn import metrics
from collections import defaultdict
import numpy as np

import keras
features = ['active_rhythm', 'bass_line', 'entrance_effect', 'highest', 'in_chord', 'lowest', 'occurrence', 
   'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling', 'duration_length']
    
def evaluate(model, testing_data, x_test, y_test, t=0.5):
    validation_data = testing_data.copy()
    validation_data = validation_data.merge_binary((model.predict(x_test) > t) * 1, skip_features=True)
    tmptrain = validation_data.to_binary('y_train')
    tmppred = validation_data.to_binary('y_pred')
    scores = []
    scores.append(pr.jaccard_similarity(tmptrain, tmppred))
    scores.append(pr.pitch_class_jaccard_similarity(tmptrain, tmppred))
    scores.append(metrics.accuracy_score(validation_data.df['y_train'], validation_data.df['y_pred']))
    scores.append(metrics.f1_score(validation_data.df['y_train'], validation_data.df['y_pred']))
    scores.append(metrics.roc_auc_score(tmptrain.flatten(), model.predict(x_test).flatten()))
    return scores

def bidirectional_gru(features):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, GRU, Concatenate, Input, Lambda
    from keras.models import Model
    input_layer = Input(shape=(39, 128 + 128 * len(features)))
    layer = Lambda(lambda x:x[:,:20], output_shape=(20,128 + 128 * len(features)))(input_layer)
    layer = GRU(200)(layer)
    layer_1 = layer
    layer = Lambda(lambda x:x[:,19:], output_shape=(20,128 + 128 * len(features)))(input_layer)
    layer = GRU(200, go_backwards=True)(layer)
    layer_2 = layer
    layer = Concatenate(axis=1)([layer_1, layer_2])
    layer = Activation('sigmoid')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(128)(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=input_layer, outputs=layer)
    model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy', pr.f1])
    return model

k = 5
display = 200
print('    %30s %20s %15s %15s %15s %15s' % ('', 'Jaccard Similarity', 'Ignore Octave', 'Accuracy', 
                                             'F1 score', 'roc_auc_score'))
avg = defaultdict(lambda : np.zeros(5))
models = {}
for i in train_list:
    models['GRU_2 %d' % i] = bidirectional_gru(features)
progress = ''

for i in train_list:
    data = ScoreData.load('score_data/cosi_%d.pkl' % i)
    
    training_data, testing_data = data.split(0.9)
    
    name = 'GRU_2 %d' % i
    model = models[name]
    length = LENGTH
    logger = pr.NBatchLogger(display=10)
    x_train, y_train = None, None
    x_test, y_test = testing_data.generate_data_5(length=length, features=features)
    data_tmp = training_data.copy()
    for key in range(-12, 13):
        data_tmp.df['ps'] = data.df['ps'] + key
        x_tmp, y_tmp = data_tmp.generate_data_5(length=length, features=features)
        if key == -12:
            x_train, y_train = x_tmp, y_tmp
        else:
            x_train, y_train = np.append(x_train, x_tmp, axis=0), np.append(y_train, y_tmp, axis=0)

    early_stop = pr.early_stop(patience=10)
    chunk = x_train.shape[0] // 1
    for j in range(0, x_train.shape[0], chunk):
        print(x_train[j : j + chunk].shape, x_test.shape, y_train[j : j + chunk].shape, y_test.shape)
        model.fit(x_train[j : j + chunk], y_train[j : j + chunk], epochs=100, batch_size=512, validation_data=(x_test, y_test), callbacks=[logger, early_stop], verbose=0)
    scores = evaluate(model, testing_data, x_test, y_test)
    avg['%s with %d' % (name, length)] += np.array(scores)
    models[name] = model
    print('%3d %30s %20.8f %15.8f %15.8f %15.8f %15.8f' % tuple([i] + ['%s with %d' % (name, length)] + scores))
    progress = '%d %d %s' % (i, 0, name)
    with open('log_slurm.txt', 'a') as f:
        f.write(progress + '\n')
        f.write('%s\n' % ('%3d %30s %20.8f %15.8f %15.8f %15.8f %15.8f' % tuple([i] + ['%s with %d' % (name, length)] + scores)))
    models[name].save('models/%s/%s.h5' % (DATE, name))
    red = Reduction('models/%s/%s.h5' % (DATE, name), ScoreData.generate_data_5, {'length' : LENGTH, 'features' : features}, 0.1)
    red.save('models/%s/%s.pkl' % (DATE, name))