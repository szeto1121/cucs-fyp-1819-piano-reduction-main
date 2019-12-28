from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import piano_reduction.tools as pr

"""These are some canned keras model used in this year"""

def dense():
    model2 = Sequential()
    model2.add(Dense(200, input_shape=(128, )))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2

def dense_with_features(features = ['active_rhythm', 'bass_line', 'entrance_effect', 'harmony', 'highest', 'in_chord', 'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling']):
    model2 = Sequential()
    model2.add(Dense(200, input_shape=(128 + 128 * len(features), )))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2

def lstm():
    model2 = Sequential()
    model2.add(LSTM(200, input_shape=(None, 128)))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2

def lstm_with_features(features = ['active_rhythm', 'bass_line', 'entrance_effect', 'harmony', 'highest', 'in_chord', 'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling']):
    model2 = Sequential()
    model2.add(LSTM(200, input_shape=(None, 128 + 128 * len(features))))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2
#model2.summary()

def gru():
    model2 = Sequential()
    model2.add(GRU(200, input_shape=(None, 128)))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2
#model3.summary()

def gru_with_features(features = ['active_rhythm', 'bass_line', 'entrance_effect', 'harmony', 'highest', 'in_chord', 'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling'], hidden=200):
    model2 = Sequential()
    model2.add(GRU(hidden, input_shape=(None, 128 + 128 * len(features))))
    model2.add(Activation('sigmoid'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128))
    model2.add(Activation('sigmoid'))
    model2.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', pr.f1])
    return model2

def bidirectional_gru(features = ['active_rhythm', 'bass_line', 'entrance_effect', 'harmony', 'highest', 'in_chord', 'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling'], length=20, hidden=200):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, GRU, Concatenate, Input, Lambda
    from keras.models import Model
    input_layer = Input(shape=(length * 2 - 1, 128 + 128 * len(features)))
    layer = Lambda(lambda x:x[:,:x.shape[1] // 2 + 1], output_shape=(length, 128 + 128 * len(features)))(input_layer)
    layer = GRU(hidden)(layer)
    layer_1 = layer
    layer = Lambda(lambda x:x[:,x.shape[1] // 2:], output_shape=(length, 128 + 128 * len(features)))(input_layer)
    layer = GRU(hidden, go_backwards=True)(layer)
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
