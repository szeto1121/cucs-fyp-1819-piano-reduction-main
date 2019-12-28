LENGTH = 39
DATE = '20190415c'
train_list = [2, 6]
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


#for i in train_list:
#    dfs = pr.load_merged_df(7, use_cache=False)
#    score = ScoreData(dfs)
#    score.save('score_data/%d.pkl' % i)
#    print(i)
#    with open('log_slurm.txt', 'a') as f:
#        f.write('Created %s\n' % ('score_data/%d.pkl' % i))

features = ['active_rhythm', 'bass_line', 'entrance_effect', 'highest', 'in_chord', 'lowest', 'occurrence', 
   'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling', 'duration_length']
k = 5

for i in train_list:
    if i == 7:
        data = ScoreData.load('score_data/%d.pkl' % i)
    else:
        data = ScoreData.load('score_data/cosi_%d.pkl' % i)
    for j in range(5):
        training_data, testing_data = data.split_in(j * 0.2, j * 0.2 + 0.2)
        
        features = ['active_rhythm', 'bass_line', 'entrance_effect', 'highest', 'in_chord', 
                    'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 
                    'sustained_rhythm', 'vertical_doubling', 'duration_length']
        
        name = 'GRU_2 -new %d %d' % (i, j)
        red = Reduction('models/%s/%s.h5' % (DATE, name), ScoreData.generate_data_5, 
                        {'length' : LENGTH, 'features' : features}, 0.1,
                        km.bidirectional_gru(features, (LENGTH + 1) // 2))
        red.train(training_data, testing_data, 200, 512, 10, 20)
        red.save('models/%s/%s.pkl' % (DATE, name))
        
        name = 'GRU_1 -new %d %d' % (i, j)
        red = Reduction('models/%s/%s.h5' % (DATE, name), ScoreData.generate_data_4, 
                        {'length' : 20, 'features' : features}, 0.1,
                        km.gru_with_features(features))
        red.train(training_data, testing_data, 300, 512, 10, 20)
        red.save('models/%s/%s.pkl' % (DATE, name))
        
        features = ['active_rhythm', 'bass_line', 'entrance_effect', 'highest', 
                    'lowest', 'occurrence', 'onset_after_rest', 'rhythm_variety', 'strong_beats', 
                    'sustained_rhythm', 'vertical_doubling', 'duration_length']
        
        name = 'GRU_2 -old %d %d' % (i, j)
        red = Reduction('models/%s/%s.h5' % (DATE, name), ScoreData.generate_data_5, 
                        {'length' : LENGTH, 'features' : features}, 0.1,
                        km.bidirectional_gru(features, (LENGTH + 1) // 2))
        red.train(training_data, testing_data, 200, 512, 10, 20)
        red.save('models/%s/%s.pkl' % (DATE, name))
        
        name = 'GRU_1 -old %d %d' % (i, j)
        red = Reduction('models/%s/%s.h5' % (DATE, name), ScoreData.generate_data_4, 
                        {'length' : 20, 'features' : features}, 0.1,
                        km.gru_with_features(features))
        red.train(training_data, testing_data, 300, 512, 10, 20)
        red.save('models/%s/%s.pkl' % (DATE, name))