import piano_reduction.tools as pr
import piano_reduction.compute_features as cf
import piano_reduction.keras_models as km
from piano_reduction.classes import ScoreData, Reduction, ReductionStack
from sklearn import metrics
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
#data = ScoreData.load('score_data/cosi_%d.pkl' % 8)
#k = 0
#training_data, testing_data = data.split_in(k * 0.2, k * 0.2 + 0.2)
features = ['active_rhythm', 'bass_line', 'entrance_effect', 'highest', 'lowest', 'occurrence', 
            'onset_after_rest', 'rhythm_variety', 'strong_beats', 'sustained_rhythm', 'vertical_doubling', 
            'duration_length']

reductions = []
for k in [1, 2, 3, 4]:
    dir_path = 'models/20190410/stack_1_%d/' % k
    filenames = []
    for LENGTH in [1, 2, 3, 4, 8, 12, 16, 20, 24]:
        filename = dir_path + ('%d' % LENGTH)
        Reduction(filename + '.h5', ScoreData.generate_data_5, 
                  {'length' : LENGTH * 2 - 1, 'features' : features}, 0.1,
                  km.bidirectional_gru(features=features, length=LENGTH, hidden=256)).save(filename + '.pkl')
        filenames.append(filename + '.pkl')
    data = ScoreData.load('score_data/cosi_%d.pkl' % 8)
    training_data, testing_data = data.split_in(k * 0.2, k * 0.2 + 0.2)    
    
    stack = ReductionStack(filenames, RandomForestClassifier(class_weight='balanced'))
    stack.train_all(training_data, testing_data, 0.9, 200, 512, 10, 20)
    stack.save(dir_path + 'stack.pkl')
    