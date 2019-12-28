from piano_reduction import tools as pr, features as ft, compute_features as cf
import os, pickle
import numpy as np
class Reduction:
    def __init__(self, model_path, generate_data, params, threshold=0.5, model=None, save=True):
        """Instantiate a reduction

        Keyword arguments:
        model_path -- the filename of the keras model
        generate_data -- the METHOD used to convert the ScoreData to numpy array usable by keras models
        params -- a python dictionary that stores the parameters for the generate_data method
        threshold -- the default threshold for the reduction to keep the notes
        model -- If it is not None, it (the newly created keras model) will be used by this reduction.
                 Otherwise, the model in model_path is loaded and used by this reduction.
        save -- whether the newly created keras model should be saved into model_path initially
        """
        self.model_path = model_path
        self.generate_data = generate_data
        self.params = params
        self.threshold = threshold
        self.hash = (-1, -1)
        from keras.models import load_model
        if model is not None:
            self.model = model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if save:
                self.model.save(model_path)
        else:
            self.model = load_model(self.model_path, custom_objects={'f1': pr.f1})

    def train(self, training_data, testing_data, epochs=200, batch_size=512, display=-1, early_stop=-1):
        """Train the Keras model of the reduction by providing the data

        Keyword arguments:
        training_data -- the ScoreData used to train the model
        testing_data -- the ScoreData used to prevent overfitting of the model
        epochs -- the number of epochs of the training
        batch_size -- the batch size of the training
        display -- if this is greater than 0, the training result will be displayed every this amount of epochs
        early_stop -- if this is greater than 0, the training will stop when there are no improvements after this amount of consective epochs
        """
        callbacks = []
        if display > 0:
            callbacks.append(pr.NBatchLogger(display=display))
        if early_stop > 0:
            callbacks.append(pr.early_stop(patience=early_stop))

        data_hash = (hash(training_data), hash(testing_data))
        filename = self.model_path[:-3] + '.tmp'
        loaded = False
        if data_hash == self.hash and os.path.isfile(filename):
            try:
                with open(filename, 'rb') as f:
                    tmp = pickle.load(f)
                x_train, y_train = tmp['x_train'], tmp['y_train']
                x_test, y_test = tmp['x_test'], tmp['y_test']
                loaded = True
                print('loaded')
            except:
                loaded = False

        if not loaded:
            x_train, y_train = None, None
            x_test, y_test = self.generate_data(testing_data, **self.params)
            data_tmp = training_data.copy()
            for key in range(-12, 13):
                data_tmp.df['ps'] = training_data.df['ps'] + key
                x_tmp, y_tmp = self.generate_data(data_tmp, **self.params)
                if key == -12:
                    x_train, y_train = x_tmp, y_tmp
                else:
                    x_train, y_train = np.append(x_train, x_tmp, axis=0), np.append(y_train, y_tmp, axis=0)

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            tmp = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
            self.hash = data_hash
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(tmp, f)
            except:
                self.hash = (-1, -1)

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(x_test, y_test), callbacks=callbacks, verbose=0)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

    def predict(self, data, threshold=None, name='y_pred'):
        """Generate the piano reduction

        Keyword arguments:
        data -- ScoreData to be Reduced
        threshold -- the threshold for keeping the notes
        name -- the Dataframe column name of the reduction results
        """
        if not threshold:
            threshold = self.threshold
        params = {**self.params}
        new_data = data.copy()
        x_test, _ = self.generate_data(new_data, **self.params)
        new_data = new_data.merge_binary((self.model.predict(x_test) > threshold) * 1, skip_features=True, name=name)
        return new_data

    def predict_2(self, data, threshold=None, name='y_pred'):
        """(Seems useless now)Generate the piano reduction by standardizing the musical piece to C major

        Keyword arguments:
        data -- ScoreData to be Reduced
        threshold -- the threshold for keeping the notes
        name -- the Dataframe column name of the reduction results
        """
        if not threshold:
            threshold = self.threshold
        params = {**self.params}
        new_data = data.transform()
        x_test, _ = self.generate_data(new_data, **self.params)

        new_data = new_data.merge_binary((self.model.predict(x_test) > threshold) * 1, skip_features=True, name=name)
        new_data = new_data.revert_transform()
        return new_data

    def predict_with_colors(self, data, threshold=None, name='y_pred'):
        """(Seems useless now)Generate the piano reduction with colors denoting the likelihood

        Keyword arguments:
        data -- ScoreData to be Reduced
        threshold -- the threshold for keeping the notes
        name -- the Dataframe column name of the reduction results
        """
        if not threshold:
            threshold = self.threshold
        params = {**self.params}
        new_data = data.copy()
        x_test, _ = self.generate_data(new_data, **self.params)
        new_data = new_data.merge_binary_threshold(self.model.predict(x_test), threshold, skip_features=True, name=name)
        def find_color(x):
            r = [[255, 0, 0, 0],
                     [255, 255, 0, 0.2],
                     [0, 255, 0, 0.5],
                     [0, 0, 255, 1]]
            for i in range(1, 4):
                if x < r[i][3]:
                    return [(r[i - 1][j] * (r[i][3] - x) + r[i][j] * (x - r[i - 1][3])) / (r[i][3] - r[i - 1][3]) for j in range(3)]
        def to_color(x):
            tmp = find_color(x)
            tmp = int(tmp[2]) + int(tmp[1]) * 256 + int(tmp[0]) * 256 * 256
            return ('#%6s' % ("{0:x}".format(tmp))).replace(' ', '0') if x > 0 else 0
        new_data.df['keep_score'] = new_data.df['keep_score'].apply(to_color)
        return new_data

    def save(self, filename):
        """Save the reduction

        Keyword arguments:
        filename -- the desired filename for the reduction
        """
        tmp = self.model
        self.model = None
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.model = tmp

    @classmethod
    def load(cls, filename):
        """Load the reduction from a file

        Keyword arguments:
        filename -- the filename of the saved reduction
        """
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
        from keras.models import load_model
        tmp.model = load_model(tmp.model_path, custom_objects={'f1': pr.f1})
        return tmp

class ReductionStack():
    """This was susposed to be used as model ensembling, but now it seems to fail.
    """
    def __init__(self, filenames, final_model):
        self.models = []
        self.model_paths = filenames
        for path in self.model_paths:
            self.models.append(Reduction.load(path))
        self.final_model = final_model

    def get_x(self, data):
        import numpy as np
        x = []
        for model in self.models:
            x_test, _ = model.generate_data(data, **model.params)
            x.append(np.log(np.ravel(model.model.predict(x_test))))
        tmp = data.get_moments()
        interval = []
        pitch = []
        for i in tmp:
            for j in range(128):
                interval.append(i)
                pitch.append(j / 128.0)
        x.append(interval)
        x.append(pitch)
        x = np.swapaxes(np.array(x), 0, 1)
        return x

    def train_small(self, training_data, testing_data, epochs=200, batch_size=512, display=-1, early_stop=-1):
        if epochs <= 0:
            return
        i = 0
        for model in self.models:
            print('Training %s' % self.model_paths[i])
            model.train(training_data, testing_data, epochs, batch_size, display, early_stop)
            model.save(self.model_paths[i])
            i += 1

    def train_final(self, data):
        print('Training Final')
        import numpy as np
        x_train = self.get_x(data)
        y_train = np.ravel(data.to_binary(col='y_train'))
        self.final_model.fit(x_train, y_train)

    def train_all(self, training_data, testing_data, training_ratio=0.9,
                  epochs=200, batch_size=512, display=-1, early_stop=-1):
        training_data_small, training_data_final = training_data.split(training_ratio)
        self.train_small(training_data_small, testing_data, epochs, batch_size, display, early_stop)
        self.train_final(training_data_final)

    def predict(self, data, name='y_pred', threshold=1):
        import numpy as np
        x_test = self.get_x(data)
        y_pred = np.reshape(self.final_model.predict(x_test), (-1, 128))
        new_data = data.copy()
        new_data = new_data.merge_binary((y_pred >= threshold) * 1, skip_features=True, name=name)
        return new_data

    def save(self, filename):
        tmp = self.models
        self.models = None
        import pickle
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self.models = tmp

    @classmethod
    def load(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
        import piano_reduction.tools as pr
        tmp.models = []
        for path in tmp.model_paths:
            tmp.models.append(Reduction.load(path))
        return tmp
