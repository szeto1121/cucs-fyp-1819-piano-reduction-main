# Piano Reduction Package

The package provides methods to perform piano reduction on a musicXML file using Keras models.

The following briefly introduce the files. Read the code for more explanations.

## Classes

### score_data

The following conversions can be done on the files:

```
MusicXML file(s) <=> Music21 Object <=> Pandas Dataframe <=> score_data object
```

Works can then be done on the `score_data` object to perform transformations on the musical pieces.

### reduction

`reduction` objects are used throughout the machine learning component. Tensorflow models can be trained and stored, and the object provides methods to automate the process of putting the `score_data` object into the models and obtaining the reduction.

### postprocessor

`postprocessor` provides a way to transform the `score_data` using other programs. Since python could be slow in finding a playable arrangement, C++ programs were used to obtain the most playable score and return it to the `postprocessor`.

## Tools

A bunch of functions are implemented in `tools` to facilitate the conversions between musicXML files and Pandas Dataframe. Some miscellaneous functions are also put here.

## Features

The musical features implemented in `features.py` can be used to provide the values of features of all notes in the Dataframe. (`compute_features.py` seems almost completely meaningless now)

## Keras Models

Different kinds of Keras models are put in `keras_models` so that less lines of code is needed to build them.
