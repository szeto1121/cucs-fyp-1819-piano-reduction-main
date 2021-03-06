# Piano Reduction 2018/19

## Setup

Python 3.6 was used in this project. Both Python 3.5 and 3.7 had their own problems and may be troublesome here.

1.  (Recommended) Use a virtualenv so that your python will not be ruined:

    ```sh
    python3 -m venv venv

    # Every time before working on the shell:
    source venv/bin/activate
    ```

2.  Install dependencies:

    ```sh
    pip3 install -r requirements.txt
    ```

3.  Install Musescore

## Development

Jupyter Lab was used so that experiments could be done easily
```sh
jupyter lab .
```
To prevent the confusion and frustration caused by importing Python packages, put your code in the main directory (here) so that you can import the "piano reduction package" easily.

## Usage

The directory `piano_reduction` contains the source codes for the piano reduction package. `demo_2.ipynb` and `20190415c.py` shows some examples of using the package. See `piano_reduction/README.md` for more details of using the package.

The directory 'report' contains the project report and presentation slides for this project.

## Contents

The usable contents are listed as follows:

| Folder Name       | Contents                                                                                         |
|-------------------|--------------------------------------------------------------------------------------------------|
| cosi              | Original and a number of reductions of Mozart's Cosi fan tutte                                   |
| demo           | Some example files used by demo_2.ipynb                                   |
| input             | The original scores collected in previous years                                                  |
| input_with_chords | The original scores collected in previous years with chords generated by some plugins            |
| output            | The reduced scores (playable by two hands) collected in previous years                           |
| piano_reduction   | The python source codes for the "piano reduction package"                                        |
| score_data        | The combined data used for performing piano reduction (can be generated from the musicXML files) |

The remaining files in the main directory can (probably) be used as examples to understand the spaghetti codes:

| Files   | Contents                                                                                                                                                                         |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *.ipynb | Some Jupyter notebooks that were used to try different things and generate results/graphs. (The older the file is, the higher the chance of having deprecated code is)           |
| *.py    | The codes used to train and generate Keras models with different training data/settings. 20190415c.py shows a clearer example of how to train the models, and the rest are difficult to read. |
| *.cpp   | The algorithm used by the post-processor of the piano reduction. Read the code for more details of input and output format.                                                      |

## Miscellaneous Traps/Advices (TBC)

1.  If you cannot show your music21 object using musescore, search for something like music21.environment.set('musicxmlPath', '/usr/bin/musescore')

2.  If your tensorflow/tensorflow-gpu doesn't run correctly, try uninstall it and install the other one

3.  It may take decades to train the models on your computer, so we applied for and used the GPU provided by department instead

4.  There are still some problems regarding of displaying the score, but they should be unimportant
