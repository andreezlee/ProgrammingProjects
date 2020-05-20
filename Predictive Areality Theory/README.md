# Predictive Areality Theory Basic Prototype

This details a prototype implementation for Predictive Areality Theory using small PyTorch feed-forward neural networks trained on data from the WALS database. This was developed in Python 3.7.6 and PyTorch version 1.5.0. The imported files from the database were taken from [here](https://github.com/cldf-datasets/wals) and are included in the ```database_files``` directory.

## Creation of Models

### Parsing the Data

Run using ```python3 parse.py```

Given the csv files, data structures containing the necessary information were created using the ```parse.py``` file, and are stored in the ```picklejar``` directory. Please refer to the proper file comments for code specifics.

### Training Models

Run using ```python3 ffnn.py```

The PyTorch implementation of the neural models is the simple class FFNN detailed in ```ffnn.py```, and uses the ```.pickle``` files to form training and validation sets. The approach used for this project created separate neural networks for each linguistic feature in the database, attempting to train pattern recognition of the relationships between language features and the approximate locations of the first native speakers.

For each network, the example set was formed with each language with the documented linguistic feature as a data point, in the format of location (latitude-longitude) and centroid of language family (average location of languages in the same language family, reflexive for isolates) mapped to the value output by taking the ```argmax``` of the ```softmax``` of the linguistic feature's possible values.

The training set would be a randomized 90% of the example set, with the validation set being the other 10%. Features were deemed unusable if there were less than 300 data points or if training the neural network led to accuracy reaching 0. Accuracy for determining each neural network's performance was an average of the training and validation accuracies on the final training epoch. Fine-tuning on metaparameters produced these results, with the models and their data stored as ```.pickle``` files as well: (Note: using the ```pickle``` library to store PyTorch structures might soon be deprecated.)

* Highest performing hidden layer size: 16 nodes
* Highest performing number of epochs: 6 epochs
* Usable features: 68 features
* Average model accuracy measure: 49.25%
* Highest model accuracy measure: 99.58%

Your mileage may vary. Please refer to the proper file comments for code specifics.

### Using the Models

Run using ```python3 analysis.py```

With the models trained and stored, the file ```analysis.py``` identifies the top 50 most accurate networks and uses them on a random distribution of locations across a slew of wide areas. The printed output shows the distribution of predictions for each of the 50 features. Please refer to the proper file comments for code specifics.
