# text-recognition
A written letter identification program, using Python's SciKit-Learn module

## get_data.py

This program allows a user to draw each letter a number of times, collecting data for the model to be trained from. 

## main.py

This program takes the training data, generates a range of features, and is then tested to see how well it can identify the letters. 

## Evaluation

The model is able to predict the correct letter first time around 75% of the time. The confusion matrix below demonstrates which letters the model struggles with the most.

![confusion-matrix](https://github.com/brixt01/text-recognition/assets/109489475/2b116219-4a76-4d30-8bdc-9555dcf1b537)
