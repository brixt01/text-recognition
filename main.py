import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels

def import_data(directory):
    dfArray = read_data(directory)
    fix_time(dfArray)
    fix_centering(dfArray)
    return dfArray

def read_data(directory):
    dfArray = []
    for filename in os.listdir(directory):
        df = pd.read_csv(f'{directory}/{filename}', names=["x", "y", "time"])
        df['char'] = filename[0]
        dfArray.append(df.drop_duplicates('time'))
    return dfArray

def fix_time(dfArray):
    for df in dfArray:
        df['time'] = df['time'] - df['time'][0]

def fix_centering(dfArray):
    for df in dfArray:
        df['x'] = df['x'] - df['x'].mean()
        df['y'] = df['y'] - df['y'].mean()

def get_features(inputdf):
    outputdf = pd.DataFrame()

    aratio = []
    for letter in inputdf:
        xrange = letter['x'].max() - letter['x'].min()
        yrange = letter['y'].max() - letter['y'].min()
        aratio.append(yrange / xrange)
    outputdf['aratio'] = aratio

    duration = []
    for letter in inputdf:
        duration.append(letter['time'].iloc[-1])
    outputdf['dur'] = duration

    medx = []
    medy = []
    for letter in inputdf:
        medx.append(letter['x'].median())
        medy.append(letter['y'].median())
    outputdf['medx'] = medx
    outputdf['medy'] = medy

    devx = []
    devy = []
    for letter in inputdf:
        devx.append(letter['x'].std())
        devy.append(letter['y'].std())
    outputdf['devx'] = devx
    outputdf['devy'] = devy

    maxx = []
    minx = []
    maxy = []
    miny = []
    for letter in inputdf:
        maxx.append(len(argrelextrema(letter['x'].values, np.greater_equal)[0]))
        minx.append(len(argrelextrema(letter['x'].values, np.less_equal)[0]))
        maxy.append(len(argrelextrema(letter['y'].values, np.greater_equal)[0]))
        miny.append(len(argrelextrema(letter['y'].values, np.less_equal)[0]))
    outputdf['maxx'] = maxx
    outputdf['minx'] = minx
    outputdf['maxy'] = maxy
    outputdf['miny'] = miny

    corr1 = []
    for letter in inputdf:
        corr1.append(letter['x'][:len(letter['x'].dropna())//2].corr(letter['y'].dropna()[:len(letter['y'].dropna())//2]))
    outputdf['corr1'] = corr1

    corr2 = []
    for letter in inputdf:
        corr2.append(letter['x'][len(letter['x'].dropna())//2:].corr(letter['y'].dropna()[len(letter['y'].dropna())//2:]))
    outputdf['corr2'] = corr2

    char = []
    for letter in inputdf:
        char.append(letter['char'].iloc[0])
    outputdf['char'] = char

    return outputdf

def train_model(df):
    xtrain = df.iloc[:, :-1]
    ytrain = df.iloc[:, -1]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xtrain, ytrain.values.ravel())

    return knn

def predict(model, df):
    xtest = df.iloc[:, :-1]
    ytest = df.iloc[:, -1]
    ypredict = pd.DataFrame(model.predict(xtest), columns=['Char']).set_index(ytest.index)
    return ytest, ypredict

def evaluate(ytest, ypredict):
    evaluation = pd.concat([ytest, ypredict], axis=1)
    evaluation.columns = ['Real', 'Prediction']
    evaluation['Validity'] = (evaluation['Real'] == evaluation['Prediction'])
    return evaluation

def plot_df(df):
    plt.plot(df['x'], df['y'])
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)

def pretty_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

if (__name__ == '__main__'):
    inputdf = import_data(os.path.dirname(os.path.abspath(''))+"/current/data")
    outputdf = get_features(inputdf)
    outputdf = outputdf.dropna()

    pretty_print(outputdf)

    sample = outputdf.sample(frac=1).reset_index(drop=True)
    train = sample.iloc[:(len(sample)//10)*8, :]
    test = sample.iloc[(len(sample)//10)*8:, :]
    train = sample

    model = train_model(train)
    yreal, ypredict = predict(model, test)
    evaluation = evaluate(yreal, ypredict)
    print(f"{round(sum(evaluation.iloc[:, -1])/len(evaluation.index)*100, 2)}%")

    confusionMatrix = confusion_matrix(y_true=evaluation["Real"], y_pred=evaluation["Prediction"])
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusionMatrix, display_labels=unique_labels(evaluation["Real"], evaluation["Prediction"]))
    confusionMatrixDisplay.plot()
