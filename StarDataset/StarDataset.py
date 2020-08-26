# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:52:26 2020

@author: Alessio Saladino
"""

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense,Flatten
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

labels = ["Brown Dwarf","Red Dwarf","White Dwarf","Main Sequence","Supergiant","Hypergiant"]

#Input:classificatore, array feature input, array feature target, nome del modello usato 
def print_predictions(classifier,X_test,y_test,model_name):
    predictions = classifier.predict(X_test)
    #labels = label_encoder.classes_
    sn.heatmap(confusion_matrix(y_test,predictions),annot=True,cmap='Blues', fmt='g',xticklabels=labels,yticklabels=labels).set_title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    print(model_name,"score: ",classifier.score(X_test,y_test))

def print_predictionsNN(classifier,X_test,y_test,model_name):
    predictions = np.argmax(classifier.predict(X_test),axis=1)
    #labels = label_encoder.classes_
    sn.heatmap(confusion_matrix(y_test,predictions),annot=True,cmap='Blues', fmt='g',xticklabels=labels,yticklabels=labels).set_title(model_name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    print(model_name,"score: ",classifier.evaluate(X_test,y_test,verbose=0)[1])

def create_network(in_shape):
    model = Sequential()
    #model.add(Flatten(input_shape=in_shape))
    #Specifico il numero di neuroni e la funzione di attivazione
    model.add(Dense(16,input_dim=6,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(6,activation="softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

#Definizuibe degli iperparametri
model_params ={
    'RandomForest':{
        'model': RandomForestClassifier(),
        'params':{
            'n_estimators':[8,16,32],
            'criterion':['gini','entropy']
            }
        },
    'DecisionTree':{
        'model': DecisionTreeClassifier(),
        'params':{
             'splitter':['best','random'],
             'criterion':['gini','entropy']
            }
        }
    }


#Caricamento del dataset
dataframe = pd.read_csv("6_class_csv.csv",sep = ",")


#Codifica delle colonne che contengono valori non numerici
label_encoder = LabelEncoder()
for column in dataframe.columns:
    if(dataframe[column][0].__class__==str):
        dataframe[column] = dataframe[column].replace(np.nan, " ",regex=True)
        dataframe[column] = label_encoder.fit_transform(dataframe[column])

print(dataframe)
#Split del dataset in train e test set
X = dataframe.drop(["Star type"],axis="columns")
y = dataframe["Star type"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Creazione dei modelli da utilizzare
model_scores =[]
for model_name,model_parameter in model_params.items():
    classifier = GridSearchCV(model_parameter['model'],model_parameter['params'],cv=5,return_train_score=False,verbose=False)
    classifier.fit(X_train,y_train)
    model_scores.append({
        'Model': model_name,
        'Best_score': classifier.best_score_,
        'Best_params': classifier.best_params_,
        })
    if model_name=='RandomForest':
        forestClassifier = classifier
        print('forest')
    if model_name=='DecisionTree':
        treeClassifier = classifier
        print('tree')
    
bayesClassifier = GaussianNB()
neuralNetworkClassifier = create_network(X_train.shape[1])

#Normalizzazione dei valori delle feature di input fra 0 e 1
min_max_scaler = MinMaxScaler()
X_trainNorm = min_max_scaler.fit_transform(X_train)
X_testNorm = min_max_scaler.fit_transform(X_test)

#Addestramento dei classificatori
bayesClassifier.fit(X_trainNorm,y_train)
neuralNetworkClassifier.fit(X_trainNorm,y_train,epochs=10,batch_size=4)


#Calcolo delle predizioni
print_predictions(treeClassifier,X_test,y_test,"Decision Tree")
print_predictions(forestClassifier,X_test,y_test,"Random Forest")
print_predictions(bayesClassifier,X_testNorm,y_test,"Naive Bayes")
print_predictionsNN(neuralNetworkClassifier,X_testNorm,y_test,"Neural Network")