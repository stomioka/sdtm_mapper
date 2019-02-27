import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import re
import numpy as np
from sas7bdat import SAS7BDAT
import boto3
import botocore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support, classification_report
from keras import backend as K
import keras.layers as layers
from keras.layers import Input, Dense, Dropout, Embedding,  Flatten
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.engine import Layer
from SDTMMapper import elmolayer as em

class SDTMModels():
    def __init__(self,domain,shape, **kwargs):
        self.domain=domain
        self.shape=shape
        
    def build_model(self, m, summary):
        if m==1:
            rawmeta = layers.Input(shape=(1,), dtype="string")
            emb = em.elmolayer()(rawmeta)
            d1 = layers.Dense(256, activation='relu')(emb)
            yhat = layers.Dense(self.shape, activation='softmax', name = "output_node")(d1)
            model = Model(inputs=[rawmeta], outputs=yhat)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            if summary == True:
                model.summary()
        elif m==2:
            rawmeta = layers.Input(shape=(1,), dtype="string")
            emb = em.elmolayer()(rawmeta)
            d1 = layers.Dense(512, activation='relu')(emb)
            d1 = layers.Dropout(0.2)(d1)
            d2 = layers.Dense(256, activation='relu')(d1)
            d2 = layers.Dropout(0.1)(d2)
            d3 = layers.Dense(128, activation='relu')(d2)
            yhat = layers.Dense(self.shape, activation='softmax', name = "output_node")(d3)
            model = Model(inputs=[rawmeta], outputs=yhat)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            if summary == True:
                model.summary()
        elif m==3:
            rawmeta = layers.Input(shape=(1,), dtype="string")
            emb = em.elmolayer()(rawmeta)
            d1 = layers.Dense(512, activation='relu')(emb)
            d2 = layers.Dense(256, activation='relu')(d1)
            d3 = layers.Dense(128, activation='relu')(d2)
            yhat = layers.Dense(self.shape, activation='softmax')(d3)
            model = Model(inputs=[rawmeta], outputs=yhat)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            if summary == True:
                model.summary()

        return model