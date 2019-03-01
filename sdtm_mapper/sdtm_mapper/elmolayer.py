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

class elmolayer(Layer):
    '''
    Sam Tomioka
    Create trainable ELMo embedding layer
    
    '''
    
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(elmolayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(elmolayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)