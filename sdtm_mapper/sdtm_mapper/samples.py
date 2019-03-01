from sdtm_mapper import SDTMModels as sdtm
import pandas as pd
import os
from keras.models import Model, load_model, Sequential

def load_sample_model(m):
    models = sdtm.SDTMModels('ae', 34)
    model = models.build_model(m, False)

    this_dir, this_filename = os.path.split(__file__)

    if m==1:
        model.load_weights(os.path.join(this_dir,'models','Elmo+sfnn+ae+Model1.h5'))
    elif m==2:
        model.load_weights(os.path.join(this_dir,'models','Elmo+fnn+ae+Model2.h5'))
    elif m==3:
        model.load_weights(os.path.join(this_dir,'models','Elmo+fnn+ae+Model3.h5'))
    return model

def load_sample_study(domain):
    ''' load sample data'''

    this_dir, this_filename = os.path.split(__file__)
    if domain=='ae':
        df=pd.read_csv(os.path.join(this_dir, 'data', 'ae_test_study.csv'))
    return df

def load_sample_decoder():
    ''' load decoder'''

    this_dir, this_filename = os.path.split(__file__)
    df = pd.read_pickle(os.path.join(this_dir, 'decode', 'train_encode.pkl'))
    if not os.path.exists('decode'):
            os.makedirs('decode')
    df.to_pickle(os.path.join('decode', 'sample_decoder.pkl'))
