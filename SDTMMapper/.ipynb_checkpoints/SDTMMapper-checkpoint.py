import pandas as pd
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
from keras.utils import to_categorical, np_utils

# Initialize session
sess = tf.Session()
K.set_session(sess)

#bucket='snvn-sagemaker-1' #data bucket

#KEY='mldata/Sam/data/project/380-000/bipolar/380-201/csr/data/raw/latest/'

class SDTMMapper:
    '''SDTMMapper
    Sam Tomioka
    Sunovion Pharmaceuticals
    
    1. load_sas_from_s3
    2. sas2meta
    3. sas_metadata_to_csv
    4. drop_sys_vars
    5. add_drop
    6. encode_sdtm_target
    7. decode_sdtm_target
    
    
    
    '''
    def __init__(self, domain, isS3, bucket='', KEY='', localpath='',**kwargs):
        
        self.dataset=domain + '.sas7bdat'
        self.domain=domain
        self.bucket=bucket
        self.KEY=KEY
        self.localpath=localpath
        self.isS3=isS3
        super(SDTMMapper, self).__init__(**kwargs)
        
    def load_sas_from_s3 (self): 
        '''KEY = s3 KEY - folder where SAS dataset is stored
        bucket = s3 bucket
        '''
        s3 = boto3.resource('s3')
        if not os.path.exists('data'):
            os.makedirs('data')
        try:
            s3.Bucket(self.bucket).download_file(os.path.join(self.KEY,self.dataset),os.path.join('data',self.dataset))
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
                
    def sas2meta(self, encode):
        '''encode like latin
        localpath is a path to the folder where the dataset is stored 
        '''
        if self.isS3==True:
            self.load_sas_from_s3()  ## bring SAS to local under data folder
            sas=SAS7BDAT(os.path.join('data',self.dataset),encoding=encode)
        elif self.isS3==False:
            sas=SAS7BDAT(os.path.join(self.localpath,self.dataset),encoding=encode)
            
        varlist=list(map(lambda i: i.name.decode("utf-8") , sas.columns))
        variablelabels=list(map(lambda i: i.label.decode("utf-8") , sas.columns))
        df = pd.DataFrame(
            {'ID': list(map(lambda i: i.name.decode("utf-8").upper() , sas.columns)),
             'text': list(map(lambda i: i.label.decode("utf-8") , sas.columns))
            })
        df['text']= df["ID"] +' '+ df["text"]
        df['text'] = df['text'].str.replace('_', ' ')
        return df
    
    def sas_metadata_to_csv(self,encode, out_csv_file):
        '''saves metadata in cvs file in test_data folder
        and create ae dataframe'''
        if self.isS3==True:
            s3 = boto3.resource('s3')
            if not os.path.exists('test_data'):
                os.makedirs('test_data')
            df=self.sas2meta(encode)
            df.to_csv(os.path.join('test_data',out_csv_file),index=False)
        elif self.isS3==False:
            if not os.path.exists('test_data'):
                os.makedirs('test_data')
            df=self.sas2meta(encode)
            df.to_csv(os.path.join('test_data',out_csv_file),index=False)
        return df
    
    def drop_sys_vars(self, metafile, edc, suffix):
        '''returns df with dropping variables and df to be classified or to be trained

        INPUT:
        1. df - training data or new metadata file output from `load_sas` with 3 columns 'ID','text', 'sdtm'
        2. edc - 'rave'
        3. suffix - regular expression of variable suffix that should be set to drop

        NOTE:
        1. Set for Sunovion
        suffix='.*_RAW$|.*_INT$|.*_CV$|.*_STD$|.*_D{1,2}$|.*_M{1,2}$|.*_Y{1,4}$'
        2. drop rave system variables
        rave=[userid,projectid,project,sdvtier, 
              studyid,environmentName,subjectId,
              StudySiteId,siteid,Site,SiteNumber,
              SiteGroup,instanceId,InstanceRepeatNumber,
              folderid,Folder,FolderName,FolderSeq,TargetDays,
              DataPageId,DataPageName,PageRepeatNumber,RecordDate,
              RecordId,recordposition,RecordActive,SaveTs,MinCreated,MaxUpdated]
        '''
        df = pd.read_csv(metafile)
        if edc=='rave':
            sysvars='SDVTIER$|TARGETDAYS$|USERID$|.*PROJECTID$|.*PROJECT$|.*STUDYID$|.*ENVIRONMENTNAME$|.*SUBJECTID$|.*STUDYSITEID$|SITEID$|SITE$|SITENUMBER$|SITEGROUP$|INSTANCEID$|INSTANCEREPEATNUMBER$|FOLDERID$|FOLDER$|FOLDERNAME$|FOLDERSEQ$|TARGETDAYS$|DATAPAGEID$|DATAPAGENAME$|PAGEREPEATNUMBER$|RECORDDATE$|RECORDID$|RECORDPOSITION$|RECORDACTIVE$|SAVETS$|MINCREATED$|MAXUPDATED$'
            drop_vars_patt=suffix+'|'+sysvars

        dropped=df[df["ID"].str.contains(drop_vars_patt, case=False, regex=True)]
        dropped.reset_index(inplace=True, drop=True)
        dropped2=dropped.copy()
        dropped2['sdtm']='DROP'
        dropped2['pred']='DROP'
        df=df[~df["ID"].str.contains(drop_vars_patt, case=False, regex=True)]
        df.reset_index(inplace=True, drop=True)
        df2=df.copy()
        x=df2['text'].str.lower()
        return dropped2, x, df2
    
    def add_drop(self, preddf,dropped):
        '''concatenate two df
        use case: this is used to concatenate the prediction and dropped dataframe created initially
        '''

        results= pd.concat([preddf, dropped])
        results.reset_index(inplace=True, drop=True)
        return results
    
    def encode_sdtm_target(self,target, encodername):
        '''encode sdtm target
        example call to add encodes
        Y=encode_sdtm_target(df['sdtm'])
        '''
        le = LabelEncoder()
        le.fit(target)
        Y = le.transform(target)
        Y = np_utils.to_categorical(Y)

        if not os.path.exists('decode'):
            os.makedirs('decode')
        target.to_pickle(os.path.join('decode', encodername+'.pkl'))
        return Y

    def decode_sdtm_target(self,predictions, encodername):
        '''decode sdtm target
        create dictionary for decoding and returns decoded values
        
        df['sdtm']=decode_sdtm_target(pred)
        '''
        predictions_=np.argmax(predictions, axis = 1)
        df = pd.read_pickle(os.path.join('decode', encodername+'.pkl')) #read target USED for TRAINING
        le = LabelEncoder()
        le.fit(df)
        dictionary = dict(zip(np.array(df), le.transform(df)))
        target = [k for i in predictions_ for k, v in dictionary.items() if i==v]  
        return target
    
