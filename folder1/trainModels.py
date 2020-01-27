#IMPORT REQUIRED LIBRARIES

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
import python_speech_features as MFCC
from sklearn import preprocessing
import warnings
import pandas as pd
from xgboost import XGBClassifier
import pickle

#GENERAL WARNING FILTER IGNORES MATCHING WARNINGS
warnings.filterwarnings("ignore")


source = "/Users/jashrathod/PycharmProjects/On-the-go-Translator/Voice-Based-Gender-Detection/pygender/train_data/youtube/"

#DESTINATION LOCATION TO STORE MODULES
dest = "/Users/jashrathod/PycharmProjects/On-the-go-Translator/Voice-Based-Gender-Detection/Modules/"

#TO GET MEL FREEQUENCY CEPSTRAL COEFFICIENTS OF THE AUDIO
def get_MFCC(sr,audio):
    
    #MFCC IS SUBSTITUE FOR python_speech_features CLASS. THIS CREATES A SHAPE FOR THE AUDIO 
    features = MFCC.mfcc( audio , sr , 0.025 , 0.01 , 13 , appendEnergy = False )
    
    #TO STANDARDIZE ANY DATASET ALONG ANY AXIS
    features = preprocessing.scale(features)

    #RETURNS STANDARDIZED DATA
    return features

def gender(gen):
    #SOURCES OF AUDIO TRAINING FILE
    #FOR DIFFERENT MODULES CLASSIFY EACH INTO SEPERATE LOCATION AND PROVIDE THE DATA ACCORDINGLY
    #HERE ONLY TWO MODULES SUCH AS MALE AND FEMALE ARE USED
    #RUN THIS PROGRAM WITH SOURCE DENOTING MALE AUDIO TRAININGS ONCE AND FEMALE AUDIO TRAININGS ONCE
    #THIS CREATES TWO DIFFERENT MODULES IN THE GIVEN DESTINATION
    #ONE IS MALE VOICE MODULE AND OTHER IS FEMALE VOICE MODULE

    #COLLECT ALL THE FILES WHICH ARE TRAINING MODULE. HERE THE AUDIO FILE IS EXPECTED TO BE IN .wav FORMAT
    # files IS A LIST OF ONLY .wav FILES FROM THE PROVIDED SOURCE LOCATION
    files = [os.path.join(source+gen,f) for f in os.listdir(source+gen) if f.endswith('.wav')]

    #TO STORE RETURNED DATA FROM get_MFCC() FUNCTION
    features = np.asarray(())

    #TO READ ALL TRAIN FILES, RUN A FOR LOOP ON THE COLLECTION
    for f in files:
        
        #sr DENOTES SAMPLING RATE AND audio DENOTES THE SOURCE SIGNAL
        sr,audio = read(f)
        
        #vector STORES THE RETRUNED VALUE FROM THE get_MFCC() FUNCTION
        vector  = get_MFCC(sr,audio)

        #FOR THE FIRST AUDIO SIGNAL features REMAINS EMPTY. SO ASSIGN THE RETURNED VALUE TO features
        if features.size == 0:
            features = vector
        else:
            #FOR THE FOLLOWING ITERATIONS DEVELOP features AS A VECTOR STACK 
            features = np.vstack((features, vector)) 

    return features

    # features_xgb = pd.DataFrame(features)
    # features_xgb['gender'] = 0

def gmm_model(gen):

    features = gender(gen)

    #ONCE ALL TRAINING FILES ARE STACKED, CREATE A GAUSSIAN MIXTURE MODEL
    gmm = GaussianMixture(n_components = 8, covariance_type='diag', max_iter = 200 , n_init = 3 )

    #fit() ESTIMATES THE MODEL PARAMETERS USING THE EM ALGORITHM
    gmm.fit(features)

    #picklefile PROVIDES THE DESTINATION TO STORE THE .gmm FILE
    # pickleFile = f.split("/")[-2].split(".wav")[0]+".gmm"
    pickleFile = gen + ".gmm"

    #MODEL SAVED IN FILE BY dump() FUNCTION
    cPickle.dump(gmm, open(dest + pickleFile,'wb'))

    print ("Modeling Completed for Gender : " + pickleFile)

def xgb_model():
    features = gender('female')
    features_female = pd.DataFrame(features)
    features_female['gender'] = 1
    
    features = gender('male')
    features_male = pd.DataFrame(features)
    features_male['gender'] = 0

    features = features_female.append(features_male, ignore_index=True)

    y = features['gender']
    X = features.drop('gender', 1)
    xgb = XGBClassifier()
    xgb.fit(X, y)
    pickle.dump(xgb, open("xgb.pickle.dat", "wb"))



gmm_model('male')
gmm_model('female')

# xgb_model()

# Dataset: http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/
