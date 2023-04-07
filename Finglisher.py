# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:52:45 2023

@author: Navid Basseri
"""

import os
import pickle
import numpy as np
import pandas as pd
import locale


from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


loc = os.path.abspath(os.getcwd())
MaxLen=50

farsi_tokenizer_path = loc+r'\farsi_tokenizer.pkl'
english_tokenizer_path = loc+r'\english_tokenizer.pkl'
model_path =  loc+r'\name_translation_model-kfold-10.h5'
with open(farsi_tokenizer_path, 'rb') as f:
    farsi_tokenizer = pickle.load(f)
        
with open(english_tokenizer_path, 'rb') as f:
    english_tokenizer = pickle.load(f)
        
from keras.models import load_model    
model = load_model(model_path)


def predict_name(name):
          
    farsi_seq = farsi_tokenizer.texts_to_sequences(name)
    farsi_pad = pad_sequences(farsi_seq, maxlen=MaxLen, padding='post')        

    prediction = model.predict(farsi_pad)
    predicted_char_index = np.argmax(prediction, axis=2)
    english_text = ''
    for i in predicted_char_index[0]:
        if i == 0:
            continue
        char = english_tokenizer.sequences_to_texts([[i]])[0]
        english_text += char
    return english_text


names= ["نوید","احمد","احسان","امید"]
for name in names:
  print(name, "->", predict_name([name.strip()]))  


