# -*- coding: utf-8 -*-
"""Fa2EnNames-10-Fold.ipynb
"""

import os
import pickle
import numpy as np
import pandas as pd
import locale

MaxLen=50
TotalNames=8541

#I used Colab for train with Dataset.xlsx in the same directory
#If you use local computer for training you can use this code to load the dataset
# loc = os.path.abspath(os.getcwd())
# filepath=loc + "\Dataset.xlsx"

filepath="Dataset.xlsx"
data = pd.read_excel(filepath)

# Get the set of Persian and Eglish characters for use in tokenizer.fit_on_texts later
farsi_chars = sorted(set(''.join(data['Names'][:TotalNames])),key=locale.strxfrm)
english_chars = sorted(set((''.join(data['Translate'][:TotalNames])).lower()))

#Just to test
print(farsi_chars)
print(english_chars)

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Prepare tokenizers
farsi_tokenizer = Tokenizer(char_level=True)
farsi_tokenizer.fit_on_texts(farsi_chars)
english_tokenizer = Tokenizer(char_level=True)
english_tokenizer.fit_on_texts(english_chars)

# Tokenizing and Padding on max lenght of the names
# padding='post' to pad from the bigining of matrix
farsi_seq=farsi_tokenizer.texts_to_sequences(data["Names"][:TotalNames])
farsi_pad = pad_sequences(farsi_seq, maxlen=MaxLen, padding='post')
english_seq=english_tokenizer.texts_to_sequences((data["Translate"][:TotalNames]).str.lower())
english_pad = pad_sequences(english_seq, maxlen=MaxLen, padding='post')

# Save Tokenizers for later recalls
with open('farsi_tokenizer.pkl', 'wb') as f:
    pickle.dump(farsi_tokenizer, f)
with open('english_tokenizer.pkl', 'wb') as f:
    pickle.dump(english_tokenizer, f)

from sklearn.model_selection import train_test_split

# Split the data into train and test sets
farsi_train, farsi_test, english_train, english_test = train_test_split(farsi_pad, english_pad, test_size=0.1, random_state=42)

from keras.models import Sequential,Model
from keras.layers import Input,LSTM, Dense, RepeatVector, TimeDistributed,Dropout,Embedding,Bidirectional, Flatten, Attention, Lambda
from keras.optimizers import Adam

model = Sequential()
model.add(Embedding(len(farsi_chars)+1, 128, input_length=MaxLen))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.05))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(128, activation='relu'))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(128, activation='relu'))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dense(len(english_chars)+1, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

from sklearn.model_selection import KFold

k = 10 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(farsi_pad, english_pad)):
    print(f"Fold {i+1}...")
    farsi_train, farsi_test = farsi_pad[train_index], farsi_pad[test_index]
    english_train, english_test = english_pad[train_index], english_pad[test_index]
    model.fit(farsi_train, to_categorical(english_train, num_classes=len(english_chars)+1),
              validation_data=(farsi_test, to_categorical(english_test, num_classes=len(english_chars)+1)),
              epochs=25, batch_size=50)

    # Evaluate the model on the test data for this fold
    loss, accuracy = model.evaluate(farsi_test, to_categorical(english_test, num_classes=len(english_chars)+1))
    print(f'Fold {i+1} Test Loss:', loss)
    print(f'Fold {i+1} Test Accuracy:', accuracy)

model.save('name_translation_model.h5')

# Now test the model
# I write this function loading all the files
# In this way that it can be used independently if needed
farsi_tokenizer_path = 'farsi_tokenizer.pkl'
english_tokenizer_path = 'english_tokenizer.pkl'
model_path =  'name_translation_model-kfold-10.h5'
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

# Test the predict function

names= ["نوید","احمد","احسان","امید"]
for name in names:
  print(name, "->", predict_name([name.strip()]))