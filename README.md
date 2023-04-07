# FarsiNames2English
 A model based on Keras and LSTM to convert names written in Farsi to English writing

Title
Farsi Names to English

Introduction

This is a training model based on Keras and LSTM to convert Persian nouns to English.
It is not possible to convert Persian names to English by algorithmic method. Either names should be fetched from a translated database or an intelligent model should be generated to convert names.
For example Persian names like امید , احمد, احسان start with the letter "ا" in Farsi. 
But their English writings : Omid, Ahmad, Ehsan starts with different English letters.
The purpose of this project is to train an LSTM model using a dataset of Persian words (mostly nouns) so that it can convert the Persian text of nouns into English.


Limitations

This model is only a training model and may produce inaccurate outputs.
I have provided the source code of the model and dataset used so that those interested can develop the model or dataset.


Requirements

Tensorflow needs to be installed
pip install tensorflow


Usage

To train the model you need to use fa2ennames_10_fold.py and dataset.xlsx
To convert the names, there are required codes in the Pinglisher.py file. The following fils also need:
english_tokenizer.pkl
farsi_tokenizer.pkl
name_translation_model-kfold-10.h5


Contact

Please Contact me on navid.basseri@outlook.com


Thanks
Thanks to google Colab and Kaggle for proccessing facilities...
