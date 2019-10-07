
# 'LSTM module '

# __author__ = 'Nicole Wang'

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re, string
import sys
import pickle
import warnings
import matplotlib.pyplot as plt
import h5py
from keras.models import model_from_json

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
import csv

import numpy as np
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing import sequence
np.random.seed(1)
maxlen=24

from stop_words import get_stop_words
stopwords = get_stop_words('en')
lemmatizer = nltk.WordNetLemmatizer()
transtbl = str.maketrans(string.punctuation, ' '*len(string.punctuation))

from util import read_glove_vec
word_to_index, word_to_vec_map = read_glove_vec('C:/Users/nwang/Desktop/nlp/glove.6B.50d.txt')

class LSTMmodel:
    
    def text_clean(self,text):
        if not isinstance(text,float) :
            text=str(text)
            text = ' '.join([appos[we] if we in appos else we for we in text.split()])
            text =text.translate(transtbl)
    #         tokens = [lemmatizer.lemmatize(t.lower(),'v')
    #                  for t in nltk.word_tokenize(text)
    #                  if t.lower() not in stopwords]
            return ' '.join(text.split())
        else:
            return np.nan
    
    
    
    def _pretrained_embedding_layer(self,word_to_vec_map, word_to_index):
        vocab_len = len(word_to_index) + 1  
        emb_dim = list(word_to_vec_map.values())[0].shape[0]


        emb_matrix = np.zeros((vocab_len, emb_dim))

        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]


        return Embedding(
            input_dim=vocab_len, 
            output_dim=emb_dim, 
            trainable=False, 
            weights=[emb_matrix])
    
    def _sentences_to_indice(self, X, word_to_index, max_len):
        m = X.shape[0]
        X_indices = np.zeros((m, max_len))

        for i in range(m):
            sentence_words = X[i].lower().split()
            j = 0
            for w in sentence_words:
                try:
                    X_indices[i, j] = word_to_index[w]
                    j = j + 1
                except:
                    X_indices[i, j] = word_to_index['unk']
                    j = j + 1

        return X_indices
    
    def _convert_to_one_hot(self,Y, C):
        Y = np.eye(C)[Y.reshape(-1)]
        return Y
    
    def _mmodel(self,input_shape, word_to_vec_map, word_to_index):
        # Input layer
        sentence_indices = Input(shape=input_shape, dtype='int32')

        # Embedding layer
        embedding_layer = self._pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)   

        # 2-layer LSTM
        X = LSTM(128, return_sequences=True, recurrent_dropout=0.5)(embeddings)  # N->N RNN
        X = Dropout(0.5)(X)
        X = LSTM(128, recurrent_dropout=0.5)(X)  # N -> 1 RNN
        X = Dropout(0.5)(X)
        X = Dense(4, activation='softmax')(X)

        # Create and return model
        model = Model(inputs=sentence_indices, outputs=X)

        return model
    
    def fit_predict(self, X_train, X_test, Y_train, Y_test, cn=4, epochs=50, batch_size=32,shuffle=True):
        X_train_indices = self._sentences_to_indice(X_train, word_to_index, maxlen)
        X_test_indices = self._sentences_to_indice(X_test, word_to_index, maxlen)
        model = self._mmodel((maxlen,), word_to_vec_map, word_to_index)
        print(model.summary())
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        Y_train_oh = self._convert_to_one_hot(Y_train, C = cn)
        Y_test_oh = self._convert_to_one_hot(Y_test, C = cn)
        history = model.fit(X_train_indices, 
                        Y_train_oh, 
                        validation_split=0.2,
                        epochs = epochs, 
                        batch_size = batch_size, 
                        shuffle=shuffle)
        loss, acc = model.evaluate(X_test_indices, Y_test_oh)
        plt.plot(history.history['loss'])
        plt.plot(history.history['acc'])
        
        
        print('----------------------------------------TEST ACCURACY: '+str(acc))
        with open('C:/Users/nwang/Desktop/nlp/model/lstm_model.json', 'w') as fp:
            fp.write(model.to_json())
        model.save_weights('C:/Users/nwang/Desktop/nlp/model/lstm_model.h5')
    
    def pretrain(self, textarray):
        path1='C:/Users/nwang/Desktop/nlp/model/lstm_model.json'
        path2='C:/Users/nwang/Desktop/nlp/model/lstm_model.h5'
        with open(path1,'r') as fp:
            xmodel=model_from_json(fp.read())
        xmodel.load_weights(path2)

        xmodel.compile(
            loss='categorical_crossentropy', 
            optimizer='adam', 
        #     metrics=[auc])
            metrics=['accuracy'])
        
        one_index= self._sentences_to_indice(textarray, word_to_index, maxlen)
        ar=xmodel.predict(one_index)
        return ar.argmax(axis=0)

    
    
    
appos = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not"

}

