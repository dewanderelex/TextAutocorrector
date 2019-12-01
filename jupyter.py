# %%

"""
    @author: Alex Nguyen
    @School: Gettysburg College class of 2022
    Jupyter notebook file. Logical, critical thought process and codes.
"""

# %%

#---------------------------------------- Import Library -------------------------------------#

# OS, IO
import os, sys, shutil

# Math Library
import numpy as np
import pandas as pd
import random
import itertools

# Display library
import matplotlib.pyplot as plt
plt.interactive(True)

# Data Preprocessing
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Deep Learning Library
from keras.models import Model, Sequential
from keras.layers import Input, RNN, LSTM, BatchNormalization, TimeDistributed, GRU, Dense, Flatten, merge, Bidirectional
from keras.layers import Embedding
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.utils import Sequence
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint


# %%

#------------------------------------------ Constants ------------------------------------------#

TEST_DATA_Story = ['This is the first time I drink beer.',
                'I love drinking beer.',
                'Do you want to drink beer?',
                'I work very hard so that I can pay for my tuition fee.',
                'I want to get a scholarship from school.',
                'The school offer me a scholarship with the value of $2000.',
                'Do you want to spend that scholarship to go and drink some beer?',
                'I think I will not spend my money on such an useless thing like drinking beer!',
                'You are right, I love to talk to you because you make me feel like I am important.',
                'Call me when you need somebody to talk to!',
                'Yes, I will call you when I want to talk to you!']

TEST_DATA_Attitude = ['I love this movie',
                        'I do not like this movie',
                        'This movie sucks',
                        'This is the worst movie in my life',
                        'This is the best movie in my life',
                        'This movie could have been better',
                        'I do not like the ending of the movie',
                        'I love it!',
                        'This movie is actually good.',
                        'Well, no!',
                        'This sucks!',
                        'This movie is wonderful!',
                        'How wonderful this movie is']
TEST_DATA_Attitude_labels = np.asarray([1,0,0,0,1,0,0,1,1,0,0,1,1])
TO_BE_PREDICTED = ['This is a wonderful movie!']


# %%


#------------------------------Load Data ---------------------------------#

def load_glove(glove_file):
    '''
    Parse a glove text file
    :param glove_file: (String) Glove file name destination.
    :return (Dict): A python glove wordmap vertor
    '''
    glove_vector = {}
    with open(glove_file, "r", encoding='utf8') as glove:
        for line in glove:
            name, vector = tuple(line.split(" ", 1))
            glove_vector[name] = np.fromstring(vector, sep=" ")
    return glove_vector

def sentence_to_vector(glove_vector, sentence):
    '''
    Convert sentence with each word into a vector based on glove wordmap matrix.
    :param glove_vector: (Dict) Dictionary of glove word and vector.
    :param sentence: (String) A sentence of to be converted.
    :return (list): 1D list of tokens
    :return (list): 2D list of associated vectors.
    '''

    tokens = sentence.strip('"(),-').lower().split(" ")
    
    words = []
    word_vectors = []

    for token in tokens:
        n = len(token)

        while len(token) > 0:
            word = token[:n]
            
            if word in glove_vector:
                words.append(word)
                word_vectors.append(glove_vector[word])
                token = token[n:]
                n = len(token)
                continue
            else:
                n -= 1
            
            if n == 0 : #and len(token) > 0:
                # TODO: To implement unknown word
                break
            # elif n == 0:
            #     print('done')
            #     break
    return words, word_vectors

def visualize_sentence(sentence):
    '''
    Visualize sentence on image
    :param sentence: (String) Sentence to visualize
    '''
    # TODO: To be implemented
    raise NotImplementedError

def load_data(glove_vector, dataset_file):
    '''
    Take a dataset file and parse data to return context and answer vector.
    :param dataset_file: (String) Path of the file.
    :return (list): List of tuple of data:
        (list): Tuple of words context in string form.
        (list): Tuple of words context in vector form.
        (list): Tuple of words question in string form.
        (list): Tuple of words question in vector form.
        (list): Tuple of words answer in string form.
        (list): Tuple of words answer in vector form.
        (list): Tuple of supporting statement in number format.
    '''

    data = []
    

    with open(dataset_file, "r", encoding='utf8') as dataset:

        context = []
        for line in dataset:

            line_number, line_data = tuple(line.split(" ", 1))

            # Number 1 for starting a new context, therefore reset context
            if line_number == '1':
                context = []

            if '\t' in line_data:
                question, answer, support = tuple(line_data.split('\t'))
                data.append((tuple(zip(*context))
                    +sentence_to_vector(glove_vector, question)
                    +sentence_to_vector(glove_vector, answer)
                    +([int(sup) for sup in support.split()],)))
            else:
                context.append(sentence_to_vector(glove_vector, line_data))

    return data

def finalize(data):
    '''
    Prepare x_train and y_train based on load_data's data
    :param data:
    :return ():
    '''
    # Code taken from https://www.oreilly.com/ideas/question-answering-with-tensorflow

    final_data = []
    for cqas in data:
        contextvs, contextws, qvs, qws, avs, aws, spt = cqas

        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs) # flatten array -> ndarray
        context_words = sum(contextws,[]) # flatten array -> list

        # Location markers for the beginnings of new sentences.
        sentence_ends = np.array(list(lengths)) 
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
    return np.array(final_data)

def tokenize(data):
    '''
    :param data: (ndarray) 1D array of sentences
    :return: (ndarray) 2D array of words
    :return: (Tokenizer) Tokenizer 
    '''

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    x = tokenizer.texts_to_sequences(data)

    return x, tokenizer

def pad_matrix(list_matrix, sentence_length=None):
    '''
    :param matrix: (list) 2D list of word indices, which is not padded
    :return: (ndarray) 2D array of padded array
    '''
    return np.asarray(pad_sequences(list_matrix, padding='post', maxlen=sentence_length))

def get_sequence_length(array_matrix):
    '''
    :param matrix: (ndarray) 2D array of word indices
    :return: (Integer) Length of a sentences
    '''
    return array_matrix.shape[1]

def get_vocab_size(tokenizer):
    '''
    :param matrix: (ndarray) 2D array of word indices
    # :param tokenizer: (Tokenizer) the actual tokenizer
    :return: (Integer) Vocabulary size
    '''
    # return len(set(array_matrix.flatten())) - 1
    return len(tokenizer.word_index) + 1


# %%


#---------------------------------------------- Parepare Data -----------------------------------------#




# %%


#----------------------------------------- Models ----------------------------------#

def biRNN(shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=shape, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model


# %%





# %%

