# coding: utf-8

# Modules
import hazm as hz
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import xml.etree.ElementTree as et
from os import listdir
from os.path import isfile, join
from collections import defaultdict


# Parameters
normalizer = hz.Normalizer()
tagger = hz.POSTagger(model='resources/postagger.model')
stemmer = hz.Stemmer()
lemmatizer = hz.Lemmatizer()

lexicon_file_name = 'final_lexi'
data_path = './data/'

lexicon = None


# Make bag_of_words
def bow(text):
    global normalizer
    global tagger
    global stemmer
    global lemmatizer
    
    text = hz.sent_tokenize(normalizer.normalize(text))

    tagged = [tagger.tag(hz.word_tokenize(sent)) for sent in text]

    bag_of_words = defaultdict(int)
    for sentence in tagged:
        words=[lemmatizer.lemmatize(w[0]).split('#')[0] if  w[1] is 'V' 
               else stemmer.stem(str(w[0])) for w in sentence]
        for w in words: bag_of_words[w]+=1

    return bag_of_words


# Read lexicon from file
def read_lexicon(lexicon_file_name=lexicon_file_name):
    with open(lexicon_file_name) as f:
        return [x.strip('\n') for x in f.readlines()]


# Making the lexicon by read_lexicon function
lexicon = read_lexicon()


# Genarate hot_array
def hot_array(bag, lexi=lexicon):
    return list([bag[w] if w in bag else 0 for w in lexi])


# Read dataset
def read_dataset(data_path=data_path):
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    data_set_list=[]
    value_list = []
    for f in files:
        root = et.parse(data_path + f).getroot()
        for e in root.find('Review').findall('Sentence'):
            if e.attrib.get('Value') is '':continue
            data_set_list.append(e.text)
            value_list.append(int(e.attrib.get('Value')))
    return data_set_list,value_list

# Make a list of hot_arrays
def hot_array_list(dataset_list):
    global lexicon
    return [hot_array(bow(i[0])) for i in dataset_list]


# Map hot_array as 0 => 1 and x => x * 10
def map_hot_array(hot_array):
    return list(map(lambda x: 1 if x is 0 else x * 10, hot_array))


def prepare_data(data_path=data_path):
    dataset,value_list = read_dataset()
    data=np.array([map_hot_array(hot_array(bow(data))) for data in dataset]).astype('float32')
    value=keras.utils.to_categorical(np.array(value_list), num_classes = 5)
    return data,value


x_train, y_train = prepare_data()


model = Sequential()
model.add(Dense(1000, activation='relu', input_dim = len(lexicon)))
model.add(Dropout(.35))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(.35))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(.35))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])


model.fit(x_train, y_train,
         epochs=30,
         batch_size=2,
         validation_split=0.2)

