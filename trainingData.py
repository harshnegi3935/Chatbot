# importing relevant libraries
import json
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import gradient_descent_v2
from nltk.stem import WordNetLemmatizer

# loading the Dataset : intents.json
intents = json.loads(open('intents.json').read())

# creating several list for model training
words = []                                  # for bow model/vocabulary for patterns
classes = []                                # for bow model/vocabulary for tags
documents = []                              # for storing each patterns
ignore_letters = ['?', '!', ',', '.']       # ignoring letters during model training

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()

# iterating over all the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)         # tokenize each pattern
        words.extend(word_list)                         # append tokens to words
        documents.append((word_list, intent['tag']))    # appending pattern to documents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])               # appending the associated tag to each pattern

# lemmatize all the words in the vocab and don't add ignore_letter characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# sorting the vocab and classes in alphabetical order and taking the set to ensure no duplicate occur
words = sorted(set(words))
classes = sorted(set(classes))
print(words)
print(classes)

# saving the words and classes in a pickle file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# creating our training data
training = []
# creating an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for document in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    word_patterns = document[0]
    # lemmatize each word - create based word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create our bag of words arrays with 1, if  word match found in each pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # add the one encoded BoW and associated classes to training
    training.append([bag, output_row])

# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)

# create train and test lists. X - patterns ,Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training Data Created")


# Neural Network Model
''' creating model - 3 layers. first layer 128 neurons, second layer 64 neurons
    and 3rd layer contains number of neurons equal to number of intents to 
    predict output intent with softmax '''
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# Compiling Model
''' using Stochastic Gradient Descent with Nesterov accelerated gradient
    gives good result for this model '''
sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting the saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print(model.summary())
print('Model Created')
