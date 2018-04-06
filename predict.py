# The file `data.csv` contains data about users.
#
# A example of each instance is:
#
# K29YU9JIX889R5XW5LMW, 0.108964741365, -1.25539868788, 1.3488876352, -1.11885608548, 0.0345592062495, mat sat, 1
#
# The first feature entry is the entry id, the next entries are continuous features and the number 6 is string based.
#
# For the string based I create a dictionary with all vocabulary, there are 4 different words: "cat", "dog", "mat", "sat"
#
# Since is a sentence I choose a LSTM to encode the sequential data into a vector 
# The remaining features are combined via a dense layer with the output of the LSTM. 
#
# For testing the performance we choose accuracy over the binary output and check via crossfolding.
# Cross Folding split the corpus in n parts, trains the model with n-1 and evaluates with the remaining part. Done n times gives a good prediction of the generalization.
#
#
import csv
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, merge, Input, concatenate
from keras.utils import to_categorical
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sys
from sklearn.model_selection import StratifiedKFold
import numpy

# keras Tokenizer
tokenizer = Tokenizer(num_words=100)

#Features arrays
continuous_features = []
word_features = []
ids = []
tags = []

with open('data.csv', 'rb') as csvfile:
	f = csv.reader(csvfile, delimiter=',')
	next(f, None)  # skip the headers
	for array in f:
		# each id is appended to the corresponding array.
		ids.append(array[0])
		# The continuous features are parsed into float values. For those features which are missing are filled with 0.0. 
		continuous_features.append([ 0.0 if not a.strip() else float(a) for a in array[1:6] ]) 
		# The string is appended to the word_feature array.
		word_features.append(array[6])
		# The binary feature is the annotation tag. 
		tags.append(int(array[7]))

# tokenizer reads all the word_features and creates a dictionary of the present words, using space as separator 
tokenizer.fit_on_texts(word_features)

#Dictionary size
dict_size = len(tokenizer.word_index)

#For each string this generates an array of indexes where each index correspond to the word index in the dictionary.
sequences = tokenizer.texts_to_sequences(word_features)

#An LSTM need all the sequences with the same length. This unify all the sequences adding 0 at the begining of the sequence.
#the longuest sequence hast 16 tokens.
sequences = pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0)

#An LSTM needs a categorical input, not the index but a one-hot representation of the word
sequences = [to_categorical(x, num_classes=dict_size+1) for x in sequences]

#Since the continuous features and the word features are readed in separated inputs, the model in keras needs an array with the two inputs as numpy arrays.
X = [ np.array(continuous_features), np.array(sequences) ]
#Also the output is expected as a numpy array
Y = np.array(tags)

# Creation of the model model
model = Sequential()

#Continuous features input 
c_input = Input(shape=(5,), name='continuous')
#A dense layer proyect the 5 dimension input space via a non-linear function to a 32 dimenstions state.
dense_out = Dense(32, activation='sigmoid')(c_input)

#Word features input. The shape of the input is given by the sequence lengg
w_input = Input(shape=(len(sequences[0]), dict_size+1) , name='word_features')
#An LSTM reads the sequence and outpus a vector of dimension 32, the same dimension of the state given by the continuous features.
lstm_out = LSTM(32) (w_input)

#The two states are concatenated and combined via a dense layer with the same dimensionality.
hidden = concatenate([dense_out, lstm_out])
hidden = Dense(64, activation='sigmoid')(hidden)

#The state is proyected into a single output for predicting the binary feature.
output = Dense(1, activation='sigmoid', name='out')(hidden)

#Putting all the layers togheter.
model = Model(inputs=[c_input, w_input], outputs=[output])
print(model.summary())
#Crossentropy is the loss function used for categorical outputs. Accuracy is the metric to ve evaluated in the test. The optimizer is standard.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Definition of Cross-fold validation. It shuffles and splits the data into 10 chunks.
kfold = StratifiedKFold(n_splits=10, shuffle=True)

#keeping random initialization
Wsave = model.get_weights()

cvscores = []
#Kfold generated a set of indexed corresponding to the i-fold for training and for testing.
for train, test in kfold.split(X[0], Y):
	
	model.set_weights(Wsave)
	#Training the model with the i-fold. The [X[0][train],X[1][train]] is needed because Keras expects an array of inputs. 
	#Since there are 2 inputs, the subarrays are getted, and the array of inputs is recreated
	model.fit([X[0][train],X[1][train]], Y[train], epochs=10,  batch_size=128)
	#Evaluation. Checks the predicted output given the model with the actual binary feature value.
	scores = model.evaluate([X[0][test],X[1][test]], Y[test], verbose=0)
	#Prints the value
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	#Saves the value for averaging the n runs
	cvscores.append(scores[1] * 100)

#Averaged output and standard deviation.
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))




