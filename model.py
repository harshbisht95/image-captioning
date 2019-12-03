from numpy import array
import os
import numpy as np
from pickle import load
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
 
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
#reduce vocab size
def reduce_vocab_size(descriptions):
        # Create a list of all the training captions
        all_train_captions = []
        for key, val in train_descriptions.items():
                for cap in val:
                        all_train_captions.append(cap)
        # Consider only words which occur at least 10 times in the corpus
        word_count_threshold = 10
        word_counts = {}
        nsents = 0
        for sent in all_train_captions:
                nsents += 1
                for w in sent.split(' '):
                        word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        return vocab

# preprocessed words 1651
# fit a tokenizer given caption descriptions
def create_tokenizer(vocab):
	ixtoword = {}
	wordtoix = {}
	tokens={}
	ix = 1
	for w in vocab:
                wordtoix[w] = ix
                ixtoword[ix] = w
                tokens[w]=ix
                ix += 1
	return tokens, ixtoword, wordtoix
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
'''
# create sequences of images, input sequences and output words for an image
def create_sequences(tokens, max_length, descriptions, photos, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)
'''
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photos[key])
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0
#get glove
def get_glove():
        # Load Glove vectors
        glove_dir = 'C:/Users/Harsh/Desktop/ML/image-captioning/glove6b200d'
        embeddings_index = {} # empty dictionary
        f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()
        return embeddings_index
def embed_matrix(embeddings_index):
        embedding_dim = 200
        # Get 200-dim dense vector for each of the 10000 words in out vocabulary
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in wordtoix.items():
                #if i < max_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                        # Words not found in the embedding index will be all zeros
                        embedding_matrix[i] = embedding_vector
        return embedding_matrix
                
# define the captioning model
def define_model(vocab_size, max_length,embedding_dim,emb_matrix):
	# feature extractor model
	inputs1 = Input(shape=(2048,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.layers[2].set_weights([emb_matrix])
	model.layers[2].trainable = False
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model
 
# train dataset
# evaluate the skill of the model

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



def extract_features1(filename):
    # Get the InceptionV3 model trained on imagenet data
    model = InceptionV3(weights='imagenet')
    # Remove the last layer (output softmax layer) from the inception v3
    model_new = Model(model.input, model.layers[-2].output)
    # extract features from each photo in the directory
    # Convert all the images to size 299x299 as expected by the
    img = image.load_img(filename, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess images using preprocess_input() from inception module
    x = preprocess_input(x)
    # get features
    feature = model_new.predict(x, verbose=0)
    # reshape from (1, 2048) to (2048,)
    #feature = np.reshape(x, x.shape[1])
    # get image id
    print(len(feature))
    return feature


# load training dataset (6K)
filename = 'C:/Users/Harsh/Desktop/ML/DataSet/image-captioning/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))


# prepare tokenizer
tokens, ixtoword, wordtoix = create_tokenizer(reduce_vocab_size(train_descriptions))
vocab_size = len(tokens) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
embeddings_index = get_glove()
embedding_matrix = embed_matrix(embeddings_index)
num_photos_per_batch=3
# define the model
model = define_model(vocab_size, max_length,200,embedding_matrix)
model.fit_generator(data_generator(train_descriptions, train_features, wordtoix, max_length, num_photos_per_batch), steps_per_epoch = 2000, epochs = 10)
# evaluate model

path = "C:/Users/Harsh/Desktop/ML/DataSet/image-captioning/Flickr8k_Dataset/New folder/846085364_fc9d23df46.jpg"

#ph = Image.open('C:\Users\Harsh\Desktop\ML\DataSet\image-captioning\Flickr8k_Dataset\New folder')
print(greedySearch(extract_features1(path)))

model.save_weights('weights.h5')



'''
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
 
# dev dataset
 
# load test set
filename = 'C:/Users/Harsh/Desktop/ML/DataSet/image-captioning/Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)
 
# fit model
 
# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
'''
