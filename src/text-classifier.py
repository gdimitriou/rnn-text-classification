import keras
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
# import tqdm as tqdm
# from keras_tqdm import TQDMNotebookCallback

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import codecs
from tqdm import tqdm

from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, GlobalMaxPool1D, Bidirectional, GlobalMaxPooling1D
from keras.layers import LSTM, GRU, Dropout, BatchNormalization, Embedding, Flatten, GlobalAveragePooling1D, \
    concatenate, Input

# Read train data
train = pd.read_csv('../resources/train.csv')
train.dropna(inplace=True)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train[list_classes]

# Read test data
test = pd.read_csv('../resources/test.csv')
test.dropna(inplace=True)

# Create tools to preprocess data
# we will remove english stop words from text as well as punctuation
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

# The maximum number of words considered is 100000
MAX_NB_WORDS = 100000

# The size of the sentences will be 250
max_seq_len = 250

raw_docs_train = train['comment_text'].tolist()
raw_docs_test = test['comment_text'].tolist()

num_classes = len(list_classes)

tokenizer = RegexpTokenizer(r'\w+')

print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))

print("pre-processing test data...")
processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))

print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  # leaky
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

# Pad sequences
# is used to ensure that all sequences in a list have the same length.
# By default this is done by padding 0 in the beginning of each sequence until each
# sequence has the same length as the longest sequence.
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

print("Done !!")

embed_dim = 300

# Load embeddings
# Used the concatenation of two pre - trained embeddings: fastText crawl-300d-2M.vec and glove.840B.300d.txt.
# They must be downloaded!
print('loading first word embeddings...')
embeddings_index = {}
f = codecs.open('../resources/glove.840B.300d.txt', encoding='utf-8')

for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

# Embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix_glove = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_glove[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix_glove, axis=1) == 0))

print('loading second word embeddings...')
EMBEDDING_FILE = '../resources/crawl-300d-2M.vec'


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))

word_index = tokenizer.word_index
embedding_matrix_crawl = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_crawl[i] = embedding_vector

del embeddings_index

# Creating model
inp = Input(shape=(max_seq_len,))
emb_glove = Embedding(nb_words, 300,
                      weights=[embedding_matrix_glove], input_length=max_seq_len, trainable=False)(inp)

emb_crawl = Embedding(nb_words, 300,
                      weights=[embedding_matrix_crawl], input_length=max_seq_len, trainable=False)(inp)

conc1 = concatenate([emb_glove, emb_crawl])
x = Bidirectional(LSTM(400, return_sequences=True))(conc1)
x = Bidirectional(GRU(400, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
out = concatenate([avg_pool, max_pool])

out = Dense(200, activation="relu")(out)
out = Dense(y_train.shape[1], activation="sigmoid")(out)

model = Model(inputs=inp, outputs=out)

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

model.fit(word_seq_train, y_train, epochs=6, batch_size=80, shuffle=True, validation_split=0.1, verbose=2)

y_test = model.predict(word_seq_test)

sample_submission = pd.read_csv("../resources/sample_submission.csv")

sample_submission[list_classes] = y_test
sample_submission.to_csv("../resources/results.csv", index=False)
