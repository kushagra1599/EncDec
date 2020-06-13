import collections
import reader
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU,RepeatVector,TimeDistributed,Dense
from keras.models import Model,Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

eng_sen=reader.load_data('data/small_vocab_en')
fre_sen=reader.load_data('data/small_vocab_fr')

def tokenize(x):
    x_tk=Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x),x_tk

def pad(x,length=None):
    if(length is None):
        length=max([len(sentence for sentence in x)])
    return pad_sequences(x,maxlen=length,padding="post")

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    return preprocess_x,preprocess_y,x_tk,y_tk
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =preprocess(eng_sen, fre_sen)

english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def encdec_model(input_shape,output_sequence,english_vocab_size,french_vocab_size):
    learning_rate=1e-3
    model=Sequential()
    model.add(GRU(128,input_shape=input_shape[1:],return_sequences=False))
    model.add(RepeatVector(output_sequence))
    model.add(GRU(128,return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(learning_rate),metrics=['accuracy'])
    return model


tmp_x=pad(preproc_english_sentences)
tmp_x=tmp_x.reshape((-1,preproc_english_sentences[1],1))
encodeco_model = encdec_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index)+1,
    len(french_tokenizer.word_index)+1)
encodeco_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
print(logits_to_text(encodeco_model.predict(tmp_x[:1])[0], french_tokenizer))