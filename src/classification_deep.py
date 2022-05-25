"""
Classification using deep learning
"""
""" Import relevant packages """
 # system tools
import os
import sys
 # argument parser
import argparse
 # simple text processing tools
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
 # data wranling
import pandas as pd
import numpy as np
 # tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,
                                     Flatten,
                                     Conv1D,
                                     MaxPooling1D,
                                     Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2
import tensorflow as tf
 # scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
 # visualisations 
import matplotlib.pyplot as plt


""" Basic functions """     
# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # plot name argument
    ap.add_argument("-p",
                    "--plot_name",
                    default="deep_history_plot",
                    help="The name you wish to save the history plot under")
    # report name argument
    ap.add_argument("-r", 
                    "--report_name", 
                    default="deep_classification_report", 
                    help="The name you wish to save the classification report under")
    # epochs argument
    ap.add_argument("-e",
                    "--epochs",
                    type=int,
                    default=5,
                    help = "The number of epochs the model runs for")
    # embedding size argument
    ap.add_argument("-m",
                    "--embed_size",
                    type=int,
                    default=300,
                    help="The number of dimensions for embeddings")
    # batch size argument
    ap.add_argument("-b",
                    "--batch_size",
                    type=int,
                    default=128,
                    help="Size of batches the data is processed by")
    args = vars(ap.parse_args())
    return args

# Function for saving the classification report    
def report_to_txt(report, report_name, epochs, embed_size, batch_size):
    # define outpath
    outpath = os.path.join("out", "deep", f"{report_name}.txt")
    # write txt
    with open(outpath,"w") as file:
        # write headings
        file.write(f"Classification report\nEpochs: {epochs}\nEmbedding dimensions: {embed_size}\nBatch size: {batch_size}\n")
        # write report
        file.write(str(report))    

# Function for saving the history plot        
def save_history(H, epochs, plot_name):
    outpath = os.path.join("out", "deep", f"{plot_name}.png")
    plt.style.use("seaborn-colorblind")
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f"History for CIFAR_10 trained on VGG16", fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Validation", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="Validation", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))

# Helper functions for text processing (made by Ross)
 # strip HTML tags
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text
 # remove accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
 # pre-process corpus function
def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
    return norm_docs

""" Classification function """    
def classification(epochs, embed_size, batch_size):
    # load and read the data
    filename = os.path.join("in", "toxic", "VideoCommentsThreatCorpus.csv")
    dataset = pd.read_csv(filename)
    # manually split the data
    X = dataset["text"].values
    y = dataset["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    # clean and norminalise data (using Ross' helper function)
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    # define out-of-vobabulary token
    t = Tokenizer(oov_token = "<UNK>")
    # fit the tokeniser on the document
    t.fit_on_texts(X_train_norm)
    # set padding value – if the kernel goes outside the matrix, we ensure that the values are still the same length
    t.word_index["<PAD>"] = 0
    # turns the texts into a list of integer values
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    # sequence normalisation
    MAX_SEQUENCE_LENGTH = 100
    # add padding to sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH)    
    # define parameters for model
     # overall vocabulary size
    VOCAB_SIZE = len(t.word_index)
     # number of dimensions for embeddings
    EMBED_SIZE = embed_size
     # number of epochs to train for
    EPOCHS = epochs
     # batch size for training
    BATCH_SIZE = batch_size
    # clear models in memory
    tf.keras.backend.clear_session()
    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))
    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                     kernel_size=4, 
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    # train model
    H = model.fit(X_train_pad, y_train,
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE,
                  validation_split = 0.1, # takes the remaining 10% of the training data, after having trained on 90%
                  verbose = True)
    # final evaluation of the model
    scores = model.evaluate(X_test_pad, y_test, verbose = 1)
    print(f"Accuracy: {scores[1]}")
    # do predictions with 0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    # make classification report
    labels = ["non-toxic", "toxic"]
    report = classification_report(y_test, predictions, target_names = labels)
    # return report for saving and history for plotting
    return report, H
          
""" Main function """        
def main():
    # parse arguments
    args = parse_args()
    # get parameters
    report_name = args["report_name"]
    epochs = args["epochs"]
    embed_size = args["embed_size"]
    batch_size = args["batch_size"]
    plot_name = args["plot_name"]
    # fix random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    # do classification
    report, H = classification(epochs, embed_size, batch_size)
    # save report
    report_to_txt(report, report_name, epochs, embed_size, batch_size)
    # save history plot
    save_history(H, epochs, plot_name)
    # print report
    return print(report)

if __name__ == "__main__":
    main()