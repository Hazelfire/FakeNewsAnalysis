from absl import logging
import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow_datasets as tfds

def gen_encoder():
    dt = pd.read_csv("data/textonly.csv")

    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        dt["text"], target_vocab_size=2**15)
    encoder.save_to_file("encoders/tweettextencoder")

def main():

    percentage = 0.7
    dt = pd.read_csv("data/newtextonly.csv")
    dt = dt.sample(frac=1)
    y_values = dt["is_fake"].values
    y_len = len(y_values)
    y_train = y_values[:int(y_len*percentage)]
    y_test = y_values[int(y_len*percentage):]
    print((y_test.tolist()))
    print((y_train.tolist()))

    encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoders/tweettextencoder")

    x_values = np.array(dt.drop(columns=["is_fake"]))
    x_len = len(x_values)
    x_train = x_values[:int(x_len*percentage)]
    x_train = np.array([encoder.encode(row[0]) for row in x_train])
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,value=0,padding='post',maxlen=256)
    x_test = x_values[int(x_len*percentage):]
    x_test = np.array([encoder.encode(row[0]) for row in x_test])
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,value=0,padding='post',maxlen=256)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(2**15,16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    model.fit(x_train, y_train, epochs=10)
    metrics = model.evaluate(x_test, y_test)
    loss = metrics[0]
    accuracy = metrics[1]
    precision = metrics[2]
    recall = metrics[3]
    F1score = 2 * (recall * precision) / (recall + precision)

    with open("results.csv", "a") as f:
        f.write("rnn,{},{},{},{},{}\n".format(accuracy, precision, recall, F1score, percentage))

    print("Report:\nAccuracy: {}%\nPrecision: {}%\nRecall: {}% \nF1 Score: {}%".format(accuracy * 100, precision * 100, recall * 100, F1score * 100))
    
main()
