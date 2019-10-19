from absl import logging
import tensorflow as tf
import numpy as np
import pandas as pd

def main():

    dt = pd.read_csv("data/normnotext.csv")
    dt = dt.sample(frac=1)
    y_values = dt["is.fake"].values
    y_len = len(y_values)
    y_train = y_values[:int(y_len*0.9)]
    y_test = y_values[int(y_len*0.9):]
    print((y_test.tolist()))
    print((y_train.tolist()))

    x_values = np.array(dt.drop(columns=["is.fake"]))
    x_len = len(x_values)
    x_train = x_values[:int(x_len*0.9)]
    x_test = x_values[int(x_len*0.9):]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    model.fit(x_train, y_train, epochs=40)
    metrics = model.evaluate(x_test, y_test)
    loss = metrics[0]
    accuracy = metrics[1]
    precision = metrics[2]
    recall = metrics[3]
    F1score = 2 * (recall * precision) / (recall + precision)
    
    with open("results.csv", "a") as f:
        f.write("feedforward,{},{},{},{}\n".format(accuracy, precision, recall, F1score))
    
    print("Report:\nAccuracy: {}%\nPrecision: {}%\nRecall: {}% \nF1 Score: {}%".format(accuracy * 100, precision * 100, recall * 100, F1score * 100))
    

main()
