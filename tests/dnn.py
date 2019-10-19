import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

def gen_encoder():
    dt = pd.read_csv("data/textonly.csv")

    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        dt["text"], target_vocab_size=2**15)
    encoder.save_to_file("encoders/tweettextencoder")

def check_weights():
    model = tf.keras.models.load_model('cnnmodel.h5')
    print(model.layers[-1].get_weights())

def test_model():
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoders/tweettextencoder")
    model = tf.keras.models.load_model('cnnmodel.h5')


def try_model():
    encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoders/tweettextencoder")
    dt = pd.read_csv("data/newtextonly.csv")
    model = tf.keras.models.load_model('cnnmodel.h5')
    
    while True:
        line = input()
        print(model.predict([encoder.encode(line)]))

#    test = np.array([encoder.encode(row[0]) for row in dt])
#    test = tf.keras.preprocessing.sequence.pad_sequences(test,value=0,padding='post',maxlen=256)

    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functor = K.function([inp, K.learning_phase()], outputs)

    
#    res = dt.sample()
#    out = functor([test, 1.])
    #print(len(out))
    #print(len(out[-2]))
    #for arr in out[-2]:
    #    print(len(arr))
    #    print(len(arr[0]))
    if True:
        res = dt.sample(1000)
        test = np.array([encoder.encode(row) for row in res["text"].values])
        test = tf.keras.preprocessing.sequence.pad_sequences(test,value=0,padding='post',maxlen=256)
        result = list(zip(functor([test, 1.])[-2], res["text"].values, model.predict(test)))
        #print(len(result))
        #print(len(result[99]))
        #print(len(result[99][0]))
        #print(len(layer_outs[0][0][0]))


        def sort_by_attribute(i):
            def wraps(a):
                return a[0][i]
            return wraps

        
        results = []
        for i in range(16):
            ressorted = sorted(result, key=sort_by_attribute(i))
            results.append([(case,sort_by_attribute(i)(case)) for case in ressorted])

        printable = pd.DataFrame(columns=[str("attr" + str(i)) for i in range(16)])
        for i in range(len(res.index)):
            for key, result in enumerate(results):
                printable.loc[i] = [str(result[i][0][1]) + " ({})".format(result[i][1]) for result in results]
        print(printable)
        printable.to_csv("out.csv")
        # Get the second last layer, the dense layer with 16 outputs



def main():

    percentage = 0.7
    frac = 1
    if True:
        dt = pd.read_csv("data/newtextonly.csv")
        dt = dt.sample(frac=frac)
        y_values = dt["is_fake"].values
        y_len = len(y_values)
        y_train = y_values[:int(y_len*percentage)]
        y_test = y_values[int(y_len*percentage):]

        encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoders/tweettextencoder")

        x_values = dt["text"].values
        x_len = len(x_values)
        x_train = x_values[:int(x_len*percentage)]
        x_train = np.array([encoder.encode(row) for row in x_train])
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,value=0,padding='post',maxlen=256)
        x_test = x_values[int(x_len*percentage):]
        x_test = np.array([encoder.encode(row) for row in x_test])
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,value=0,padding='post',maxlen=256)
    else:
        dttrain = pd.read_csv("data/splittextonlytrain.csv")
        dttest = pd.read_csv("data/splittextonlytest.csv")
        dttrain = dttrain.sample(frac=frac)
        dttest = dttest.sample(frac=frac)
        y_train = dttrain["is_fake"].values
        y_test = dttest["is_fake"].values

        x_train = dttrain["text"].values
        x_test = dttest["text"].values

        encoder = tfds.features.text.SubwordTextEncoder.load_from_file("encoders/tweettextencoder")

        x_train = np.array([encoder.encode(row) for row in x_train])
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,value=0,padding='post',maxlen=256)

        x_test = np.array([encoder.encode(row) for row in x_test])
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,value=0,padding='post',maxlen=256)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(2**15,16,input_length=256),
#        tf.keras.layers.Conv1D(32, kernel_size=5,activation=tf.nn.relu),
        tf.keras.layers.GlobalMaxPooling1D(),
#        tf.keras.layers.Conv1D(64, kernel_size=5,activation=tf.nn.relu),
#        tf.keras.layers.GlobalMaxPooling1D(),
#        tf.keras.layers.Conv1D(64, kernel_size=5,activation=tf.nn.relu),
#        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    model.fit(x_train, y_train, epochs=10)
    model.save("cnnmodel2.h5");
    metrics = model.evaluate(x_test, y_test)
    loss = metrics[0]
    accuracy = metrics[1]
    precision = metrics[2]
    recall = metrics[3]
    F1score = 2 * (recall * precision) / (recall + precision)

    print("Report:\nAccuracy: {}%\nPrecision: {}%\nRecall: {}% \nF1 Score: {}%".format(accuracy * 100, precision * 100, recall * 100, F1score * 100))

    with open("results.csv", "a") as f:
        f.write("dnn,{},{},{},{},{}\n".format(accuracy, precision, recall, F1score, percentage, len(dttrain.index) + len(dttest.index)))

    
#gen_encoder()
main()
#try_model()
#check_weights()
