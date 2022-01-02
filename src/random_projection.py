import numpy as np
import string
import json
import time

from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.svm import SVC

class Metrics:
    @staticmethod
    def accuracy(predictions, ground_truth):
        assert len(predictions) == len(ground_truth), "The length of the predictions and the ground truth must be the same"
        return np.mean(predictions == ground_truth)

def preprocessing(data):
    preprocessed_data = []

    for element in data:
        element = element.lower()
        element = "".join([char for char in element if char not in string.punctuation])
        tokenized_element = word_tokenize(element)

        preprocessed_data.append( " ".join(tokenized_element) )

    return preprocessed_data

def feature_matrix(data):
    matrix = CountVectorizer()

    return matrix.fit_transform(data).toarray()

def random_projection(X, epsilon=0.1):
    n, m = np.shape(X)[0], np.shape(X)[1]

    p = round( np.log(n) / np.square(epsilon) )
    R = np.random.standard_normal( size=(m, p) )

    return ( 1 / np.sqrt(p) ) * np.matmul(X, R)

def main():
    with open("workdir/clinc150_uci/data_full.json") as file:
        data = json.load(file)

    train_X_data, train_y_data = map( list, zip(*data["train"]) )
    test_X_data, test_y_data = map( list, zip(*data["test"]) )

    train_X_data, train_y_data = shuffle(train_X_data, train_y_data)
    test_X_data, test_y_data = shuffle(test_X_data, test_y_data)

    train_X_n = np.shape(train_X_data)[0]
    train_y_n = np.shape(train_y_data)[0]

    data_X = np.concatenate( (train_X_data, test_X_data), axis=0 )
    data_preprocessed = preprocessing(data_X)
    data_matrix = feature_matrix(data_preprocessed)
    rp_matrix = random_projection(data_matrix, 0.1)

    test_X = rp_matrix[train_X_n:]
    train_X = np.delete(rp_matrix, np.s_[train_X_n:], 0)

    data_y = np.concatenate( (train_y_data, test_y_data) )

    label_encoder = LabelEncoder()
    label_encoder.fit(data_y)
    data_y_encoded = label_encoder.transform(data_y)

    test_y = data_y_encoded[train_y_n:]
    train_y = np.delete(data_y_encoded, np.s_[train_y_n:], 0)

    start = time.time()
    svc = SVC(kernel="linear", C=100).fit(train_X, train_y)
    total = time.time() - start
    print(f"Training time: {total}")

    predictions = svc.predict(test_X)
    accuracy = Metrics.accuracy(predictions, test_y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
