import numpy as np
import string
import json

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
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
    matrix_fit = matrix.fit_transform(data)

    return matrix_fit.toarray()

def random_projection(X, epsilon=0.1):
    n, m = np.shape(X)[0], np.shape(X)[1]

    p = round( np.log(n) / np.square(epsilon) )
    R = np.random.standard_normal( size=(m, p) )

    return ( 1 / np.sqrt(p) ) * np.matmul(X, R)

def main():
    with open("workdir/clinc150_uci/data_full.json") as file:
        data = json.load(file)

    train_X_data, train_y = map( list, zip(*data["train"]) )
    test_X_data, test_y = map( list, zip(*data["test"]) )

    train_X_n = np.shape(train_X_data)[0]
    test_X_n = np.shape(test_X_data)[0]

    data = np.concatenate( (train_X_data, test_X_data), axis=0 )
    data_preprocessed = preprocessing(data)
    data_matrix = feature_matrix(data_preprocessed)
    rp_matrix = random_projection(data_matrix, 0.1)

    test_X = rp_matrix[train_X_n:]
    train_X = np.delete(rp_matrix, np.s_[train_X_n:], 0)

    svc = SVC(kernel="linear", C=100).fit(train_X, train_y)
    predictions = svc.predict(test_X)
    accuracy = Metrics.accuracy(predictions, test_y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()