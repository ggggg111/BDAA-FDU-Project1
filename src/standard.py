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
        """Calculate the mean value given the prediction and ground-truth values.

        Args:
            predictions (numpy.ndarray): [description]
            ground_truth (numpy.ndarray): [description]

        Returns:
            numpy.float64: [description]
        """
        assert len(predictions) == len(ground_truth), "The length of the predictions and the ground truth must be the same"

        return np.mean(predictions == ground_truth)

def preprocessing(data):
    """Data preprocessing. Each element is converted into lowercase, punctuation characters are removed, and tokenized for further preprocessing if desired.

    Args:
        data (numpy.ndarray): All the standard dataset features.

    Returns:
        list: Each preprocessed feature element of the dataset.
    """
    preprocessed_data = []

    for element in data:
        element = element.lower()
        element = "".join([char for char in element if char not in string.punctuation])
        tokenized_element = word_tokenize(element)

        preprocessed_data.append( " ".join(tokenized_element) )

    return preprocessed_data

def feature_matrix(data):
    """Create the feature matrix, also known as dictionary in the NLP context.

    Args:
        data (list): All the preprocessed features of the dataset.

    Returns:
        numpy.ndarray: Word dictionary.
    """
    matrix = CountVectorizer()

    return matrix.fit_transform(data).toarray()

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

    test_X = data_matrix[train_X_n:]
    train_X = np.delete(data_matrix, np.s_[train_X_n:], 0)

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
