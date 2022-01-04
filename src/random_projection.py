import numpy as np
import string
import json
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.svm import SVC

class Metrics:
    @staticmethod
    def accuracy(predictions, ground_truth):
        """Calculate the mean value of the accuracy given the prediction and ground-truth values.

        Args:
            predictions (numpy.ndarray): Prediction values vector.
            ground_truth (numpy.ndarray): Ground-truth values vector.

        Returns:
            numpy.float64: Accuracy values mean.
        """
        assert len(predictions) == len(ground_truth), "The length of the predictions and the ground truth must be the same"

        return np.mean(predictions == ground_truth)

def preprocessing(data):
    """Data preprocessing. Each element is converted into lowercase and punctuation characters are removed.

    Args:
        data (numpy.ndarray): All the standard dataset features.

    Returns:
        list: Each preprocessed feature element of the dataset.
    """
    preprocessed_data = []

    for element in data:
        element = element.lower()
        element = "".join([char for char in element if char not in string.punctuation])

        preprocessed_data.append(element)

    return preprocessed_data

def feature_matrix(data):
    """Create the feature matrix, also known as dictionary in the NLP context. Each feature is an individual token.

    Args:
        data (list): All the preprocessed features of the dataset.

    Returns:
        numpy.ndarray: Word dictionary.
    """
    matrix = CountVectorizer()

    return matrix.fit_transform(data).toarray()

def random_projection(X, epsilon=0.1):
    """Perform a Gaussian Random Projection for dimensionality reduction of the dataset.

    Args:
        X (numpy.ndarray): The features of the dataset.
        epsilon (float, optional): Error threshold, ranging [0, 1]; the higher the value, the more reduction is performed, but the accuracy will reduce. Defaults to 0.1.

    Returns:
        numpy.ndarray: The new features, with its dimensions reduced.
    """
    n, m = np.shape(X)[0], np.shape(X)[1]

    p = round( np.log(n) / np.square(epsilon) )
    R = np.random.standard_normal( size=(m, p) )

    return ( 1 / np.sqrt(p) ) * np.matmul(X, R)

def main():
    # Open dataset file
    with open("workdir/clinc150_uci/data_full.json") as file:
        data = json.load(file)

    # Get both the training and testing data from the dataset
    train_X_data, train_y_data = map( list, zip(*data["train"]) )
    test_X_data, test_y_data = map( list, zip(*data["test"]) )

    # The data is ordered, so it needs to be shuffled
    train_X_data, train_y_data = shuffle(train_X_data, train_y_data)
    test_X_data, test_y_data = shuffle(test_X_data, test_y_data)

    # Get number of elements of the training data
    train_n = np.shape(train_X_data)[0]

    # Join training and testing feature data for feature preprocessing
    data_X = np.concatenate( (train_X_data, test_X_data), axis=0 )
    # Perform data preprocessing in NLP context
    data_preprocessed = preprocessing(data_X)
    # Obtain feature matrix, in order to have trainable and testable data
    data_matrix = feature_matrix(data_preprocessed)
    # Dimensionality reduction using Gaussian Random Projection
    rp_matrix = random_projection(data_matrix, 0.1)

    # Assign both testing and training feature data using the past number of training instances
    test_X = rp_matrix[train_n:]
    train_X = np.delete(rp_matrix, np.s_[train_n:], 0)

    # Join both training and testing label data
    data_y = np.concatenate( (train_y_data, test_y_data) )

    # Encode the labels, so that they are an integer instead of a string
    label_encoder = LabelEncoder()
    label_encoder.fit(data_y)
    data_y_encoded = label_encoder.transform(data_y)

    # Assign both testing and training label data using the past number of training instances
    test_y = data_y_encoded[train_n:]
    train_y = np.delete(data_y_encoded, np.s_[train_n:], 0)

    # Train a SVM model for classification
    start = time.time()
    svc = SVC(kernel="linear", C=100).fit(train_X, train_y)
    total = time.time() - start
    print(f"Training time: {total}")

    # Perform the predictions using the desired model
    predictions = svc.predict(test_X)
    accuracy = Metrics.accuracy(predictions, test_y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
