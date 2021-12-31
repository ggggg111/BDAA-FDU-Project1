import string
import json

from nltk import ngrams
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

def preprocessing(data):
    preprocessed_data = []
    
    for element in data:
        element = element.lower()
        element = "".join([char for char in element if char not in string.punctuation])
        tokenized_element = word_tokenize(element)
        
        preprocessed_data.append(tokenized_element)
        
    return preprocessed_data

def main():
    with open("workdir/clinc150_uci/data_full.json") as file:
        data = json.load(file)

    train_X_data, train_y_data = map( list, zip(*data["train"]) )
    test_X_data, test_y_data = map( list, zip(*data["test"]) )
    
    train_X = preprocessing(train_X_data)
    test_X = preprocessing(test_X_data)

if __name__ == "__main__":
    main()