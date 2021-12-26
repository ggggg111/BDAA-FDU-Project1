import nltk
import json
import numpy as np

def main():
    with open("workdir/clinc150_uci/data_full.json") as f:
        data = json.load(f)

    train_X, train_y = map( list, zip(*data["train"]) )
    test_X, test_y = map( list, zip(*data["test"]) )

    print( len(train_X) )
    print( len(train_y) )

    print( len(test_X) )
    print( len(test_y) )

if __name__ == "__main__":
    main()