import nltk

def main():
    sentence = "hello world"
    tokens = nltk.word_tokenize(sentence)
    print(tokens)

if __name__ == "__main__":
    main()